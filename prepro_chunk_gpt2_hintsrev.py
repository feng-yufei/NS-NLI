import os
import pickle
import numpy as np
from tqdm import tqdm
import csv
import json
from collections import Counter
import torch
from transformers import GPT2Model, GPT2Tokenizer
from chunking import chunking_sent
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import random
lemmatizer = WordNetLemmatizer()
random.seed(0)
# inputs
root = './preprocess/snli'

# intermediate output
file_encoded = './data/snli.pkl'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = '<pad>'
label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

default_projection = [0, 1, 2, 3, 4, 5, 6]
proj_shuffle = projection_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 0: 6}
quantifiers = {'no', 'at most three', 'less than three', 'few', 'some', 'at least three', 'more than three',
               'a few', 'several', 'many', 'at most ten', 'every', 'all', 'any'}
down_quantifiers = {'no', 'at most three', 'less than three', 'few', 'doesn\'t', 'cannot', 'can\'t', 'n\'t', 'never', 'not', 'every', 'all', 'neither', 'any', 'each'}
rm = {'the', 'a', 'in', 'on', 'is', 'are', 'of', 'to', ''}

def rm_func(phrases):
    words = [x.lower() for x in phrases.split(' ') if x.lower() not in rm]
    return ' '.join(words).strip()

def wn_check(p, q):
    p = lemmatizer.lemmatize(p)
    q = lemmatizer.lemmatize(q)
    #if p == q:
    #    return 'equal'
    p_hyper = set()
    q_hyper = set()
    p_synsets = set(wn.synsets(p))
    #print(p_synsets)
    q_synsets = set(wn.synsets(q))

    entail = False
    r_entail = False
    antonymy = False
    #cohypnym = False

    for syn_1 in p_synsets:
        for syn_2 in q_synsets:
            common = syn_1.lowest_common_hypernyms(syn_2)
            if len(common) > 0:
                for c in common:
                    if c == syn_1:
                        r_entail = True
                    if c == syn_2:
                        entail = True
                    #if c !=syn_1 and c != syn_2:
                    #    cohypnym = True

    if not r_entail and not entail:
        p_antonymy = {}
        q_antonymy = {}
        for t_p in p_synsets:
            for t_q in q_synsets:
                if t_p not in p_antonymy:
                    p_antonymy[t_p] = set()
                    for l in t_p.lemmas():
                        if l.antonyms():
                            p_antonymy[t_p].add(l.antonyms()[0].name())
                if t_q not in q_antonymy:
                    q_antonymy[t_q] = set()
                    for l in t_q.lemmas():
                        if l.antonyms():
                            q_antonymy[t_q].add(l.antonyms()[0].name())
                t_p_token = t_p.name().split('.')[0]
                t_q_token = t_q.name().split('.')[0]
                if t_p_token in q_antonymy[t_q] or t_q_token in p_antonymy[t_p]:
                    # print(p, q)
                    antonymy = True
                    break
            if antonymy:
                break

    rel = 'none'
    if r_entail and not entail:
        rel = 'r_entail'
    if entail and not r_entail:
        rel = 'f_entail'
    if antonymy:
        rel = 'antonymy'

    #if rel == 'none' and cohypnym:
    #    rel = 'cohypnym'
    return rel

def read_from_file(file):
    with open(file, 'rb') as f:
        samples = pickle.load(f)
    return samples


def split_quantifier(phrases):
    new_phrases = []
    for phe in phrases:
        found = False
        for q in down_quantifiers:
            if q in phe.lower() and q in phe.strip().replace('  ', ' ').split(' '):
                assemb = []
                pre = phe.lower().split(q)[0]
                aft = phe.lower().split(q)[1]
                assemb.append(pre + q)
                if aft.strip() != '':
                    assemb.append(aft)
                new_phrases += assemb
                found = True
                break
        if not found:
            new_phrases.append(phe.strip())
    return new_phrases


def tokens_and_polarity(sent_tokens, sent_polarity):
    def convert_tokens(tk):
        if tk.startswith('Ġ'):
            return tk[1:]
        else:
            return '##' + tk
    def permute(s):
        return [proj_shuffle[x] for x in s]

    down = False
    second = False
    fst = 0
    orig_sent = ' '.join(sent_tokens)
    for q in down_quantifiers:
        if q == 'few' and 'a few' in orig_sent:
            continue
        if orig_sent.startswith(q + ' '):
            down=True
            second = True

            fst = len(q.split(' '))
        elif ' ' + q + ' ' in orig_sent.strip():
            down = True
            second = True
            qidx = orig_sent.index(q)

            fst = len(tokenizer.tokenize(orig_sent[:qidx]))
        if q in ['each', 'all', 'any', 'every']:
            second = False
            #print(orig_sent[:qidx], fst)
    gpt2_tokens_m = tokenizer.tokenize(orig_sent)
    gpt2_tokens = [convert_tokens(tk) for tk in gpt2_tokens_m]
    gpt2_tokens[0] = gpt2_tokens[0].replace('##', '')
    gpt2_project = []
    ind = -1
    for tk in gpt2_tokens:
        if not tk.startswith('##'):
            ind += 1
        if ind < len(sent_polarity):
            gpt2_project.append(permute(sent_polarity[ind]))
        else:
            gpt2_project.append(default_projection)

    if gpt2_tokens[-1].replace('##', '') == '.':
        gpt2_tokens_m = gpt2_tokens_m[:-1]
        gpt2_tokens = gpt2_tokens[:-1]
        gpt2_project = gpt2_project[:-1]
    #print('###################')
    #print(gpt2_tokens)

    #print(down)
    #print(len(gpt2_project))
    qtf_proj = -1
    if down:
        qtf_proj = fst - 1
        if  fst < len(gpt2_project) and gpt2_project[fst][1] != 2:
            gpt2_project[fst] = [0, 2, 1, 3, 4, 5, 6]
        for i in range(fst+1, len(gpt2_project)):
            #if second:
            gpt2_project[i] = gpt2_project[fst]

    return gpt2_tokens_m, gpt2_tokens, gpt2_project, qtf_proj

def phrases_index(sent, tokens):
    orig_sent = ' '.join(sent)
    if orig_sent.endswith('.'):
        orig_sent = orig_sent[:-1]
    #print(orig_sent)
    phrases = chunking_sent(orig_sent)

    #print(phrases)
    if phrases[-1] == '':
        phrases = phrases[:-1]
    phrases = split_quantifier(phrases)
    #print('#', phrases)
    #print(phrases)
    phrase_len = [len(p.replace(' ', '')) for p in phrases]
    groups = []

    p_id = 0
    acclen = 0
    phrase_token = []
    #print(phrase_len)
    for i, token in enumerate(tokens):
        acclen += len(token.replace('##', ''))
        phrase_token.append(i)
        #print(token, phrase_token, p_id, acclen)
        if p_id < len(phrase_len) and acclen >= phrase_len[p_id]:
            p_id += 1
            acclen = 0
            groups.append(phrase_token)
            phrase_token = []
    if len(phrase_token) > 0:
        groups.append(phrase_token)
    #print(orig_sent)
    #print(len(phrases), len(groups))
    #print([(p[0], p[-1]) for p in groups])
    #print(tokens)
    #print()
    return [(p[0]+1, p[-1]+1) for p in groups], phrases

def fix_proposal(pm_phrases, hp_phrases, align):

    def get_top_align(tk_pos, pm_phrases, align):
        token2phrase = {}
        scores = [0] * len(pm_phrases)
        curr = 0
        for pid, phrase in enumerate(pm_phrases):
            for tk in phrase.strip().split(' '):
                token2phrase[curr] = pid
                curr += 1
        #print(token2phrase, tk_pos)
        for tk in tk_pos:
            if tk in align.keys() and align[tk] in token2phrase.keys():
                scores[token2phrase[align[tk]]] += 1
        top_index = scores.index(max(scores))
        #print(scores)
        if max(scores) == 0:
            return '---'
        return pm_phrases[top_index]

    def check_entail(p, q):
        rel = set()
        for tkp in p.strip().split(' '):
            for tkq in q.strip().split(' '):
                res = wn_check(tkp, tkq)
                if res == 'f_entail':
                    rel.add(1)
                if res == 'r_entail':
                    rel.add(2)
                if res == 'antonymy':
                    rel.add(4)
        return list(rel)
    #print(pm_phrases)
    #print(hp_phrases)
    #print(align['sureAlign'])
    align = [x for x in align['sureAlign'].split(' ')]
    if len(align) == 1 and align[0] == '':
        align = {}
    else:
        for t in range(len(align)):
            align[t] = tuple([int(x) for x in align[t].split('-')])
        align = {x[0]: x[1] for x in align}

    hints = {}
    top_align_phrases = []
    ctk = 0
    for i, p in enumerate(hp_phrases):
        hints[i] = []
        #if p in pm_phrases:
        #    hints[i] = [0, 1]

        tk_pos = []
        p_tokens = p.strip().split(' ')
        #print(p_tokens)
        for pos in range(len(p_tokens)):
            tk_pos.append(ctk + pos)
        #print(tk_pos)
        top_ph = get_top_align(tk_pos, pm_phrases, align)
        top_align_phrases.append((top_ph, p))
        p_rm = rm_func(p)
        top_ph_rm = rm_func(top_ph)
        if p_rm in top_ph_rm:
            hints[i] += [0, 1]
            #if p_rm.strip() == top_ph_rm.strip():
            #    hints[i] +=[0]
            #else:
            #    hints[i] += [0, 1]
        elif top_ph_rm in p_rm:
            hints[i].append(2)

        if len(hints[i]) > 0:
            #print(p, ' -> ', top_ph, '\t', hints[i])
            ctk += len(p.strip().split(' '))
            continue
        hints[i] += check_entail(top_ph_rm, p_rm)
        #print(p, ' -> ', top_ph, '\t', hints[i])


        ctk += len(p.strip().split(' '))


    #print()
    return hints, top_align_phrases
def convert_to_gpt2_format(samples, tokenizer, label_dict, pad=128, alignment=None):

    count = Counter()
    tokenized_samples = []
    for ii, sample in enumerate(tqdm(samples, ascii=True)):
        #print(ii)
        sample_gpt2 = {}
        align = alignment[ii]
        if 'evidence' in sample.keys():
            sample_gpt2['evidence'] = sample['evidence']
        premise_tokens, premise_tokens_read, premise_project, _ = tokens_and_polarity(sample['sent_2_tokens'], sample['sent_2_projection'])
        #premise_phrases = phrases_index(sample['sent_1_tokens'], premise_tokens)
        hypothesis_tokens, hypothesis_tokens_read, hypothesis_project, qtf_proj = tokens_and_polarity(sample['sent_1_tokens'], sample['sent_1_projection'])

        #for x, y in zip(hypothesis_tokens, hypothesis_project):
        #    print(x, y)
        #print()
        #print(' '.join(premise_tokens).replace('Ġ', ''))
        #print(hypothesis_tokens)

        premise_phrases, raw_pm_phrases = phrases_index(sample['sent_2_tokens'], premise_tokens_read)
        hypothesis_phrases, raw_hp_phrases = phrases_index(sample['sent_1_tokens'], hypothesis_tokens_read)
        for st, ed in hypothesis_phrases:
            if qtf_proj != -1 and st <= qtf_proj <= ed:
                for j in range(st, ed+1):
                    hypothesis_project[j-1] = [0, 1, 2, 3, 4, 5, 6]
        hints, top_align = fix_proposal(raw_pm_phrases, raw_hp_phrases, align)
        #print(raw_pm_phrases)
        #print(raw_hp_phrases)
        #print(hints, '\n')
        if sample['y'] == 1 or sample['y'] == 'contradiction':
            for j, proj in enumerate(hypothesis_project):
               if proj[4] != 4:
                    hypothesis_project[j][4] = 4

        sample_gpt2['top_align'] = top_align
        hypothesis_phrases = [(st + len(premise_tokens) +1, ed + len(premise_tokens) +1) for st, ed in hypothesis_phrases]


        count[len(hypothesis_phrases)] += 1
        sample_gpt2['hints'] = hints
        sample_gpt2['premise_length'] = len(premise_tokens)
        sample_gpt2['hypothesis_length'] = len(hypothesis_tokens)
        sample_gpt2['hypothesis_phrases'] = hypothesis_phrases

        concat_sent = [tokenizer.bos_token] + premise_tokens + [tokenizer.eos_token] + hypothesis_tokens + [tokenizer.eos_token]
        concat_project = [default_projection] + premise_project + [default_projection] + hypothesis_project + [default_projection]

        #phrases = []
        #for st, ed in hypothesis_phrases:
        #    phrases.append(' '.join(concat_sent[st:ed + 1]).replace('Ġ', ''))
        #print('\t*'.join(phrases))
        #print()
        #if ii == 174:
        #    print()
        #    exit()
        # tokens and ids
        n_valid = len(concat_sent)
        sample_gpt2['tokens'] = concat_sent.copy()
        while len(concat_sent) < pad:
            concat_sent.append(tokenizer.pad_token)
            concat_project.append(default_projection)
        sample_gpt2['token_idxs'] = tokenizer.convert_tokens_to_ids(concat_sent)
        sample_gpt2['projections'] = concat_project

        # label
        if sample['y'] in [0, 1, 2]:
            sample_gpt2['y'] = sample['y']
        else:
            sample_gpt2['y'] = label_dict[sample['y']]

        # segment id
        segment_idxs = [1] * n_valid + [0] * (pad-n_valid)
        for i in range(len(segment_idxs)):
            segment_idxs[i] = 0
            if concat_sent[i] == tokenizer.eos_token and i > 0:
                break
        assert sum(segment_idxs) < len(segment_idxs)
        sample_gpt2['segment_idxs'] = segment_idxs

        # mask
        masks = [1] * n_valid + [0] * (pad - n_valid)
        sample_gpt2['masks'] = masks

        assert len(concat_sent) == len(sample_gpt2['token_idxs'])
        assert len(concat_sent) == len(sample_gpt2['segment_idxs'])
        assert len(concat_sent) == len(sample_gpt2['masks'])
        tokenized_samples.append(sample_gpt2)
        #for k, v in sample_bert.items():
        #    print(v)
        #exit()
    #print(count)
    #exit()
    return tokenized_samples


def main(file, align_file, save):
    samples = read_from_file(file)
    with open(align_file, 'r', encoding='utf-8') as fa:
        aligns = json.load(fa)
    converted_samples = convert_to_gpt2_format(samples, tokenizer, label_dict=label_dict, alignment=aligns)
    with open(save, 'wb') as f:
        pickle.dump(converted_samples, f)


if __name__ == '__main__':
    main('./data/snli/train_records.pkl', './data/snli/aligned_snli_train.json', './data/snli_gpt2/train_records5_rev.pkl')
    main('./data/snli/dev_records.pkl', './data/snli/aligned_snli_dev.json', './data/snli_gpt2/dev_records5_rev.pkl')
    main('./data/snli/test_records.pkl', './data/snli/aligned_snli_test.json', './data/snli_gpt2/test_records5_rev.pkl')
    #main('./data/help/test_records.pkl', './data/help/aligned_help_test.json', './data/standalone/help_train_records3.pkl')
    #main('./data/esnli/full_records.pkl', './data/esnli/aligned_esnli_test.json', './data/esnli/esnli_test_records4_rev.pkl')

 