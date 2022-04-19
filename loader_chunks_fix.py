import numpy as np
import pickle
from itertools import chain
#from metrics import partial_match_score

map_rel = {'eq': [0, 1], 'ent_f': [0, 1], 'ent_r': [2], 'alt': [4], 'ind': [6]}
remove_from_rationale = ['is', 'are', 'a', 'at', 'to', 'the', 'for', 'in', 'with', 'an']
def array(x, dtype=np.int32):
    return np.array(x, dtype=dtype)


def load_pkl(file):
    # load pickle file
    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data

rel_map = {0:'entailment', 1:'entailment', 2:'neutral', 3:'contradiction', 4:'contradiction', 5:'neutral', 6:'neutral'}
class DataLoader:

    def __init__(self, data_file, rev_data_file=None, rev_id=None,  max_word=128, max_phrase=36, hold=False):

        self.max_word = max_word
        self.max_phrase = max_phrase
        self.dataset = load_pkl(data_file)
        #if hold:
        #    self.dataset = self.dataset[0:1000]
        #"""
        if rev_data_file:
            rev_data = load_pkl(rev_data_file)
            rev_ids = load_pkl(rev_id)
            for rid in rev_ids:
                sample = rev_data[rid]
                sample['y'] = 3
                self.dataset.append(sample)

        assert self.max_word == len(self.dataset[0]['token_idxs'])

        self.sample_index = np.arange(0, len(self.dataset))
        print('Training Samples: {} Loaded'.format(len(self.dataset)))

        self.pos = 0

    def __len__(self):
        return len(self.dataset)

    def iter_reset(self, shuffle=True):
        self.pos = 0
        if shuffle:
            np.random.shuffle(self.sample_index)

    def sampled_batch(self, batch_size, phase='train'):

        index = self.sample_index
        n = len(self.dataset)
        # batch iterator, shuffle if train
        self.iter_reset(shuffle=True if phase == 'train' else False)
        #self.iter_reset(shuffle=False)

        while self.pos < n:



            X_batch = []
            ID = []
            Y_batch = []
            sgid_batch = []
            M_batch = []
            Proj_batch = []
            batch_maxlen = 0
            batch_phrase_maxlen = 0
            hypo_len = []  # length of hypothesis
            hints = []

            phrase_len = []
            Phrase_start = []
            Phrase_end = []

            for i in range(batch_size):
                ID.append(index[self.pos])
                sample = self.dataset[index[self.pos]]
                if len(sample['hypothesis_phrases']) > self.max_phrase:
                    self.pos += 1
                    if self.pos >= n:
                        break
                    continue
                X_batch.append(sample['token_idxs'][0:self.max_word])
                Y_batch.append(sample['y'])
                sgid_batch.append(sample['segment_idxs'][0:self.max_word])
                M_batch.append(sample['masks'][0:self.max_word])
                Proj_batch.append(sample['projections'][:self.max_word]) # batchsize, len, 6
                slen = min(sample['premise_length'] + sample['hypothesis_length'] + 3, self.max_word)
                batch_maxlen = max(batch_maxlen, slen)
                hypo_len.append(sample['hypothesis_length'])
                if 'hints' in sample.keys():
                    hints.append(sample['hints'])
                else:
                    hints.append([])
                start = [0] * self.max_phrase
                end = [0] * self.max_phrase
                for p, (st, ed) in enumerate(sample['hypothesis_phrases']):
                    if p < len(start):
                        start[p] = st
                        end[p] = ed
                Phrase_start.append(start)
                Phrase_end.append(end)
                batch_phrase_maxlen = max(batch_phrase_maxlen, len(sample['hypothesis_phrases']))
                phrase_len.append(len(sample['hypothesis_phrases']))

                self.pos += 1
                if self.pos >= n:
                    break

            yield ID, array(X_batch)[:, 0:batch_maxlen], array(M_batch)[:, 0:batch_maxlen], \
                  array(sgid_batch)[:, 0:batch_maxlen], array(Proj_batch)[:, 0:batch_maxlen, :], \
                  array(Y_batch), array(phrase_len), array(Phrase_start)[:, 0:batch_phrase_maxlen], \
                  array(Phrase_end)[:, 0:batch_phrase_maxlen], hints

    def display(self, index, predict, proof):
        proof = proof[0]
        sample = self.dataset[index]
        tokens = sample['tokens']
        sep = []
        for i, t in enumerate(tokens):
            #if t == '[SEP]':
            if t == '<|endoftext|>':
                sep.append(i)
        #premise = ' '.join(tokens[(sep[0] + 1): sep[1]]).replace(' ##', '')
        #hypothesis = tokens[0:sep[0]]

        premise = ' '.join(tokens[(sep[0] + 1): sep[1]]).replace('Ġ', '')
        hypothesis = tokens[(sep[1])+1:sep[2]]
        phrases = []
        for st, ed in sample['hypothesis_phrases']:
            phrases.append(' '.join(tokens[st:ed+1]).replace('Ġ', ''))
        proof = proof.tolist()[1:len(phrases) +1]

        if 'switch_points' in sample.keys():
            print(sample['switch_points'], sample['switch_states'])
        print(premise)
        print('\t*'.join(phrases))
        print('\t'.join([str(p) for p in proof]))
        print('Pred: ', predict, 'True: ', sample['y'], '\n')

    def compare_explain(self, index, predict, proof):

        def concat(ph):
            rat = ''
            ph = ph.strip()
            for i in range(len(ph)):
                if ph[i] == ' ' and ph[i+1] != 'Ġ':
                    continue
                rat += ph[i]
            return rat.replace('Ġ', '')

        sample = self.dataset[index]
        tokens = sample['tokens']
        sep = []
        for i, t in enumerate(tokens):
            #if t == '[SEP]':
            if t == '<|endoftext|>':
                sep.append(i)

        def iou(pred, truth, premise=None):
            truth = truth[1]
            pred_tokens = []
            truth_tokens = []
            premise_tokens = premise.split(' ') + remove_from_rationale
            for p in pred:
                p = p.replace('.', '')
                pred_tokens += p.lower().strip().replace('  ', ' ').split(' ')
            pred_tokens = [p for p in pred_tokens if p not in premise_tokens]
            for p in truth:
                p = p.replace('.', '')
                truth_tokens += p.lower().strip().replace('  ', ' ').split(' ')
            num = len(set(pred_tokens) & set(truth_tokens))
            denom = len(set(pred_tokens) | set(truth_tokens))
            g_iou = 0 if denom == 0 else num / denom
            return g_iou

        def iou2(pred, truth, premise=None, threshold=0.5):
            truth = truth[1]
            pred_phs = []
            truth_phs = []
            premise_tokens = premise.split(' ') + remove_from_rationale
            for p in pred:
                p = p.replace('.', '')
                ptks = p.lower().strip().replace('  ', ' ').split(' ')
                ptks = [t for t in ptks if t not in premise_tokens]
                pred_phs.append(ptks)
            for p in truth:
                p = p.replace('.', '')
                truth_phs.append(p.lower().strip().replace('  ', ' ').split(' '))
            ious = []
            for pph in pred_phs:
                best_iou = 0
                for tph in truth_phs:

                    num = len(set(pph) & set(tph))
                    denom = len(set(pph) | set(tph))
                    iou = 0 if denom == 0 else num / denom
                    if iou > best_iou:
                        best_iou = iou
                ious.append(best_iou)
            threshold_hit = sum(int(x >= threshold) for x in ious)
            precision = threshold_hit / len(pred) if len(pred) > 0 else 0
            recall = threshold_hit / len(truth) if len(truth) > 0 else 0
            return precision, recall

        #premise = ' '.join(tokens[(sep[0] + 1): sep[1]]).replace(' ##', '')
        #hypothesis = tokens[0:sep[0]]

        premise = ' '.join(tokens[(sep[0] + 1): sep[1]]).replace('Ġ', '')
        hypothesis = tokens[(sep[1])+1:sep[2]]
        phrases = []
        phrases_r = []
        for st, ed in sample['hypothesis_phrases']:
            phrases.append(' '.join(tokens[st:ed+1]).replace('Ġ', ''))
            phrases_r.append(' '.join(tokens[st:ed + 1]))
        actions = proof[1].tolist()[1:len(phrases) +1]
        proof = proof[0].tolist()[1:len(phrases) +1]

        rationale = []
        prev_rel = 0
        for i in range(len(phrases)):
            rel = proof[i]
            ph = phrases_r[i]
            act = actions[i]
            if rel_map[rel] != rel_map[prev_rel]:
                rationale.append(concat(ph))
                #print(sample['top_align'][i])
            if rel == 1 and prev_rel == 0 and predict==0:
                rationale.append(concat(ph))
                #print(print(sample['top_align'][i]))
            prev_rel = rel
        """
        print(premise)
        print('\t*'.join(phrases))
        print('\t'.join([str(p) for p in proof]))
        print('\t'.join([str(p) for p in actions]))
        print('Pred: ', predict, 'True: ', sample['y'])
        """
        g_iou = iou(rationale, sample['evidence'], premise=premise)
        p, r = iou2(rationale, sample['evidence'], premise=premise)
        #print(sample['evidence'][0], rationale, 'global_iou: {:.2f}, p: {:.2f}, r: {:.2f}'.format(g_iou, p, r), '\n')

        return g_iou, p, r



    def compare_explain2(self, index, predict, proof):


        sample = self.dataset[index]
        tokens = sample['tokens']
        orig_tokens = []
        curr = ''
        endoftok = 0
        for t in tokens:
            if t == '<|endoftext|>':
                endoftok += 1
            if endoftok < 2 or t == '<|endoftext|>':
                continue
            if t.startswith('Ġ'):
                orig_tokens.append(curr)
                curr = t.replace('Ġ', '')
            else:
                curr += t.strip()
        if curr != '':
            orig_tokens.append(curr)

        sep = []
        for i, t in enumerate(tokens):
            #if t == '[SEP]':
            if t == '<|endoftext|>':
                sep.append(i)

        premise = ' '.join(tokens[(sep[0] + 1): sep[1]]).replace('Ġ', '')
        hypothesis = tokens[(sep[1])+1:sep[2]]
        phrases = []
        phrases_r = []
        for st, ed in sample['hypothesis_phrases']:
            phrases.append(' '.join(tokens[st:ed+1]).replace('Ġ', ''))
            phrases_r.append(' '.join(tokens[st:ed + 1]))
        actions = proof[1].tolist()[1:len(phrases) +1]
        proof = proof[0].tolist()[1:len(phrases) +1]

        rationale = []
        prev_rel = 0
        phrase_tokens = []
        curr_tk = 0
        for i in range(len(phrases)):
            rel = proof[i]
            ph = phrases_r[i]
            phrase_tokens.append([curr_tk + x for x in range(len(ph.split(' ')))])
            curr_tk += len(ph.split(' '))
            act = actions[i]
            if rel_map[rel] != rel_map[prev_rel]:
                rationale.append((ph.replace('Ġ', ''), rel))
                #print(sample['top_align'][i])
            if rel == 1 and prev_rel == 0 and predict==0:
                rationale.append((ph.replace('Ġ', ''), rel))
                #print(print(sample['top_align'][i]))
            prev_rel = rel
        #print(phrases)
        #print(proof)
        #print(phrase_tokens)

        def compare(phrase_tokens, history, switches):

            hit = 0
            for sw in switches:
                #print(sw)
                sw_pos = sw[0]
                sw_rel = sw[1]
                for tks, rel in zip(phrase_tokens, history):
                    if len(set(tks) & set(sw_pos)) > 0 and rel in map_rel[sw_rel]:
                        hit += 1
                        break
            acc = hit / len(switches)
            return acc

        switch =[(x[0][1], x[1]) for x in zip(sample['switch_points'], sample['switch_states'])]
        acc = compare(phrase_tokens, proof, switch)
        #print(acc, '\n')
        return acc

if __name__ == "__main__":
    iterator = DataLoader(data_file='./data/snli_bert/test_records2.pkl')
    for id, x, m, i, p, y, lens, st, ed in iterator.sampled_batch(32, 'dev'):

        print(x.shape)
        print(m.shape)
        print(i.shape)
        print(p.shape)
        print(y.shape)
        print(lens)
        print(st.shape)
        print(ed.shape)
        print()
