import os
import transformers
from loader_chunks_fix import DataLoader
import numpy as np
from time import time
import datetime
import torch
from tqdm import tqdm
from sampling_model_fix3 import GPT2Classifier
from torch.nn.parallel import DataParallel
import random
import json
import pickle
def run_epoch(model, data_iterator, optimizer, scheduler, phase='train', batch_size=16, binary=False, explain=False, nl_explain=False):

    if phase == 'train':
        model.train()
    else:
        model.eval()

    rev_label = []
    t_correct = 0
    t_loss = 0
    n_all = 0
    t0 = time()
    count = 0

    global_iou = []
    precision = []
    recall = []
    exp_acc = []

    for ids, x_batch, m_batch, i_batch, p_batch, y_batch, hypo_len, start, end, hints in tqdm(data_iterator.sampled_batch(batch_size=batch_size, phase=phase),
                                                   total=int(len(data_iterator) / batch_size), ascii=True):
        #if count == 28:
        #    print('#')
        #count += 1
        x_batch = torch.tensor(x_batch, dtype=torch.int64).cuda()
        m_batch = torch.tensor(m_batch, dtype=torch.float32).cuda()
        i_batch = torch.tensor(i_batch, dtype=torch.int64).cuda()
        p_batch = torch.tensor(p_batch, dtype=torch.int64).cuda()
        y_batch = torch.tensor(y_batch, dtype=torch.int64).cuda()
        hypo_len = torch.tensor(hypo_len, dtype=torch.int64).cuda()
        start_batch = torch.tensor(start, dtype=torch.int64).cuda()
        end_batch = torch.tensor(end, dtype=torch.int64).cuda()
        #print(x_batch.shape)
        if torch.max(end_batch) >= 128:
            continue

        # forward
        #print(x_batch)
        batch_pred, batch_loss, proof = model(input_idxs=x_batch, masks=m_batch, segment_idxs=i_batch,
                                       projections=p_batch, hypothesis_len=hypo_len, ys=y_batch,
                                       phrase_start=start_batch, phrase_end=end_batch, hints=hints, train=phase=='train')
        batch_loss = batch_loss.mean()
        # update model params
        if phase == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()


        #y_batch = torch.clamp(y_batch, min=0, max=2).type(torch.int64)
        n_sample = y_batch.shape[0]
        n_all += n_sample
        t_loss += batch_loss.item() * n_sample


        predict = torch.argmax(batch_pred, dim=1)
        if binary:
            for j in range(predict.shape[0]):
                if predict[j] == 1:
                    predict[j] = 2



        t_correct += torch.sum(predict == y_batch).item()
        if nl_explain:
            for i in range(y_batch.shape[0]):
                eacc = data_iterator.compare_explain2(ids[i], predict[i].item(), (proof[0][i], proof[1][i]))
                exp_acc.append(eacc)
        if explain:
            for i in range(y_batch.shape[0]):
                g_iou, prc, rcl = data_iterator.compare_explain(ids[i], predict[i].item(), (proof[0][i], proof[1][i]))
                global_iou.append(g_iou)
                precision.append(prc)
                recall.append(rcl)


    print("{} Loss: {:.4f},  Accuarcy: {:.2f}%, {:.2f} Seconds Used:".
          format(phase, t_loss / n_all, 100 * t_correct / n_all, time() - t0))

    if explain:
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
        global_iou = sum(global_iou) / len(global_iou)
        f1 = 2 * precision * recall / (precision + recall)
        print('IOU: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(global_iou, precision, recall, f1))
    if nl_explain:
        print('Explanation Acc: {:.3f}'.format(sum(exp_acc)/len(exp_acc)))




if __name__ == "__main__":
    # lex_1_0 bert 49.97% nnl 89.01%
    # lex_1_1 bert 79.06% nnl 99.98% (seed=1) 99% (seed=2) 97% (seed=5)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    # default for all
    max_word = 128
    batch_size = 16
    learning_rate = 5e-5
    label_size = 3
    n_epochs = 20
    warmup_proportion = 0.1


    test_iterator = DataLoader('./data/snli_gpt2/test_records3.pkl')
    test_iterator2 = DataLoader('./data/standalone/help_test_records2.pkl')
    test_iterator3 = DataLoader('./data/standalone/med_test_records2.pkl')
    test_iterator4 = DataLoader('./data/standalone/monli_test_records2.pkl')
    test_iterator5 = DataLoader('./data/esnli/esnli_test_records3.pkl')
    test_iterator6 = DataLoader('./data/natlogic2hop/natlogic2hop_test_records.pkl')

    bert_model = GPT2Classifier(n_class=3).cuda()
    #bert_model = DataParallel(bert_model)

    # traininig sample size and warming up
    num_train_steps = int(test_iterator1.__len__() / batch_size * n_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    optimizer = transformers.AdamW(filter(lambda x: x.requires_grad, bert_model.parameters()), lr=learning_rate, eps=1e-8)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    bert_model.load_state_dict(torch.load('./temp_check/saved_model_chunk_gpt2_fixrev614ent1_5.pt'))
    print('Start Training ... ')
    for i in range(0, n_epochs):
        print('Epoch {}...'.format(i))

        #run_epoch(bert_model, train_iterator, optimizer, scheduler, phase='train', batch_size=batch_size)
        #torch.save(bert_model.state_dict(), 'saved_model_chunk_gpt2_fix.pt')
        #exit()
        with torch.no_grad():
            run_epoch(bert_model, test_iterator, optimizer, scheduler, phase='test', batch_size=batch_size)
            run_epoch(bert_model, test_iterator2, optimizer, scheduler, phase='test', batch_size=batch_size, binary=True)
            run_epoch(bert_model, test_iterator3, optimizer, scheduler, phase='test', batch_size=batch_size, binary=True)
            run_epoch(bert_model, test_iterator4, optimizer, scheduler, phase='test', batch_size=batch_size, binary=True)
            run_epoch(bert_model, test_iterator5, optimizer, scheduler, phase='test', batch_size=batch_size, explain=True)
            run_epoch(bert_model, test_iterator6, optimizer, scheduler, phase='test', batch_size=batch_size, nl_explain=True)
            exit()
        print('')

