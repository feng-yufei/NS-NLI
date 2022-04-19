import torch
from logic_table import construct_table
from random import random
import numpy as np
logic = construct_table(map=True).tolist()
#label2rel = {0: [0, 1], 1: [2, 5, 6]}
label2rel = {0: [0, 1], 1: [3, 4], 2: [2, 5, 6], 3: [2]}
rel2label = {0: 0, 1: 0, 3: 1, 4: 1, 2: 2, 5: 2, 6: 2}
ptf = False
#ptf = True


def simulate(actions, projections=None):
    curr = 0
    history = [0]
    for i, act in enumerate(actions):
        if projections:
            p_act = projections[i][act]
        else:
            p_act = act
        curr = logic[curr][p_act]
        history.append(curr)
    return curr, history


def search_fix(actions, local_relations, projections, y, n, skip):
    # actions:  relations that is selected by the agent : list
    # relation_history: relations after aggregation : list
    # probs: prob of selected action : list
    # local relations: distribution of original relation  : len * 7
    # ys : final label : atomic

    #print(actions)
    actions_ = actions.copy()
    queue = []
    for step in range(0, n):
        if step in skip:
            continue
        prob = local_relations[step]
        #print(prob)
        p, rel = torch.topk(prob, k=3)
        for i, r in enumerate(rel.tolist()):
            if r == actions[step]:
                continue
            new_actions = actions_.copy()
            new_actions[step] = r
            fixed_result, _ = simulate(new_actions, projections=projections)
            #print('tried', new_actions, fixed_result in label2rel[y.item()])
            #print(actions)
            #print('fix pos: ', step, 'fix rel: ', r, 'final res: ', fixed_result, 'target: ', label2rel[y.item()], '\n')
            if fixed_result in label2rel[y.item()]:
                queue.append((step, r, p[i].item()))  # lowest confidence
    #print(queue)
    #exit()
    queue.sort(key=lambda x: x[2], reverse=True)
    return queue

def insert_fix(externals, actions, local_relations, projections, y, n):
    # externals: a list of knowledge guided fixes [(step, rel)]

    actions_ = actions.copy()
    fix_pool = []
    random_pool = []
    for step in range(n):
        if step >= len(externals):
            continue
        hints = externals[step]
        prob = local_relations[step].tolist()
        for h in hints:
            new_actions = actions_.copy()
            new_actions[step] = h
            fixed_result, _ = simulate(new_actions, projections)
            if fixed_result in label2rel[y.item()]:
                fix_pool.append((step, h, prob[h]))

            random_pool.append((step, h, prob[h]))


    fix_pool.sort(key=lambda x: x[2], reverse=True)
    if ptf:
        print('fix pool', fix_pool)
        print('random pool', random_pool)
    M = 3
    fix_log = []
    for i in range(M):

        if len(fix_pool) > 0 and random() < 0.8:
            step, h, prob = fix_pool[0]
            new_actions = actions_.copy()
            new_actions[step] = h
            fixed_result, _ = simulate(new_actions, projections)
            if fixed_result in label2rel[y.item()]:
                actions_[step] = h
                fix_log.append((step, h, 'fix'))
                if (step, h, prob) in fix_pool:
                    fix_pool.remove((step, h, prob))
                if (step, h, prob) in random_pool:
                    random_pool.remove((step, h, prob))
        elif len(random_pool) > 0:
            sp_prob = [x[2] for x in random_pool]
            sp_prob = np.array(sp_prob) / sum(sp_prob)
            sp_idx = int(np.argmax(np.random.multinomial(1, sp_prob)))
            step, h, prob = random_pool[sp_idx]
            accept_ratio = prob / local_relations[step].tolist()[actions_[step]]
            if random() < accept_ratio:
                actions_[step] = h
                fix_log.append((step, h, 'random'))
                if (step, h, prob) in fix_pool:
                    fix_pool.remove((step, h, prob))
                if (step, h, prob) in random_pool:
                    random_pool.remove((step, h, prob))

    last_state, history_ = simulate(actions_, projections)
    if ptf:
        print('fixed', fix_log, history_[1:])
    return actions_, history_, last_state


def fix_nnl_path(externals, actions, local_relations, relation_history, projections, y, n):
    actions_ = actions[1:n+1].tolist().copy()
    projections_ = projections[1:n+1].tolist()
    history_ = relation_history[0:n+1].tolist()

    for k, v in externals.items():
        if 4 in v:
            externals[k].remove(4)
    if ptf:
        print('actions: ', actions_)
        print('history: ', history_)
    last_state = relation_history[n]  # relation history has 1 + n entries

    if len(externals) > 0:
        actions_, history_, last_state = insert_fix(externals, actions_, local_relations, projections_, y, n)

    #"""
    if last_state not in label2rel[y.item()]:
        #print('old', actions_)
        fix_queue = search_fix(actions_, local_relations, projections_, y, n, skip=[])
        good_fix = []
        normal_fix = []
        for ff in fix_queue:
            if ff[0] >=len(externals):
                normal_fix.append(ff)
            elif len(externals[ff[0]]) == 0:
                good_fix.append(ff)
            elif ff[1] in externals[ff[0]]:
                good_fix.append(ff)
            else:
                normal_fix.append(ff)
        good_fix.sort(key=lambda x: x[2], reverse=True)
        normal_fix.sort(key=lambda x: x[2], reverse=True)
        fix_queue = good_fix + normal_fix
        if ptf:
            print(fix_queue)
        if len(fix_queue) > 0:
            stp, rr, _ = fix_queue[0]
            actions_[stp] = rr
            last_state, history_ = simulate(actions_, projections_)
        #print('new', actions_)
        #if len(fix_queue) == 0:
        #    print()
    #"""
    #print('act: ', actions_)
    return actions_, history_, last_state