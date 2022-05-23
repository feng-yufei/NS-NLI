import torch
from torch import nn
from transformers import GPT2Model
from logic_table import construct_table
import random
from random import random as rnd
from fix2 import fix_nnl_path

label2rel = {0: [0, 1], 1: [3, 4], 2: [2, 5, 6], 3: [2]}
rel2label = {0: 0, 1: 0, 3: 1, 4: 1, 2: 2, 5: 2, 6: 2}

ptf = False
#ptf = True

class GPT2Classifier(nn.Module):

    def __init__(self, n_class):
        super(GPT2Classifier, self).__init__()
        self.model = GPT2Model.from_pretrained('gpt2')
        self.local_relation_layer = nn.Sequential(nn.Dropout(0.1), nn.Linear(2*768, 768, bias=True), nn.ReLU(), nn.Linear(768, 7))
        self.projection_layer = None
        self.classification_layer = nn.Sequential(nn.Dropout(0.1), nn.Linear(2*768, n_class, bias=True))
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.logic = construct_table(map=True).cuda()
        self.rm = torch.tensor([0, 0, 0, 1, 0, 1, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        #self.local_relation_layer.apply(weights_init)

    def forward(self, input_idxs, masks, segment_idxs, projections, hypothesis_len, phrase_start, phrase_end, ys=None, hints=None, train=False):

        def get_vectors(tensor, index):

            ind1 = torch.arange(index.shape[0], dtype=torch.int64).unsqueeze(-1).repeat([1, index.shape[1]]).view(-1)
            ind2 = index.view(-1)
            return tensor[ind1, ind2].view(index.shape[0], index.shape[1], -1)
            # index: bs * m

        # projection batch * length * 7
        batch_size = ys.shape[0]
        max_agg = torch.max(hypothesis_len).item()

        outputs = self.model(input_idxs, token_type_ids=segment_idxs, attention_mask=masks)
        #logits = self.classification_layer(outputs[0][range(batch_size), torch.sum(masks, dim=-1, dtype=torch.int64)-1])
        #ce_loss = self.loss_func(logits, ys)
        #return logits, ce_loss, (0, 0)

        hidden_states = outputs[0]  # (bs, length, 768)
        phrase_start_rep = get_vectors(hidden_states, phrase_start)
        phrase_end_rep = get_vectors(hidden_states, phrase_end)

        phrase_rep = torch.cat([phrase_start_rep, phrase_end_rep], dim=-1)
        local_relations = torch.softmax(self.local_relation_layer(phrase_rep) - self.rm * 1e06, dim=-1)  # (bs, length, 7)


        # aggregation
        prev_relation = torch.zeros([batch_size], dtype=torch.int64).cuda()  # bs, 1
        relation_action = prev_relation.unsqueeze(-1).cuda()
        relation_history = prev_relation.unsqueeze(-1).cuda()
        probs = 0.5 * torch.ones([batch_size, 1]).cuda()  # bs, 1
        used_projections = torch.zeros([batch_size, 1, 7], dtype=torch.int64).cuda()
        #reward = torch.zeros([batch_size, 1], dtype=torch.float32).cuda()
        choices = []
        #print()
        for i in range(0, max_agg):
            #torch.manual_seed(0)
            # sample relation at current time-step
            transition = local_relations[:, i]   # bs, 7
            #print(transition)
            #mlogits = torch.cat([mlogits, transition.unsqueeze(1)], dim=1)
            if train:
                
                sampled_relation = torch.multinomial(transition, num_samples=1).squeeze(-1)
                #print(sampled_relation.shape)5W*

            else:
                sampled_relation = torch.argmax(transition, dim=-1)

            # choices.append(sampled_relation.item())
            relation_action = torch.cat([relation_action, sampled_relation.unsqueeze(-1)], dim=-1)
            sampled_prob = transition[range(batch_size), sampled_relation]
            # sampled_prob = transition[range(batch_size), ys]
            probs = torch.cat([probs, sampled_prob.unsqueeze(-1)], dim=-1)
            # perform 7*7 table
            #print(projections.shape)
            proj = projections[range(batch_size), phrase_start[:, i]]
            used_projections = torch.cat([used_projections, proj.unsqueeze(1)], dim=1)
            if ptf:
                print(proj.squeeze().tolist(), sampled_relation.item(), [float(format(x, '.2f')) for x in transition.squeeze().tolist()])
            agg_result = self.logic[prev_relation, self.project(sampled_relation, proj)]
            #agg_result = self.logic[prev_relation, sampled_relation]
            # agg_result = sampled_relation
            # phrase start

            relation_history = torch.cat([relation_history, agg_result.unsqueeze(-1)], dim=-1)

            # reward shaping
            #reward_curr = self.shape_reward(prev_relation, agg_result, ys, hypothesis_len, i)
            #reward = torch.cat([reward, reward_curr.unsqueeze(-1)], dim=-1)
            prev_relation = agg_result

        #print(choices)
        last_state = relation_history[range(batch_size), hypothesis_len]   # bs

        relation_history_bk = relation_history.clone().detach()

        reward_raw = self.shape_reward(relation_history, ys, hypothesis_len, 0.5)
        policy_gradient_loss_raw = self.loss(probs, reward_raw, hypothesis_len, relation_history, ys, local_relations)

        logit_entail = (last_state == 0) + (last_state == 1)
        logit_contradiction = (last_state == 3) + (last_state == 4)
        logit_neutral = (last_state == 2) + (last_state == 5) + (last_state == 6)
        logits = torch.cat([logit_entail.unsqueeze(-1), logit_contradiction.unsqueeze(-1), logit_neutral.unsqueeze(-1)], dim=-1).type(torch.int64)
        if ptf:
            relation_history_bk = [str(x) for x in relation_history_bk[0].tolist()]
            print('\t'.join(relation_history_bk[1:]))
            print('reward0: ', reward_raw[0].tolist()[1:])

        #return logits, policy_gradient_loss_raw, (relation_history, relation_action)
        #last_prob = probs[range(batch_size), hypothesis_len]
        #print(hints[0])
        if train:
            fixed_actions, relation_history, last_state = self.back_search(relation_action, relation_history,
                                                                           local_relations, used_projections,
                                                                           ys, hypothesis_len, hints)
            probs = 0.5 * torch.ones([batch_size, 1]).cuda()  # bs, 1
            for i in range(0, max_agg):

                transition = local_relations[:, i]

                sampled_prob = transition[range(batch_size), fixed_actions[:, i]]
                probs = torch.cat([probs, sampled_prob.unsqueeze(-1)], dim=-1)


        #last_reward = logit_entail * (ys == 0) + logit_contradiction * (ys == 1) + logit_neutral * (ys == 2)  # bs, 1
        #last_reward = 2 * last_reward.type(torch.float32) - 1
        reward = self.shape_reward(relation_history, ys, hypothesis_len, gamma=1.0, fix=False)
        if ptf:
            print('hints: ', hints[0])
            relation_history_bk2 = [str(x) for x in relation_history[0].tolist()]
            print('\t'.join(relation_history_bk2[1:]))
            print('reward1: ', reward[0].tolist()[1:])




        #last_reward -= last_reward.mean()
        #print(last_state.tolist())
        #print(ys.tolist())
        #print(last_reward.tolist())
        #print(last_prob)
        #exit()
        #loss_ = -torch.log(last_prob) * last_reward
        #print(loss_.mean())
        #print()
        #loss = - torch.log(last_prob) * (last_reward > 0).type(torch.float32) - torch.log(1-last_prob+0.0001) * (last_reward < 0).type(torch.float32)


        #return logits, loss, 0
        #reward = reward - reward.mean()



        #logits = self.classification_layer(outputs[0][:, 0])
        policy_gradient_loss = self.loss(probs, reward, hypothesis_len, relation_history, ys, local_relations)
        policy_gradient_loss = 0.5 * policy_gradient_loss_raw + 0.5 * policy_gradient_loss
        #print()
        return logits, policy_gradient_loss, (relation_history, relation_action)

    def project(self, relation, projection):
        return projection[range(relation.shape[0]), relation]

    def back_search(self, relation_action, relation_history, local_relations, projections, y, hypo_len, hints):

        fixed_actions = torch.zeros(relation_action.shape, dtype=torch.int64).cuda()
        fixed_history = torch.zeros(relation_action.shape, dtype=torch.int64).cuda()
        last_states = torch.zeros(relation_action.shape[0], dtype=torch.int64).cuda()
        for i in range(relation_history.shape[0]):
            f_rel, f_hist, lst_state = fix_nnl_path(hints[i], relation_action[i],
                                                    local_relations[i], relation_history[i],
                                                    projections[i], y[i], hypo_len[i])
            fixed_actions[i, 1:hypo_len[i]+1] = torch.tensor(f_rel, dtype=torch.int64).cuda()
            fixed_history[i, 0:hypo_len[i]+1] = torch.tensor(f_hist, dtype=torch.int64).cuda()
            last_states[i] = lst_state
        return fixed_actions[:, 1:], fixed_history, last_states

    def shape_reward(self, relation_history, ys, hypo_len, gamma=0.5, fix=False):

        def compute_reward(history, y, hp_len):

            rwd = torch.zeros([history.shape[0]])
            label_rel = label2rel[y.item()]
            hit_obj = history[hp_len].item() in label_rel
            final_rwd = 2 * int(hit_obj) - 1


            gamma_ = 1 if hit_obj else gamma
            #gamma_ = 0.5
            for i in range(1, hp_len + 1):

                rwd[i] = final_rwd * pow(gamma_, hp_len - i)
                #print('*', rwd[0], final_rwd * pow(gamma, hp_len - i))
                if history[i] in [2, 5, 6] and not hit_obj:
                    rwd[i] = -1
                    break

                #if history[i] == 0 and not hit_obj and i < hp_len:
                #    rwd[i] = final_rwd * pow(gamma_*0.5, hp_len - i)
                    #rwd[i] = min(0.2, rwd[i])

                #if y.item() == 2 and history[hp_len].item() == 6:
                #    rwd[i] = rwd[i] * 1.5

                #if y.item() == 2 and history[hp_len].item() == 2:
                #    rwd[i] = rwd[i] * 0.5


                if hit_obj and history[hp_len] == 0:
                    rwd[i] = min(0, rwd[i])

                if history[i] in label_rel:
                    rwd[i] = max(0, rwd[i])

            #if prev in [5, 6]:
            #    reward = 0

            return rwd

        def compute_reward_fix(history, y, hp_len):

            rwd = torch.zeros([history.shape[0]])
            label_rel = label2rel[y.item()]

            for i in range(1, hp_len + 1):
                if history[i-1] not in label_rel and history[i] in label_rel:
                    rwd[i] = 1

            return rwd

        reward = torch.zeros(relation_history.shape, dtype=torch.float32).cuda()  # bs
        for i in range(relation_history.shape[0]):
            if not fix:
                reward[i, :] = compute_reward(relation_history[i], ys[i], hypo_len[i])
            else:
                reward[i, :] = compute_reward_fix(relation_history[i], ys[i], hypo_len[i])
        return reward

    def loss(self, probs, reward, hypo_len, relation_history, ys, local_relations):
        #reward = reward.unsqueeze(-1).repeat([1, probs.shape[1]])
        mask = torch.ones_like(probs).cuda()
        for i in range(hypo_len.shape[0]):
            mask[i, hypo_len[i]+1:] = 0
            #mask[i, 0:hypo_len[i]] = 0
        #print(probs.shape, reward.shape)
        #probs = probs[:, 1:]
        #mask = mask[:, 1:]
        #reward = reward[:, 1:]
        loss_ = - torch.log(probs) * reward * (reward > 0).type(torch.float32) * mask - torch.log(1-probs+0.001) * torch.abs(reward) * (reward < 0).type(torch.float32) * mask
        """
        loss_1 = - torch.log(probs) * reward * (reward > 0).type(torch.float32) * mask
        loss_2 = - torch.log(1-probs) * torch.abs(reward) * (reward < 0).type(torch.float32) * mask
        print(relation_history)
        print(reward)
        print(ys)
        print(probs)
        #print(local_relations)
        print(loss_1)
        print(loss_2)
        print(loss_)
        #print()
        """
        return torch.sum(loss_, dim=-1)

if __name__ == '__main__':
    def get_vectors(tensor, index):
        ind1 = torch.arange(index.shape[0], dtype=torch.int64).unsqueeze(-1).repeat([1, index.shape[1]]).view(-1)
        ind2 = index.view(-1)
        print(ind1, ind2)
        return tensor[ind1, ind2].view(index.shape[0], index.shape[1], -1)
        # index: bs * m


    A = torch.tensor([[0.6, 0.2, 0.3],[0.4, 0.1, 0.5]])
    A = A.unsqueeze(-1).repeat([1, 1, 2])
    A[:, :, 1] += 1
    print(A)
    ind = torch.tensor([[1, 2], [2, 1]], dtype=torch.int64)
    B = get_vectors(A, ind)
    print(B)
