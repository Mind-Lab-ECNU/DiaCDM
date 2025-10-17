# coding: utf-8
# 2021/4/1 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from IPython.core.guarded_eval import dict_keys
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / (d_k ** 0.5)
        self.q_linear = nn.Linear(4096, d_k)
        self.k_linear = nn.Linear(4096, d_k)
        self.v_linear = nn.Linear(4096, d_k)
        self.d_k = d_k
    def forward(self, query, key, value, mask=None):
        # Step 1: Compute attention scores
        if query.size(1)!=self.d_k:
            query = self.q_linear(query)
            key = self.k_linear(key)
            value = self.v_linear(value)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # Step 2: Apply mask (if provided)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Step 3: Normalize scores using softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Step 4: Compute the weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.rel=ScaledDotProductAttention(self.knowledge_dim)
        self.e_diff_att=ScaledDotProductAttention(self.knowledge_dim)
        self.rel_linear=nn.Linear(self.knowledge_dim,self.knowledge_dim)

        self.relu=nn.ReLU()


        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Sequential(nn.Linear(4096,2048),nn.Linear(2048,1024),nn.Linear(1024,512),nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,1))
        self.e_difficulty_1 = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point,teacher):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty_1 = torch.sigmoid(self.e_difficulty_1(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(teacher))  # * 10
        e_diff,_=self.e_diff_att(teacher, teacher,teacher)
        know_rel=self.rel_linear(input_knowledge_point)
        know_rel, attention_weights=self.rel(know_rel,self.e_difficulty_1(input_exercise),self.e_difficulty_1(input_exercise))
        # prednet
        know_rel=torch.sigmoid(self.relu(know_rel))

        e_diff=torch.sigmoid(self.relu(e_diff))

        input_x = (stat_emb - k_difficulty)*input_knowledge_point*know_rel
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb,teacher, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                teacher: torch.Tensor = teacher.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb,teacher)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb,teacher, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            teacher: torch.Tensor = teacher.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb,teacher)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)
