# coding: utf-8
# 2023/7/3 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score,mean_squared_error, f1_score
from EduCDM import CDM
from torch_geometric.nn import GCNConv, global_mean_pool
import json
import csv
import os
import time
class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(16, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.linear = torch.nn.Linear(input_dim, 16)

    def forward(self, x, edge_index, batch):
        x=self.linear(x)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x



class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)



class Net(nn.Module):

    def __init__(self, exer_n, student_n, knowledge_n, mf_type, dim,KCs,input_dim):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = 256, 128  # changeable
        self.KCs = KCs.to('cuda')
        self.input_dim = input_dim
        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.2)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.2)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        self.teacher_diff = GNNEncoder(input_dim=self.input_dim, hidden_dim=self.emb_dim*2, output_dim=self.emb_dim)
        self.teacher_disc = GNNEncoder(input_dim=self.input_dim, hidden_dim=self.emb_dim * 2, output_dim=1)
        self.student=nn.Linear(self.input_dim,self.emb_dim)
        self.evaluate=nn.Linear(self.input_dim,self.emb_dim)

        self.attention=nn.MultiheadAttention(embed_dim=self.emb_dim, num_heads=2,batch_first=True)
        self.KC_emb=nn.Linear(input_dim,self.emb_dim)


        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(3 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full3 = nn.Linear(3 * self.emb_dim, self.emb_dim)
            self.stat_full4 = nn.Linear(self.emb_dim, 1)
            self.stat_full5 = nn.Linear(3 * self.emb_dim, self.emb_dim)
            self.stat_full6 = nn.Linear(self.emb_dim, 1)

            self.teacher_text_emb=nn.Linear(self.input_dim,self.emb_dim)
            self.l1=nn.Parameter(nn.Parameter(torch.rand(1)) )
            self.l2 = nn.Parameter(nn.Parameter(torch.rand(1)))

        # initialize
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param)
        # nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point,teacher_text,teacher,student,evaluate,visual=False):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stu_res=self.student(student)

        batch, dim = stu_emb.size()

        KC_bia=self.KCs.repeat(batch, 1).view(batch, self.knowledge_n, -1)

        KC_bia=self.KC_emb(KC_bia)
        start_time = time.time()
        exer_emb = self.teacher_diff(teacher.x, teacher.edge_index, teacher.batch)
        e_discrimination = torch.sigmoid(self.teacher_disc(teacher.x, teacher.edge_index, teacher.batch))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码运行时间：{elapsed_time} 秒")
        # exer_emb=self.teacher_text_emb(teacher_text)#消融AMR
        evaluate=self.evaluate(evaluate)

        # get knowledge proficiency

        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        exer_emb_stu= exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)

        print("KC_bia.shape:", KC_bia.shape)
        print("exer_emb_stu.shape:", exer_emb_stu.shape)

        KC_text_rel,rel=self.attention(KC_bia, exer_emb_stu, exer_emb_stu)
        rel = KC_text_rel * input_knowledge_point.view(batch, self.knowledge_n, -1)

        torch.set_printoptions(threshold=torch.inf)  # threshold 设置为无限大


        stu_res= stu_res.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        evaluate = evaluate.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)



        # rel=stu_emb#KC消融
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            stat_emb_matc = torch.sigmoid(self.stat_full1(torch.cat((stu_emb,rel,evaluate), dim=-1)))
            stat_emb_matc = torch.sigmoid(self.stat_full2(stat_emb_matc)).view(batch, -1)

            stat_emb_tea = torch.sigmoid(self.stat_full3(torch.cat((stu_emb, rel,stu_res), dim=-1)))
            stat_emb_tea = torch.sigmoid(self.stat_full4(stat_emb_tea)).view(batch, -1)

            stat_emb_stu = torch.sigmoid(self.stat_full3(torch.cat((stu_emb, stu_res, evaluate), dim=-1)))
            stat_emb_stu = torch.sigmoid(self.stat_full4(stat_emb_stu)).view(batch, -1)


            stat_emb=self.l1*stat_emb_matc + self.l2*stat_emb_tea+(1-self.l1-self.l2)*stat_emb_stu

            # stat_emb = self.l2 * stat_emb_tea + (1 - self.l2) * stat_emb_stu#qM
            # stat_emb = self.l1 * stat_emb_matc  + (1 - self.l1) * stat_emb_stu#ts
            # stat_emb = self.l1 * stat_emb_matc  + (1 - self.l1) * stat_emb_tea #se



        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination




        # prednet
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))


        if visual==True:
            stu_emb=torch.sigmoid(stu_emb.sum(dim=-1, keepdim=False))
            return output_1.view(-1),stu_id,input_exercise,input_knowledge_point,stat_emb,stu_emb,stat_emb_matc,stat_emb_tea,stat_emb_stu,k_difficulty,e_discrimination
        return output_1.view(-1)



class KaNCD(CDM):
    def __init__(self, **kwargs):
        super(KaNCD, self).__init__()
        mf_type = kwargs['mf_type'] if 'mf_type' in kwargs else 'gmf'
        self.net = Net(kwargs['exer_n'], kwargs['student_n'], kwargs['knowledge_n'], mf_type, kwargs['dim'],kwargs['KCs'],kwargs['input_dim'])



    def train(self, train_set, train_graph,valid_set,valid_graph, lr=0.001, device='cpu', epoch_n=15):
        logging.info("traing... (lr={})".format(lr))
        self.net = self.net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        for epoch_i in range(epoch_n):
            self.net.train()
            epoch_losses = []
            batch_count = 0

            for batch_data,batch_graph in zip(train_set,train_graph):
                batch_count += 1
                user_info, item_info, knowledge_emb, teacher_text,student, evalu, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                teacher_text: torch.Tensor = teacher_text.to(device)
                student: torch.Tensor = student.to(device)
                evalu: torch.Tensor = evalu.to(device)
                teacher: torch.Tensor = batch_graph.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb,teacher_text,teacher,student,evalu)

                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())
            # self.processData("IRE_Comta",epoch_i,"train", train_set, train_graph)
            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            logging.info("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            auc, acc,rmse,f1 = self.eval(valid_set, valid_graph)
            print("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))
            logging.info("[Epoch %d] auc: %.6f, acc: %.6f" % (epoch_i, auc, acc))

        return auc, acc

    def eval(self, test_data,test_graph, device="cuda",visual=False,output_file="./NONE.csv"):
        logging.info('eval ... ')
        self.net = self.net.to(device)
        self.net.eval()
        y_true, y_pred = [], []
        output_file = output_file
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            header = ["student_id", "item_id", "Q_matrix", "threeLevelSum","stuState","queMatch", "staInRes", "staInTea", "diff", "discri", "correct","pred"]
            writer.writerow(header)
            for batch_data,batch_graph in zip(test_data,test_graph):
                user_info, item_info, knowledge_emb, teacher_text,student, evalu, y = batch_data
                user_info: torch.Tensor = user_info.to(device)
                item_info: torch.Tensor = item_info.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                teacher_text: torch.Tensor = teacher_text.to(device)
                student: torch.Tensor = student.to(device)
                evalu: torch.Tensor = evalu.to(device)
                teacher: torch.Tensor = batch_graph.to(device)
                y: torch.Tensor = y.to(device)
                pred = self.net(user_info, item_info, knowledge_emb, teacher_text,teacher, student, evalu)

                if visual:
                    pred,stu,item,Q_matrix,t,init,t_e,t_s,s_e,diff,evaluate = self.net(user_info, item_info, knowledge_emb, teacher_text, teacher, student, evalu,visual=True)
                    correct=y
                    pred_1 = [round(x) for x in pred.cpu().tolist()]
                    stu = stu.cpu().tolist()
                    item = item.cpu().tolist()
                    Q_matrix = Q_matrix.cpu().tolist()
                    t = t.cpu().tolist()
                    init = init.cpu().tolist()
                    t_e = t_e.cpu().tolist()
                    t_s = t_s.cpu().tolist()
                    s_e = s_e.cpu().tolist()
                    diff = diff.cpu().tolist()
                    evaluate = evaluate.cpu().tolist()
                    correct = correct.cpu().tolist()

                    for j in range(len(stu)):
                        row = [
                            stu[j],
                            item[j],
                            list(Q_matrix[j]),
                            list(t[j]),
                            list(init[j]),
                            list(t_e[j]),
                            list(t_s[j]),
                            list(s_e[j]),
                            list(diff[j]),
                            list(evaluate[j]),
                            correct[j],
                            pred_1[j]
                        ]
                        writer.writerow(row)


                y_pred.extend(pred.detach().cpu().tolist())
                y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5),mean_squared_error(y_true, y_pred),f1_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)

    def processData(self,dataname,epoch,type,test_data,test_graph):

        ep=str(epoch)
        file_dir="../../data/"+dataname
        file_name=file_dir+'/'+type+'_ep'+ep+'.csv'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.eval(test_data,test_graph,visual=True,output_file=file_name)
        # students= torch.arange(0, 123,device='cuda')
        # stu_emb = self.net.student_emb(students)  # shape: (123, embedding_dim)
        #
        # # 3. 变换形状并重复
        # stu_emb = stu_emb.view(123, 1, self.net.emb_dim).repeat(1, self.net.knowledge_n, 1)
        # # shape: (batch_size, knowledge_n, embedding_dim)
        #
        # # 4. 沿最后一个维度求和并应用 sigmoid
        # stu_emb = torch.sigmoid(stu_emb.sum(dim=-1, keepdim=False))
        # # shape: (batch_size, knowledge_n)
        # stu_emb_cpu = stu_emb.cpu().tolist()
        # # 5. 将每一行保存为 CSV 文件中的一个列表
        # output_csv = file_name
        # with open(output_csv, mode="w", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["stusta"])  # 列名
        #     for row in stu_emb.tolist():
        #         writer.writerow([row])  # 每一行是一个 list



