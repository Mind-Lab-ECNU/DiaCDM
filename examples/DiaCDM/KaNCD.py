# coding: utf-8
# 2023/3/7 @ WangFei
import sys
sys.path.append("/team_code/JR/CDM/EduCDM-main/")
import logging
from EduCDM import KaNCD
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

def train(data_name):
    if data_name == "elion":
        train_data = pd.read_csv("../../data/elion/train_elion.csv")
        valid_data = pd.read_csv("../../data/elion/valid_elion.csv")
        test_data = pd.read_csv("../../data/elion/test_elion.csv")
        df_item = pd.read_csv("../../data/elion/item.csv")

    if data_name == "mathdial":
        train_data = pd.read_csv("../../data/mathdial/train.csv")
        valid_data = pd.read_csv("../../data/mathdial/valid.csv")
        test_data = pd.read_csv("../../data/mathdial/test.csv")
        df_item = pd.read_csv("../../data/mathdial/item.csv")
        knowledge_num = 147

    if data_name == "comta":
        train_data = pd.read_csv("../../data/IRE_Comta/train.csv")
        valid_data = pd.read_csv("../../data/IRE_Comta/valid.csv")
        test_data = pd.read_csv("../../data/IRE_Comta/test.csv")
        df_item = pd.read_csv("../../data/IRE_Comta/item.csv")
        knowledge_num = 165
    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    batch_size = 32
    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
    knowledge_n = np.max(list(knowledge_set))


    def transform(user, item, item2knowledge, score, batch_size):
        knowledge_emb = torch.zeros((len(item), knowledge_n))
        for idx in range(len(item)):
            knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

        data_set = TensorDataset(
            torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
            torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
            knowledge_emb,
            torch.tensor(score, dtype=torch.float32)
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=True)


    train_set, valid_set, test_set = [
        transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
        for data in [train_data, valid_data, test_data]
    ]
    auc_all=0
    acc_all=0
    for i in range(5):

        logging.getLogger().setLevel(logging.INFO)
        cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='ncf2', dim=20)
        cdm.train(train_set, valid_set, epoch_n=50, device="cuda", lr=0.002)
        cdm.save("kancd.snapshot")

        cdm.load("kancd.snapshot")
        auc, accuracy = cdm.eval(test_set, device="cuda")
        auc_all += auc
        acc_all += accuracy
    auc_ave = auc_all / 5
    acc_ave = acc_all / 5
    print(data_name)
    print("auc_ave: %.6f, accuracy_ave: %.6f" % (auc_ave, acc_ave))
if __name__ == "__main__":
    train("mathdial")
    # train("comta")
    # train("elion")


