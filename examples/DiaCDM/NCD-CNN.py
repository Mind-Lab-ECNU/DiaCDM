# coding: utf-8
# 2023/3/7 @ WangFei
import sys
import time


sys.path.append("/team_code/JR/CDM/EduCDM-main/")

sys.path.append("/team_code/JR/CDM/EduCDM-main/EduCDM/")
from EduCDM.NCD_CNN import KaNCD
from transition_amr_parser.parse import AMRParser
from torch_geometric.data import DataLoader as graphDataLoader
from dia.AMR_Graph import AMR,amr_to_graph
import sys
sys.path.append("/team_code/JR/CDM/EduCDM-main/")
import logging
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import  json
import requests



tqdm.pandas()
model_name="/team_code/JR/Meta-Llama-3.1-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)  # 分词
    with torch.no_grad():
        outputs = model(**inputs)  # 前向传播
    # 提取句子级别的表示（平均池化）
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)  # (1, hidden_size)
    return sentence_embedding.squeeze(0).numpy()

def train(data_name):


    if data_name == "IRE_mathdial":
        train_data = pd.read_csv("../../data/IRE_mathdial/train.csv")
        valid_data = pd.read_csv("../../data/IRE_mathdial/valid.csv")
        test_data = pd.read_csv("../../data/IRE_mathdial/test.csv")
        df_item = pd.read_csv("../../data/IRE_mathdial/item.csv")
        knowledge_num = 147
        with open('../../data/IRE_mathdial/kc_dict_mathdial_atc.json', 'r') as f:
            KCs = json.load(f)
        texts = list(KCs.keys())[:147]

    if data_name == "IRE_Comta":
        train_data = pd.read_csv("../../data/IRE_Comta/train.csv")
        valid_data = pd.read_csv("../../data/IRE_Comta/valid.csv")
        test_data = pd.read_csv("../../data/IRE_Comta/test.csv")
        df_item = pd.read_csv("../../data/IRE_Comta/item.csv")
        with open('../../data/comta/kc_dict_comta_atc.json', 'r') as f:
            KCs = json.load(f)
        texts = list(KCs.keys())[:163]
        knowledge_num = 165
    if data_name == "comta":
        train_data = pd.read_csv("../../data/IRE_Comta/train.csv")
        valid_data = pd.read_csv("../../data/IRE_Comta/valid.csv")
        test_data = pd.read_csv("../../data/IRE_Comta/test.csv")
        df_item = pd.read_csv("../../data/IRE_Comta/item.csv")
        knowledge_num = 165

    if data_name == "elion":
        train_data = pd.read_csv("../../data/elion/train_elion.csv")
        valid_data = pd.read_csv("../../data/elion/valid_elion.csv")
        test_data = pd.read_csv("../../data/elion/test_elion.csv")
        df_item = pd.read_csv("../../data/elion/item.csv")
        with open('../../data/comta/kc_dict_comta_atc.json', 'r') as f:
            KCs = json.load(f)
        texts = list(KCs.keys())[:163]
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


    def transform(type,user, item, item2knowledge, teacher,student,eval,score, batch_size):
        knowledge_emb = torch.zeros((len(item), knowledge_n))
        for idx in range(len(item)):
            knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0
        file_path_tea = '../../data/' + data_name + '/' + type + '_teacher.npy'
        file_path_stu = '../../data/' + data_name + '/' + type + '_student.npy'
        file_path_eval = '../../data/' + data_name + '/' + type + '_eval.npy'

        if not os.path.exists(file_path_stu):
            encoded_series = teacher.progress_apply(lambda x: encode_text(x, tokenizer, model))
            teacher_tensor = torch.tensor(encoded_series, dtype=torch.float32)
            numpy_array_stu = teacher_tensor.numpy()  # 转换为 numpy 数组
            np.save(file_path_tea, numpy_array_stu)

            encoded_series = student.progress_apply(lambda x: encode_text(x, tokenizer, model))
            student_tensor = torch.tensor(encoded_series, dtype=torch.float32)
            numpy_array_stu = student_tensor.numpy()  # 转换为 numpy 数组
            np.save(file_path_stu, numpy_array_stu)

            encoded_series_eval = eval.progress_apply(lambda x: encode_text(x, tokenizer, model))
            eval_tensor = torch.tensor(encoded_series_eval, dtype=torch.float32)
            numpy_array_eval = eval_tensor.numpy()  # 转换为 numpy 数组
            np.save(file_path_eval, numpy_array_eval)
        else:
            loaded_numpy_array = np.load(file_path_tea)  # 从 .npy 文件加载
            teacher_tensor = torch.from_numpy(loaded_numpy_array)

            loaded_numpy_array = np.load(file_path_stu)  # 从 .npy 文件加载
            student_tensor = torch.from_numpy(loaded_numpy_array)

            loaded_numpy_array_eval = np.load(file_path_eval)  # 从 .npy 文件加载
            eval_tensor = torch.from_numpy(loaded_numpy_array_eval)

        data_set = TensorDataset(
            torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
            torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
            knowledge_emb,
            teacher_tensor,
            student_tensor,
            eval_tensor,
            torch.tensor(score, dtype=torch.float32)
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=False)



    def transform_graph(parser,model,tokenizer,type,teacher,  batch_size):
        teachers=[]
        file_path_amr = '../../data/' + data_name + '/' + type + 'teachers.pt'
        if not os.path.exists(file_path_amr):
            for teacher_batch in teacher:
                teachers.append(amr_to_graph(AMR(parser,teacher_batch),model,tokenizer))
            torch.save(teachers, file_path_amr)
        else:
            teachers = torch.load(file_path_amr)
        return graphDataLoader(teachers, batch_size=batch_size, shuffle=False)


    ile_path_amr = '../../data/' + data_name + '/' + "train" + 'teachers.pt'
    if not os.path.exists(ile_path_amr):
        parser = AMRParser.from_pretrained('AMR3-structbart-L')
        train_AMR_set = transform_graph(parser,model,tokenizer,"train", train_data["teacher"], batch_size)
        valid_AMR_set = transform_graph(parser,model,tokenizer,"valid", valid_data["teacher"], batch_size)
        test_AMR_set = transform_graph(parser,model,tokenizer,"test", test_data["teacher"], batch_size)
    else:
        train_AMR_set = transform_graph(None,model,tokenizer,"train", train_data["teacher"], batch_size)
        valid_AMR_set = transform_graph(None,model,tokenizer, "valid", valid_data["teacher"], batch_size)
        test_AMR_set = transform_graph( None,model,tokenizer,"test", test_data["teacher"], batch_size)

    train_set = transform("train", train_data["user_id"], train_data["item_id"], item2knowledge, train_data["teacher"],
                          train_data["student"], train_data["evaluation"],
                          train_data["score"], batch_size)
    valid_set = transform("valid", valid_data["user_id"], valid_data["item_id"], item2knowledge, valid_data["teacher"],
                          valid_data["student"], valid_data["evaluation"],
                          valid_data["score"], batch_size)
    test_set = transform("test", test_data["user_id"], test_data["item_id"], item2knowledge, test_data["teacher"],
                         test_data["student"], test_data["evaluation"],
                         test_data["score"], batch_size)

    auc_all=0
    acc_all=0


    embeddings = []
    kc_file='../../data/' + data_name + '/'  + 'KCs.pt'
    if not os.path.exists(kc_file):
        for text in texts:
            # 将文本编码为模型所需的格式
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)  # 假设最大长度为512
            outputs = model.get_input_embeddings()(inputs['input_ids'])  # 获取嵌入向量
            embeddings.append(outputs.mean(dim=1))  # 平均池化，或使用其他方式汇总

        # 将嵌入向量合并成一个163x4096的 tensor
        final_tensor = torch.stack(embeddings).squeeze()  # 如果每个向
        torch.save(final_tensor, kc_file)
    else:
        final_tensor = torch.load(kc_file)
    for i in range(1):
        logging.getLogger().setLevel(logging.INFO)
        cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='ncf2', dim=20,KCs=final_tensor)
        cdm.train(train_set, train_AMR_set,valid_set,valid_AMR_set, epoch_n=30, device="cuda:2", lr=0.0002)
        cdm.save("kancd.snapshot")

        cdm.load("kancd.snapshot")

        file_dir = "../../data/" + data_name
        file_name_valid = file_dir + '/' +  'output_valid' + '.csv'
        file_name_test = file_dir + '/' + 'output_test' + '.csv'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)


        auc, accuracy = cdm.eval(valid_set,valid_AMR_set, device="cuda:2",output_file=file_name_valid,visual=False)
        auc, accuracy = cdm.eval(test_set,test_AMR_set, device="cuda:2",output_file=file_name_test,visual=False)
        auc_all += auc
        acc_all += accuracy
        print("auc:{},acc:{}".format(auc, accuracy))

    auc_ave = auc_all / 1
    acc_ave = acc_all / 1
    print(data_name)
    print("auc_ave: %.6f, accuracy_ave: %.6f" % (auc_ave, acc_ave))



def train_until_success(model_name, max_retries=500, delay=3):
    retries = 0
    while retries < max_retries:
        try:
            train(model_name)  # 尝试运行 train 函数
            print("训练成功！")
            break  # 如果没有抛出异常，表示训练成功，退出循环
        except Exception  as e:  # 捕获超时异常
            print(f"发生超时错误: {e}")
            retries += 1  # 增加重试次数
            if retries < max_retries:
                print(f"准备重新尝试... ({retries}/{max_retries})")
                time.sleep(delay)  # 等待一段时间后再重试
            else:
                print("重试次数已用尽，无法成功训练模型。")
                break  # 如果超过最大重试次数，退出循环
if __name__ == "__main__":

    # 调用函数
    # train_until_success("elion")

    train("elion")
    # train("IRE_mathdial")
    # train("IRE_Comta")
    # train("comta")


