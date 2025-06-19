# coding: UTF-8
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import *
from bert_model import Model
from Trainer import Trainer

import ipdb

torch.cuda.set_device(0)

# bert_path = "bert-base-uncased/"
# tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")  # 初始化分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []  # input char ids
input_types = []  # segment ids
input_masks = []  # attention mask
label = []  # 标签
pad_size = 64  # 也称为 max_len (前期统计分析，文本长度最大值为38，取32即可覆盖99%)

pad_size2 = 10

df = pd.read_excel('train_en.xlsx')

df2 = pd.read_excel('test.xlsx')
# df2 = df2.fillna(method='ffill')
df2 = df2.ffill()  # 使用新式填充方法
tdict = df2.groupby('研究领域')['确认后翻译'].apply(list).to_dict()

# ipdb.set_trace()

labelset = {}
tmp = 0
ss = {}
ss['id'] = []
ss['type'] = []
ss['mask'] = []
tdict['其他'] = []
tdict['其他'].append('qita')
for index, row in df.iterrows():
    # print(row['成果所属研究方向'])
    # ipdb.set_trace()

    if tdict[str(row['成果所属研究方向'])][0] not in labelset.keys():
        labelset[tdict[str(row['成果所属研究方向'])][0]] = tmp
        tmp = tmp + 1
        x1 = tokenizer.tokenize(str(tdict[str(row['成果所属研究方向'])][0]))
        tokens = ["[CLS]"] + x1 + ["[SEP]"]
        # ipdb.set_trace()
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)

        if len(ids) < pad_size2:
            types = types + [1] * (pad_size2 - len(ids))  # mask部分 segment置为1
            masks = masks + [0] * (pad_size2 - len(ids))
            ids = ids + [0] * (pad_size2 - len(ids))
        else:
            types = types[:pad_size2]
            masks = masks[:pad_size2]
            ids = ids[:pad_size2]

        ss['id'].append(ids)
        ss['type'].append(types)
        ss['mask'].append(masks)

print(tmp)

for index, row in df.iterrows():
    x1 = str(row['标题']) + str(row['关键词']) + str(row['摘要'])  # str(row['成果所属研究方向']) +

    x1 = tokenizer.tokenize(x1)
    tokens = ["[CLS]"] + x1 + ["[SEP]"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * (len(ids))
    masks = [1] * len(ids)

    # 短则补齐，长则切断
    if len(ids) < pad_size:
        types = types + [1] * (pad_size - len(ids))  # mask部分 segment置为1
        masks = masks + [0] * (pad_size - len(ids))
        ids = ids + [0] * (pad_size - len(ids))
    else:
        types = types[:pad_size]
        masks = masks[:pad_size]
        ids = ids[:pad_size]
    input_ids.append(ids)
    input_types.append(types)
    input_masks.append(masks)

    assert len(ids) == len(masks) == len(types) == pad_size
    label.append(int(labelset[tdict[str(row['成果所属研究方向'])][0]]))

# ipdb.set_trace()

print()

random_order = list(range(len(input_ids)))
np.random.seed(2020)  # 固定种子
np.random.shuffle(random_order)
print(random_order[:10])

# 4:1 划分训练集和测试集
input_ids_train = np.array([input_ids[i] for i in random_order[:int(len(input_ids) * 0.8)]])
input_types_train = np.array([input_types[i] for i in random_order[:int(len(input_ids) * 0.8)]])
input_masks_train = np.array([input_masks[i] for i in random_order[:int(len(input_ids) * 0.8)]])
y_train = np.array([label[i] for i in random_order[:int(len(input_ids) * 0.8)]])
print(input_ids_train.shape, input_types_train.shape, input_masks_train.shape, y_train.shape)

input_ids_test = np.array([input_ids[i] for i in random_order[int(len(input_ids) * 0.8):]])
input_types_test = np.array([input_types[i] for i in random_order[int(len(input_ids) * 0.8):]])
input_masks_test = np.array([input_masks[i] for i in random_order[int(len(input_ids) * 0.8):]])
y_test = np.array([label[i] for i in random_order[int(len(input_ids) * 0.8):]])
print(input_ids_test.shape, input_types_test.shape, input_masks_test.shape, y_test.shape)

BATCH_SIZE = 16
train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_types_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_types_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

DEVICE = torch.device("cuda")
model = Model().to(DEVICE)
model2 = Model().to(DEVICE)
# print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

NUM_EPOCHS = 20
best_acc = 0.0
PATH = 'base_en/en_model.pth'  # 定义模型保存路径

for epoch in range(1, NUM_EPOCHS + 1):  # 3个epoch
    Trainer.train(model, model2, DEVICE, train_loader, ss, optimizer, epoch)
    acc = Trainer.test(model, model2, DEVICE, test_loader, ss)
    if best_acc < acc:
        best_acc = acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, PATH)  # 保存最优模型
    print("acc is: {:.4f}, best acc is {:.4f}n".format(acc, best_acc))