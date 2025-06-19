# coding: UTF-8
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam


class Model(nn.Module):
    def __init__(self, num_classes=50):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("E:/bert/bert-base-uncased/")
        for param in self.bert.parameters():
            param.requires_grad = True
        # 分类任务的输出层
        self.fc = nn.Linear(768, num_classes)
        # 评分任务的输出层改为回归输出（单节点）
        self.score_fc = nn.Linear(768, 1)  # 回归输出，预测1.0-5.0的分数

    def forward(self, x):
        context = x[0]
        types = x[1]
        mask = x[2]
        _, pooled = self.bert(context,
                              token_type_ids=types,
                              attention_mask=mask,
                              output_all_encoded_layers=False)
        out = self.fc(pooled)
        # 使用sigmoid将输出映射到0-1范围，然后缩放到1.0-5.0
        score_out = torch.sigmoid(self.score_fc(pooled)) * 4.0 + 1.0
        return out, score_out