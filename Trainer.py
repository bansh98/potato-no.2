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
import ipdb


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(model, model2, device, train_loader, ss, optimizer, epoch):  # 训练模型
        model.train()
        best_acc = 0.0#初始化
        model2.train()
        s1, s2, s3 = torch.LongTensor(np.array(ss['id'])).to(device), torch.LongTensor(np.array(ss['type'])).to(
            device), torch.LongTensor(np.array(ss['mask'])).to(device)

        for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):#enumerate自动遍历标签
            start_time = time.time()
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            y_pred, score_pred = model([x1, x2, x3])  # 得到预测结果和评分
            final, _ = model([s1, s2, s3])
            y_pred = y_pred @ final.t()#矩阵乘法（类似注意力机制），增强分类特征
            # 计算评分标签（基于准确度）
            with torch.no_grad():
                pred = y_pred.max(-1, keepdim=True)[1]
                accuracy = pred.eq(y.view_as(pred)).float()
                # 将准确度转换为1.0-5.0的评分
                score_labels = accuracy * 4.0 + 1.0  # 1.0-5.0的连续值

            model.zero_grad()  # 手动梯度清零
            # 计算分类损失和评分损失
            loss_cls = F.cross_entropy(y_pred, y.squeeze())
            # 使用MSE损失进行回归
            loss_score = F.mse_loss(score_pred, score_labels)
            # 总损失
            loss = loss_cls + 0.5 * loss_score  # 可以调整评分损失的权重

            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:  # 打印loss
                print('Train Epoch: {} [{}/{} ({:.2f}%)]tLoss: {:.6f}, Score Loss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(x1),
                    len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.item(), loss_score.item()))
            #平衡多线程学习
    def test(model, model2, device, test_loader, ss):  # 测试模型, 得到测试集评估结果
        model.eval()
        test_loss = 0.0
        acc = 0
        score_rmse = 0.0  # 评分RMSE
        model2.eval()
        s1, s2, s3 = torch.LongTensor(np.array(ss['id'])).to(device), torch.LongTensor(np.array(ss['type'])).to(
            device), torch.LongTensor(np.array(ss['mask'])).to(device)
        final, _ = model([s1, s2, s3])

        for batch_idx, (x1, x2, x3, y) in enumerate(test_loader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            with torch.no_grad():
                y_, score_pred = model([x1, x2, x3])
                y_ = y_ @ final.t()

                # 计算评分标签
                pred = y_.max(-1, keepdim=True)[1]
                accuracy = pred.eq(y.view_as(pred)).float()
                # 将准确度转换为1.0-5.0的评分
                score_labels = accuracy * 4.0 + 1.0

                # 计算评分RMSE
                score_rmse += torch.sqrt(F.mse_loss(score_pred, score_labels)).item()

            test_loss += F.cross_entropy(y_, y.squeeze())
            pred = y_.max(-1, keepdim=True)[1]
            acc += pred.eq(y.view_as(pred)).sum().item()#统计正确的评分预测总数。

        test_loss /= len(test_loader)
        score_rmse /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Score RMSE: {:.4f}'.format(
            test_loss, acc, len(test_loader.dataset),
            100. * acc / len(test_loader.dataset),
            score_rmse))
        return acc / len(test_loader.dataset)