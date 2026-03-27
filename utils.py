import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#用于对邻接矩阵的归一化，得到归一化的拉普拉斯矩阵
def normalize_A(A, symmetry=False):
    A = F.relu(A)   #对邻接矩阵relu激活，确保矩阵非负
    if symmetry:    #是否对称化？
        A = A + torch.transpose(A, 0, 1)     #A+ A的转置
        d = torch.sum(A, 1)   #对A的第1维度求和
        d = 1 / torch.sqrt(d + 1e-10)    #d的-1/2次方
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L

#生成切比雪夫邻接矩阵
def generate_cheby_adj(A, K,device):
    support = []    #存储切比雪夫多项式矩阵的列表
    for i in range(K):
        if i == 0:
            # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
            temp = torch.eye(A.shape[1])
            temp = temp.to(device)
            support.append(temp)
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support 


class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, eeg_tensor,fnirs_tensor, y_tensor):
        self.eeg = eeg_tensor
        self.fnirs = fnirs_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.eeg[index],self.fnirs[index], self.y[index]

    def __len__(self):
        return len(self.y)
