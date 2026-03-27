import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import GraphConvolution, Linear
from utils import generate_cheby_adj, normalize_A
from layers import *
from timm.layers import DropPath, trunc_normal_
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CapsuleConvLayer, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv0(x))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze operation
        y = self.fc(y).view(b, c, 1, 1)  # Excitation operation
        return x * y.expand_as(x)

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_channels,
                              out_channels=32,
                              kernel_size=9,
                              stride=1,
                              padding=4,
                              bias=True)

    def forward(self, x):
        return self.conv0(x)

class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq+1e-8)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        u = [self.units[i](x) for i in range(self.num_units)]
        u = torch.cat(u, dim=1)
        u = torch.cat((x, u), dim=1)
        return u

    def routing(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).cuda()
        
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = CapsuleLayer.squash(s_j)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)
class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()   #https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def forward(self, x,L): 
        device = x.device
        adj = generate_cheby_adj(L, self.K, device)     #生成切比雪夫多项式邻接矩阵
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)     #对累加结果应用relu激活函数处理
        return result
class ContrastiveLoss(nn.Module):
    def __init__(self,device='cuda', temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        #self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        batch_size,_ = emb_i.shape
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(emb_i.device)).float()
        pos_idx = torch.arange(batch_size, device=representations.device)
        sim_ij = similarity_matrix[pos_idx, pos_idx + batch_size]  # 正样本对 (emb_i 和 emb_j)
        sim_ji = similarity_matrix[pos_idx + batch_size, pos_idx]  # 反向正样本对 (emb_j 和 emb_i)
        positives = torch.cat([sim_ij, sim_ji], dim=0)                 # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss

class DGCNN(nn.Module):
    def __init__(self, eeg_xdim=(208,31,5),fnirs_xdim = (208,18,5), k_adj=2, num_out=12, nclass=2, n_layers=2,n_heads=4,d_model=12,d_k=16,d_v=16,d_ff=64,fc_num = 12,temporature = 0.5,target = 12,fc1 = 256,fc2 = 64,dropout_p = 0.5,in_unit=64,unit_size=12):
        #xdim: (batch_size*num_nodes*num_features_in)
        #k_adj: num_layers/切比雪夫多项式的阶数
        #num_out: num_features_out/输出特征的数量
        super(DGCNN, self).__init__()
        self.K = k_adj
        
        #-------fnirs-dgcnn---------------
        self.layer_fnirs = Chebynet(fnirs_xdim, k_adj, num_out)
        self.BN_fnirs = nn.BatchNorm1d(fnirs_xdim[2])  #对第二维（第一维为batch_size)进行标准化 xdim[2]=5
        self.A_fnirs = nn.Parameter(torch.FloatTensor(fnirs_xdim[1], fnirs_xdim[1]))
        nn.init.xavier_normal_(self.A_fnirs)
        
        #-------eeg-dgcnn-----------------
        self.layer_eeg = Chebynet(eeg_xdim, k_adj, num_out)
        self.BN_eeg = nn.BatchNorm1d(eeg_xdim[2])  #对第二维（第一维为batch_size)进行标准化 xdim[2]=5
        self.A_eeg = nn.Parameter(torch.FloatTensor(eeg_xdim[1], eeg_xdim[1]))
        nn.init.xavier_normal_(self.A_eeg)
        
        
        #--------cross-attention-----------------------
        self.transformer1 = Encoder(n_layers,n_heads,d_model,d_k,d_v,d_ff)
        self.transformer2 = Encoder(n_layers,n_heads,d_model,d_k,d_v,d_ff)
        
        #----------对比学习---------
        # target = 32
        # EEG投影头
        self.eeg_projector = nn.Sequential(
            nn.Conv1d(31, target, kernel_size=1),
            nn.BatchNorm1d(target),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten()
        )
        
        # fNIRS投影头
        self.fnirs_projector = nn.Sequential(
            nn.Conv1d(18, target, kernel_size=1),
            nn.BatchNorm1d(target),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 对比损失模块
        self.contrastive_loss = ContrastiveLoss(temperature = temporature)

        #胶囊网络

        self.digits = CapsuleLayer(in_units=num_out,
                                   in_channels=49, 
                                   num_units=2,
                                   unit_size=unit_size,
                                   use_routing=True)

        
    
    def forward(self, eeg,fnirs):
        #-------eeg dgcnn------------
        eeg = self.BN_eeg(eeg.transpose(1, 2)).transpose(1, 2)   #因为第三维 才为特征维度,batchnorm1d因为是1d所以相当于在第2维作batchnorm（从1维开始计数，第一维为bs)
        L_eeg = normalize_A(self.A_eeg)   #A是自己设置的59 * 59的可训练参数  及邻接矩阵     #邻接矩阵进行归一化，得到归一化的拉普拉斯矩阵
        result_eeg = self.layer_eeg(eeg, L_eeg)
        #-------fnirs-dgcnn---------
        fnirs = self.BN_fnirs(fnirs.transpose(1, 2)).transpose(1, 2)   #因为第三维 才为特征维度,batchnorm1d因为是1d所以相当于在第2维作batchnorm（从1维开始计数，第一维为bs)
        L_fnirs = normalize_A(self.A_fnirs)   #A是自己设置的59 * 59的可训练参数  及邻接矩阵     #邻接矩阵进行归一化，得到归一化的拉普拉斯矩阵
        result_fnirs = self.layer_fnirs(fnirs, L_fnirs)
        
        #-----------对比学习----------------------
        # 对比学习特征投影
        proj_eeg = self.eeg_projector(result_eeg)  # [batch, target_channels]
        proj_fnirs = self.fnirs_projector(result_fnirs)  # [batch, target_channels]
        #print(proj_eeg.shape)
        #print(proj_fnirs.shape)
        # 计算对比损失
        contrast_loss = self.contrastive_loss(proj_eeg, proj_fnirs)
        
        #---------cross-model---------
        #print(torch.diag(self.A_eeg).unsqueeze(-1).shape)
        result_eeg = result_eeg + torch.sigmoid(torch.mean(self.A_eeg, dim=1).unsqueeze(-1))
        result_fnirs = result_fnirs + torch.sigmoid(torch.mean(self.A_fnirs, dim=1).unsqueeze(-1))
        result1,attn1 = self.transformer1(result_eeg,result_fnirs)  #输出应为bs，62，64
        result2,attn2 = self.transformer2(result_fnirs,result_eeg)
        #print(result1.shape)
        #----------classify---------
        result = torch.cat((result1,result2),axis=1)
        result = result.transpose(1,2)#bs,12,49
        #print(result.shape)
        result = CapsuleLayer.squash(result)
        result = self.digits(result)
       
        return result,contrast_loss#,attn1,attn2
    def loss(self, output, target, size_average=True):
        return self.margin_loss(output, target, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))
        zero = Variable(torch.zeros(1)).cuda()
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2
        
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)
        
        if size_average:
            L_c = L_c.mean()
        return L_c
    def forward_for_tsne(self, eeg, fnirs):
        #-------eeg dgcnn------------
        eeg = self.BN_eeg(eeg.transpose(1, 2)).transpose(1, 2)   #因为第三维 才为特征维度,batchnorm1d因为是1d所以相当于在第2维作batchnorm（从1维开始计数，第一维为bs)
        L_eeg = normalize_A(self.A_eeg)   #A是自己设置的59 * 59的可训练参数  及邻接矩阵     #邻接矩阵进行归一化，得到归一化的拉普拉斯矩阵
        result_eeg = self.layer_eeg(eeg, L_eeg)
        #-------fnirs-dgcnn---------
        fnirs = self.BN_fnirs(fnirs.transpose(1, 2)).transpose(1, 2)   #因为第三维 才为特征维度,batchnorm1d因为是1d所以相当于在第2维作batchnorm（从1维开始计数，第一维为bs)
        L_fnirs = normalize_A(self.A_fnirs)   #A是自己设置的59 * 59的可训练参数  及邻接矩阵     #邻接矩阵进行归一化，得到归一化的拉普拉斯矩阵
        result_fnirs = self.layer_fnirs(fnirs, L_fnirs)
        
        #---------cross-model---------
        result_eeg = result_eeg + torch.sigmoid(torch.mean(self.A_eeg, dim=1).unsqueeze(-1))
        result_fnirs = result_fnirs + torch.sigmoid(torch.mean(self.A_fnirs, dim=1).unsqueeze(-1))
        result1,attn1 = self.transformer1(result_eeg,result_fnirs)  #输出应为bs，62，64
        result2,attn2 = self.transformer2(result_fnirs,result_eeg)
        #print(result1.shape)
        #----------classify---------
        features = torch.cat((result1,result2),axis=1)
        
        return features
'''
import os
import copy
if __name__ =='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TORCH_HOME'] = './' #setting the environment variable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #input : sample/batch_size,channel,fetures   64 62 18
    eeg = torch.randn(4,31,5).to(device)
    fnirs = torch.randn(4,18,5).to(device)
    model = DGCNN(eeg_xdim=[4,31,5],fnirs_xdim=[4,18,5]).to(device)
    out,ls = model.forward(eeg,fnirs)
    print(out.shape)
'''
