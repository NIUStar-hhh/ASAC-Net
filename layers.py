import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from einops import rearrange
from torch.nn.utils import weight_norm

class GraphConvolution(nn.Module):

    def __init__(self, num_in, num_out, bias=False):

        super(GraphConvolution, self).__init__()

        self.num_in = num_in
        self.num_out = num_out
        # self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out).cuda())   #Pytorch nn.Parameter() 创建模型可训练参数  https://blog.csdn.net/hxxjxw/article/details/107904012
                                                                                 #torch.FloatTensor(a,b)   随机生成aXb格式的tensor

        self.weight = nn.Parameter(torch.FloatTensor(num_in, num_out))
        nn.init.xavier_normal_(self.weight)  #大致就是使可训练参数服从正态分布
        self.bias = None
        if bias:
            # self.bias = nn.Parameter(torch.FloatTensor(num_out).cuda())
            self.bias = nn.Parameter(torch.FloatTensor(num_out))
            nn.init.zeros_(self.bias)                 #有偏置 则置为0

    def forward(self, x, adj):
        out = torch.matmul(adj, x)    #矩阵/向量 乘法  邻接矩阵和输入数据矩阵乘法       
        out = torch.matmul(out, self.weight)    #上一步结果与权重矩阵作矩阵乘法
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

#dual模型中的transformer模块
# Transformer 部分
import numpy as np
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_head = 5

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        #scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context,attn#attn新改
 
class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        #.adapter = Adapter(d_model)
        
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        context ,attn= ScaledDotProductAttention(self.d_k)(Q, K, V)#attn新改
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual),attn#新改

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        #self.adapter = Adapter(d_model)
        
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        #output = self.adapter(output)
        return nn.LayerNorm(self.d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads,d_model,d_k,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff)

    def forward(self, Q,K,V):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs,attn = self.enc_self_attn(Q, K, V) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs,attn


class Encoder(nn.Module):
    def __init__(self,n_layers,n_heads,d_model,d_k,d_v,d_ff):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_heads,d_model,d_k,d_v,d_ff) for _ in range(n_layers)])

    def forward(self, m1,m2):
        enc_outputs = m1
        last_attn = None
        for layer in self.layers:
            enc_outputs ,attn= layer(enc_outputs,m2,m2)
            last_attn = attn # 更新 attn
        return enc_outputs,last_attn
#---------------------------------------------------
#-----------ECA即插即用-------------------
import torch
from torch import nn
import math
 
class ECAAttention(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel, b=1, gama=2):
        # 继承父类初始化
        super(ECAAttention, self).__init__()
 
        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size
        #print(kernel_size)
        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2
 
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()
 
    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h = inputs.shape
 
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1])
 
        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs
#-------------scsa----------即插即用
import typing as t
import torch
import torch.nn as nn
from einops import rearrange

class SCSA1D(nn.Module):
    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        down_sample_mode: str = 'avg_pool',
        attn_drop_ratio: float = 0.,
        gate_layer: str = 'sigmoid',
    ):
        super(SCSA1D, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim**-0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'Dimension must be divisible by 4'
        self.group_chans = group_chans = self.dim // 4

        # 时间维度卷积
        self.local_dwc = nn.Conv1d(group_chans, group_chans, 
                                 kernel_size=group_kernel_sizes[0],
                                 padding=group_kernel_sizes[0]//2,
                                 groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans,
                                    kernel_size=group_kernel_sizes[1],
                                    padding=group_kernel_sizes[1]//2,
                                    groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans,
                                    kernel_size=group_kernel_sizes[2],
                                    padding=group_kernel_sizes[2]//2,
                                    groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans,
                                    kernel_size=group_kernel_sizes[3],
                                    padding=group_kernel_sizes[3]//2,
                                    groups=group_chans)

        # 注意力门控
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_t = nn.GroupNorm(4, dim)  # 时间维度归一化

        # 通道注意力部分
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        
        # 1D卷积替代2D卷积
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv1d(dim, dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        # 下采样适配
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool1d(1)
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.seq_to_chans
                self.conv_d = nn.Conv1d(dim * window_size, dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool1d(kernel_size=window_size, stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool1d(kernel_size=window_size, stride=window_size)

    def seq_to_chans(self, x: torch.Tensor) -> torch.Tensor:
        """将时间维度展开到通道维度"""
        b, c, l = x.size()
        return x.view(b, c * self.window_size, l // self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: (B, C, L)
        b, c, l = x.size()
        
        # 时间注意力
        x_t = x # (B, 1, L)
        l_x_t, g_x_t_s, g_x_t_m, g_x_t_l = torch.split(x_t, self.group_chans, dim=1)
        
        # 时间注意力计算
        x_t_attn = self.sa_gate(self.norm_t(torch.cat((
            self.local_dwc(l_x_t),
            self.global_dwc_s(g_x_t_s),
            self.global_dwc_m(g_x_t_m),
            self.global_dwc_l(g_x_t_l)
        ), dim=1)))
        x_t_attn = x_t_attn.view(b, c, l)  # (B, C, L)
        
        # 应用时间注意力
        x = x * x_t_attn

        # 通道注意力
        y = self.down_func(x)  # (B, C, L')
        y = self.conv_d(y)
        _, _, l_ = y.size()
        
        # 自注意力机制
        y = self.norm(y)
        q = self.q(y)  # (B, C, L')
        k = self.k(y)
        v = self.v(y)
        
        # 重组为多头形式
        q = rearrange(q, 'b (h_n h_d) l -> b h_n h_d l', 
                     h_n=self.head_num, h_d=self.head_dim)
        k = rearrange(k, 'b (h_n h_d) l -> b h_n h_d l',
                     h_n=self.head_num, h_d=self.head_dim)
        v = rearrange(v, 'b (h_n h_d) l -> b h_n h_d l',
                     h_n=self.head_num, h_d=self.head_dim)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scaler  # (B, h_n, L', L')
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = attn @ v  # (B, h_n, h_d, L')
        
        # 重组输出
        attn = rearrange(attn, 'b h_n h_d l -> b (h_n h_d) l')
        attn = attn.mean(dim=-1, keepdim=True)  # (B, C, 1)
        attn = self.ca_gate(attn)
        
        return x * attn
#------------CA即插即用-------------
import torch
from torch import nn


class CA_Block(nn.Module):
    def __init__(self, channel, h, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
       
        self.avg_pool_x = nn.AdaptiveAvgPool1d(1)
        
        self.conv_1x1 = nn.Conv1d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv1d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
    
        self.sigmoid_h = nn.Sigmoid()
        

    def forward(self, x):

        x_h = self.avg_pool_x(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(x_h))

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_relu))
        
        out = x * s_h.expand_as(x) 
        return out
#-------------------------------AGCA-----------------
import torch
import torch.nn as nn
from torch.nn import init


class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)
        self.A0 = torch.eye(hide_channel)
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)
        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _ = y.size()
        y = y.permute(0,2,1)
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C) 
        #print(y.device)
        #print(A1.device)
        #print(self.A0.device)
        self.A0=self.A0.cuda()
        #print(self.A0.device)
        #print(self.A2.device)
        A = (self.A0 * A1) + self.A2
        y = torch.matmul(y, A)
        y = self.relu(self.conv3(y))
        y = y.permute(0,2,1)
        y = self.sigmoid(self.conv4(y))

        return x * y
#---------------eca----------------
import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(1,2)).transpose(1,2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
#-------------------
#--------------------时间注意力池化--------------------
# 注意力池化层
class AttentionPoolingLayer(nn.Module):
    def __init__(self, input_dim, num_latent_queries, num_heads):
        super(AttentionPoolingLayer, self).__init__()
        self.num_latent_queries = num_latent_queries
        self.latent_queries = nn.Parameter(torch.randn(num_latent_queries, input_dim))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
 
    def forward(self, x):
        
        B, N, C ,F= x.shape
        # 合并 C 和 F 维度，形成 (B, N, C*F)
        x = x.permute(0, 1, 3, 2)      # (B, N, F, C)
        x = x.reshape(B, N, C * F)      # (B, N, C*F)
        x = x.permute(1, 0, 2)          # (N, B, C*F)
        latent_queries = self.latent_queries.unsqueeze(1).expand(-1, B, -1)  # shape: (L, B, C)
        
        # Multihead Attention
        attn_output, _ = self.multihead_attn(latent_queries, x, x)  # shape: (L, B, C)
        attn_output = attn_output.permute(1, 0, 2)  # shape: (B, L, C)
        attn_output = attn_output.view(B, self.num_latent_queries, C, F)  # (B, L, C, F)
        return attn_output
#-----------------SEnet-------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1) #对应Excitation操作
        return x * y.expand_as(x)
    
#---------LSTM-------------------
class LSTM(nn.Module):
    def __init__(self, 
                 input_size1, 
                 hidden_size, 
                 num_layers, 
                 dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size1, hidden_size)
        self.shared_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, eeg):
        eeg = self.linear1(eeg)
        h = torch.zeros((1, eeg.size(0), self.hidden_size), dtype=eeg.dtype, device=eeg.device)
        c_e = torch.zeros((1, eeg.size(0), self.hidden_size), dtype=eeg.dtype, device=eeg.device)       
        eeg, _ = self.shared_lstm(eeg, (h, c_e))        
        eeg = self.dropout(eeg)       
        return eeg
