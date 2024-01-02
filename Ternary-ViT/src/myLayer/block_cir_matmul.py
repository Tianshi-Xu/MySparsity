import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init

class BlockCirculantLayer(nn.Module):
    def __init__(self, in_features, out_features, block_size):
        super(BlockCirculantLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.p = self.in_features//block_size
        self.q = self.out_features//block_size
        
        if(self.in_features % block_size > 0):
            self.p += 1

        if(self.out_features % block_size > 0):
            self.q += 1
        

        # 初始化权重参数
        self.weight = nn.Parameter(torch.Tensor(self.p, self.q, self.block_size))
        init.xavier_uniform_(self.weight)
        # 初始化偏置参数
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x):
        # 实现前向传播
        padd_x_size = self.p * self.block_size - self.in_features
        print(x.shape)
        x = F.pad(x, [padd_x_size,0], "constant", 0)
        print(x.shape)
        x = x.reshape(-1,self.p,self.block_size)
        
        x_freq = torch.fft.rfft(x)
        w_freq = torch.fft.rfft(self.weight)
        x_freq = torch.unsqueeze(x_freq,1)
        x_freq = torch.tile(x_freq,[1,self.q,1,1])
        h_freq = x_freq * w_freq
        h = torch.fft.irfft(h_freq)
        h = torch.sum(h,axis=2)
        h = torch.reshape(h,(-1,self.q * self.block_size))
        # x_complex = torch.complex(x, torch.zeros_like(x))
        # w_complex = torch.complex(self.weight[...,0], torch.zeros_like(self.weight[...,0]))
        # x_freq = torch.fft.fft(x_complex)
        # w_freq = w_complex
        # x_freq = torch.unsqueeze(x_freq,1)
        # x_freq = torch.tile(x_freq,[1,self.q,1,1])
        # h_freq = x_freq * w_freq
        
        # h_freq = torch.sum(h_freq,axis=2)

        # h = torch.fft.ifft(h_freq)
        
        # h = torch.real(h)
        # h = torch.reshape(h,(-1,self.q * self.block_size))

        
        if self.q * self.block_size > self.out_features:
            h = h[:, :self.out_features]


        # h += self.bias
        
        return h
# 暂时只支持1x1卷积
class BlockCirculantConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, block_size):
        super(BlockCirculantConv, self).__init__()

        self.in_features = kernel_size*kernel_size*in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.block_size = block_size
        self.q = self.in_features//block_size
        self.p = self.out_features//block_size
        
        if(self.in_features % block_size > 0):
            print("self.in_features % block_size:",self.in_features % block_size)
            self.q += 1
        
        if(self.out_features % block_size > 0):
            print("self.out_features % block_size:",self.out_features % block_size)
            self.p += 1

        # 初始化权重参数
        self.weight = nn.Parameter(torch.Tensor(self.p, self.q, self.block_size,2))
        init.xavier_uniform_(self.weight)
        # 初始化偏置参数
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x):
        # 实现前向传播
        batch_size=x.shape[0]
        # print("x1:",x.shape)
        H = x.shape[2]
        x = F.unfold(x, self.kernel_size, stride=self.stride,padding=self.kernel_size//2)
        # print("x2:",x.shape)
        padd_x_size = self.q * self.block_size - self.in_features
        # import pdb;pdb.set_trace()
        # print("padd_x_size:",padd_x_size)
        x = F.pad(x, [0,0,0,padd_x_size], "constant", 0)
        # print("after pad:",x.shape)
        x = x.reshape(-1,self.q,self.block_size)
        # print("x1:",x.shape)
        x = torch.complex(x, torch.zeros_like(x))
        w = torch.complex(self.weight[...,0], self.weight[...,1])
        x = torch.fft.fft(x)
        w = w
        x = torch.unsqueeze(x,1)
        x = torch.tile(x,[1,self.p,1,1])
        # print("x_freq:",x_freq.shape)
        # print("w_freq:",w_freq.shape)
        h = torch.sum(x * w,axis=2)

        h = torch.fft.ifft(h)
        
        h = torch.real(h)
        h = torch.reshape(h,(-1,self.p * self.block_size))

        
        if self.p * self.block_size > self.out_features:
            h = h[:, :self.out_features]

        # print("h1:",h.shape)
        # h += self.bias
        h = h.view(batch_size,H//self.stride*H//self.stride,self.out_features)
        h = h.permute(0,2,1)
        # print("h2:",h.shape)
        # h = F.fold(h, (H//self.stride,H//self.stride), kernel_size=1, stride=1)
        h = h.view(batch_size,self.out_features,H//self.stride,H//self.stride)
        return h
    
    def __str__(self):
        additional_info = "block_size: " + str(self.block_size)
        return super(BlockCirculantConv, self).__str__() + "\n" + additional_info
if __name__ == '__main__':
# 示例用法
# 输入特征维度为10，输出特征维度为5，块大小为2
    block_circulant_layer = BlockCirculantConv(64,256,3,2,2)
    input_data = torch.zeros(10,64,16,16)  # 3个样本，每个样本有10个特征
    output_data1 = block_circulant_layer(input_data)
    # linear = nn.Linear(11, 11)
    # output_data2=linear(input_data)
    print(output_data1.shape)
    print(output_data1.flatten()[0:10])
    # print(output_data2)
    # x = torch.tensor([[-5,-4,-3,-2,-1,0,1,2,3,1,2,3],])
    # w = torch.tensor([[0,1,2,-1,-2,-3,1,2,3,0,-1,-2],[0,-1,-2,0,1,2,-1,-2,-3,1,2,3],
    #                 [1,2,3,0,-1,-2,0,1,2,-1,-2,-3],[-1,-2,-3,1,2,3,0,-1,-2,0,1,2]])
    # y = torch.matmul(w,x.T)
    # print(y)

