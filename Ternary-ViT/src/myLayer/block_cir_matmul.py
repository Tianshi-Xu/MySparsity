import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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
        self.weight = nn.Parameter(torch.Tensor(self.p, self.q, self.block_size))
        init.xavier_uniform_(self.weight)
        # 初始化偏置参数
        # self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x):
        # 实现前向传播
        batch_size=x.shape[0]
        # print("x1:",x.shape)
        H = x.shape[2]
        x = F.unfold(x, self.kernel_size, stride=self.stride,padding=self.kernel_size//2)
        # print("unfold x2:",x.shape)
        padd_x_size = self.q * self.block_size - self.in_features
        # import pdb;pdb.set_trace()
        # print("padd_x_size:",padd_x_size)
        x = F.pad(x, [0,0,0,padd_x_size], "constant", 0)
        # print("after pad:",x.shape)
        x = x.reshape(-1,self.q,self.block_size)
        # print("x1:",x.shape)
        
        x = torch.fft.rfft(x)
        w = torch.fft.rfft(self.weight)
        x = torch.unsqueeze(x,1)
        # print("before tile x:",x.shape)
        # x = torch.tile(x,[1,self.p,1,1])
        # print("x2:",x.shape)
        # print("w:",w.shape)
        h = torch.sum(x * w,axis=2)
        h = torch.fft.irfft(h)
        h = torch.reshape(h,(-1,self.p * self.block_size))
        # x = torch.complex(x, torch.zeros_like(x))
        # w = torch.complex(self.weight[...,0], self.weight[...,1])
        # x = torch.fft.fft(x)
        # w = w
        # x = torch.unsqueeze(x,1)
        # x = torch.tile(x,[1,self.p,1,1])
        # print("x2:",x.shape)
        # h = torch.sum(x * w,axis=2)

        # h = torch.fft.ifft(h)
        
        # h = torch.real(h)
        # h = torch.reshape(h,(-1,self.p * self.block_size))

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
    
class NewBlockCirculantConv(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, block_size):
        super(NewBlockCirculantConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.block_size = block_size
        self.padding = kernel_size//2
        self.p = self.in_features//block_size
        self.q = self.out_features//block_size
        
        if(self.in_features % block_size > 0):
            print("self.in_features % block_size:",self.in_features % block_size)
            self.p += 1
        
        if(self.out_features % block_size > 0):
            print("self.out_features % block_size:",self.out_features % block_size)
            self.q += 1

        # 初始化权重参数
        self.weight = nn.Parameter(torch.zeros(self.q, self.p, self.block_size,self.kernel_size,self.kernel_size))
        
        init.kaiming_uniform_(self.weight)
        # 初始化偏置参数
        # self.bias = nn.Parameter(torch.Tensor(out_features))

    def forward(self, x):
        # 实现前向传播
        w = torch.zeros(self.q,self.p,self.block_size,self.block_size,self.kernel_size,self.kernel_size).cuda()
        # print(self.weight[0,0,:,0,0])
        for i in range(self.block_size):
            w[:,:,:,i,:,:] = self.weight.roll(shifts=i, dims=2)
        # print(w[0,0,:,:,0,0])
        # print(w.shape)
        w = w.permute(0,2,1,3,4,5)
        # print(w.shape)
        w = w.reshape(self.q*self.block_size,self.p*self.block_size,self.kernel_size,self.kernel_size)
        
        return F.conv2d(x,w,None,self.stride,self.padding)
    
    def __str__(self):
        additional_info = "block_size: " + str(self.block_size)
        return super(BlockCirculantConv, self).__str__() + "\n" + additional_info

class LearnableCir(nn.Module):
    
    def __init__(self, in_features, out_features, kernel_size, stride,feature_size,pretrain,finetune=False):
        super(LearnableCir, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.feature_size = feature_size
        self.pretrain = pretrain
        self.finetune = finetune
        self.padding = kernel_size//2
        self.tau = 1.0
        self.search_space = []
        search=2
        while search<=16 and in_features %search ==0 and out_features %search ==0 and feature_size*feature_size*search <= 4096:
            self.search_space.append(search)
            search *= 2
        if pretrain:
            self.alphas = nn.Parameter(torch.ones(len(self.search_space)+1), requires_grad=False)
        else:
            self.alphas = nn.Parameter(torch.ones(len(self.search_space)+1), requires_grad=True)
        self.weight = nn.Parameter(torch.zeros(out_features,in_features, kernel_size,kernel_size))
        self.alphas_after = None
        init.kaiming_uniform_(self.weight)
        print("in_features,out_features,kernel_size,feature_size,search_space:",in_features,out_features,kernel_size,feature_size,self.search_space)
    
    def trans_to_cir(self):
        search_space = self.search_space
        alphas_after = gumbel_softmax(self.alphas,tau=self.tau,hard=False,finetune=self.finetune)
        # weight=torch.zeros(self.out_features,self.in_features, self.kernel_size,self.kernel_size).cuda()
        weight=alphas_after[0]*self.weight
        for idx,block_size in enumerate(search_space):
            if torch.abs(alphas_after[idx+1]) <1e-6:
                continue
            # print("block_size:",block_size)
            q=self.out_features//block_size
            p=self.in_features//block_size
            # print(self.weight.shape)
            tmp = self.weight.reshape(q,block_size, p, block_size, self.kernel_size,self.kernel_size)
            tmp = tmp.permute(0,2,1,3,4,5)
            # if block_size == 8:
            #     print("---------")
            #     print(tmp[0,0,:,:,0,0])
            #     print("---------")
            # tmp = torch.mean(tmp, dim=3, keepdim=False)
            w = torch.zeros(q,p,block_size,block_size,self.kernel_size,self.kernel_size).cuda()
            # print(tmp[0,0,:,0,0])
            tmp_compress = torch.zeros(q,p,block_size,self.kernel_size,self.kernel_size).cuda()
            for i in range(block_size):
                # print("tmp:",tmp.shape)
                diagonal = torch.diagonal(tmp, offset=i, dim1=2, dim2=3)
                if i>0:
                    diagonal2 = torch.diagonal(tmp, offset=-block_size+i, dim1=2, dim2=3)
                    diagonal = torch.cat((diagonal,diagonal2),dim=4)
                    # print("diagonal:",diagonal.shape)
                # print("diagonal:",diagonal.shape)
                assert diagonal.shape[4] == block_size
                mean_of_diagonals = torch.mean(diagonal, dim=4, keepdim=True)
                mean_of_diagonals = mean_of_diagonals.permute(0,1,4,2,3)
                tmp_compress[:,:,i,:,:] = mean_of_diagonals[:,:,0,:,:]
            for i in range(block_size):
                w[:,:,:,i,:,:] = tmp_compress.roll(shifts=i, dims=2)
            # print(w[0,0,:,:,0,0])
            w = w.permute(0,2,1,3,4,5)
            # print(w.shape)
            w = w.reshape(q*block_size,p*block_size,self.kernel_size,self.kernel_size)
            weight=weight+alphas_after[idx+1]*w
        return weight

    def get_alphas_after(self):
        return gumbel_softmax(self.alphas,tau=self.tau,hard=False,finetune=self.finetune)
    
    def set_tau(self,tau):
        self.tau = tau
    
    def forward(self, x):
        if not self.pretrain:
            weight=self.trans_to_cir()
        else:
            weight = self.weight
        return F.conv2d(x,weight,None,self.stride,self.padding)

    def __str__(self):
        additional_info = "search_space: " + str(self.search_space)
        return super(LearnableCir, self).__str__() + "\n" + additional_info
    
def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1,finetune=False) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    if finetune:
        # idx = torch.argmax(logits)
        # logits = torch.zeros_like(logits)
        # logits[idx] = 1
        return F.softmax(logits/tau, dim=dim)
    return F.softmax(logits, dim=dim)
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def comm(H,W,C,K,b):
    # print("H,W,C,b:",H,W,C,b)
    N=4096
    tmp_a = torch.tensor(math.floor(N/(H*W*b)))
    print("tmp_a:",tmp_a)
    tmp_b = torch.max(torch.sqrt(tmp_a),torch.tensor(1))-1
    print("tmp_b:",tmp_b)
    print("rotate:",math.ceil((C/b/tmp_a)*2*tmp_b))
    return torch.tensor(math.ceil((C/b/tmp_a)*2*tmp_b)+0.0238*(H*W*C*K)/(N*b))  
if __name__ == '__main__':
# 示例用法
# 输入特征维度为10，输出特征维度为5，块大小为2
    # layer = LearnableCir(32,64,1,1,8,False).cuda()
    # layer.trans_to_cir()
    print(comm(32,32,32,32*6,4))
    # block_circulant_layer = NewBlockCirculantConv(64,256,3,1,8)
    # input_data = torch.ones(10,64,16,16)  # 3个样本，每个样本有10个特征
    # output_data1 = block_circulant_layer(input_data)
    # linear = nn.Linear(11, 11)
    # output_data2=linear(input_data)
    # print(output_data1.shape)
    # print(output_data1.flatten()[0:10])
    # print(output_data2)
    # x = torch.tensor([[-5,-4,-3,-2,-1,0,1,2,3,1,2,3],])
    # w = torch.tensor([[0,1,2,-1,-2,-3,1,2,3,0,-1,-2],[0,-1,-2,0,1,2,-1,-2,-3,1,2,3],
    #                 [1,2,3,0,-1,-2,0,1,2,-1,-2,-3],[-1,-2,-3,1,2,3,0,-1,-2,0,1,2]])
    # y = torch.matmul(w,x.T)
    # print(y)
    
