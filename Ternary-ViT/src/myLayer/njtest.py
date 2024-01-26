import torch
import torch.nn as nn

def nj_conv():
    H=4
    W=4
    x = torch.randn(1,1,H, W)
    w = torch.ones(2,1,3, 3)
    y = nn.functional.conv2d(x, w, padding=0)
    poly_x = x.flatten()
    O=W*2+2
    poly_w = torch.zeros(H*W)
    poly_w[O]=w[0,0,0,0]
    poly_w[O-1]=w[0,0,0,1]
    poly_w[O-2]=w[0,0,0,2]
    poly_w[O-W]=w[0,0,1,0]
    poly_w[O-W-1]=w[0,0,1,1]
    poly_w[O-W-2]=w[0,0,1,2]
    poly_w[O-W*2]=w[0,0,2,0]
    poly_w[O-W*2-1]=w[0,0,2,1]
    poly_w[O-W*2-2]=w[0,0,2,2]
    output_poly=torch.zeros(O+H*W)
    poly_w = torch.tensor(poly_w)
    ntt_w = torch.fft.fft(poly_w)
    ntt_x = torch.fft.fft(poly_x)
    print(ntt_w)
    print("---------")
    ntt_y = torch.fft.ifft(ntt_w*ntt_x)
    for i in range(H*W):
        for j in range(O+1):
            output_poly[i+j] += poly_x[i]*poly_w[j]
    print(ntt_y)
    print("---------")
    print(output_poly)
    print("---------")
    print(y.flatten())
    
def nj_matmul():
    poly_x = torch.tensor([1,2,3,4])
    poly_w = torch.tensor([4,3,2,1])
    ntt_w = torch.fft.fft(poly_w)
    ntt_x = torch.fft.fft(poly_x)
    ntt_y = torch.fft.ifft(ntt_w*ntt_x)
    print(ntt_y)
    print("---------")


if __name__ == '__main__':
    nj_matmul()