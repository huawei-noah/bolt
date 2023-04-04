import torch
import torch.nn as nn
from torch.autograd import Function
import torch.onnx
import torch.nn.functional as F

class FpixAdaConv(Function):
    @staticmethod
    def forward(ctx, x, values, keys, queries, s, k):
        B, C, H, W = x.shape
        B, L, H, W = queries.shape 

        pad = k // 2
        x_pad = F.pad(x, pad=(pad, pad, pad, pad), mode='reflect')

        queries = queries.permute(0,2,3,1) #[B, H, W, L]
        sim = torch.einsum("BHWL, nL ->BHWn", queries, keys) 
        indices = torch.argmax(sim, dim=3) #[B, H, W]
        print("indices.shape", indices.shape)

        out = torch.empty_like(x.unsqueeze(2).repeat(1,1,s**2,1,1)) # [B, C, S(s^2), H, W]
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    index = indices[b,h,w]
                    value = values[index] #S(s^2), K(k^2)
                    input_patch = x_pad[b, :, h:h+k, w:w+k].reshape(C,-1) #C, K(k^2)
                    result = torch.einsum("CK, SK->CS", input_patch, value) #C, S(s^2)
                    out[b, :, :, h, w] = result
        out = F.pixel_shuffle(out, s).squeeze(2)
        return out

    @staticmethod
    def symbolic(g, x, values, keys, queries, s, k):
        pad = k // 2
        kwargs = {"dilations_i": [1, 1],
                  "group_i": 1,
                  "kernel_shape_i": [k, k],
                  "pads_i": [pad, pad, pad, pad],
                  "pad_mode_s": 'reflect',
                  "strides_i": [1, 1]}
        return g.op('ConvPixAda', x, values, keys, queries, **kwargs)

pixAdaConv = FpixAdaConv.apply

class PixAdaConvNet(nn.Module):
    #s = int(math.sqrt(self.values.shape[-2]))
    #k = int(math.sqrt(self.values.shape[-1]))
    def __init__(self, s=4, k=5, n=3000, L=72):
        super(PixAdaConvNet, self).__init__()

        self.keys = torch.randn(n, L).cuda()  # [n, L]
        print("keys shape:", self.keys.shape)

        self.values = torch.randn(n, s**2, k**2).cuda()   # [n, S(s^2), K(k^2)]
        print("values shape:", self.values.shape)

        self.s = s
        self.k = k
        print("upscale factor:", self.s, ", kernel size", self.k)

    def forward(self, x, queries):
        out = pixAdaConv(x, self.values, self.keys, queries, self.s, self.k)
        return out

L=72
net = PixAdaConvNet().cuda()
x = torch.ones(1,3,100,100).cuda()
queries = torch.ones(1,L,100,100).cuda()
torch.onnx.export(net, (x,queries), 'PixAdaConvNet.onnx', opset_version=13, enable_onnx_checker=False)
