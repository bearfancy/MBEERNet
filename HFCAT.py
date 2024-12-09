import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rich import print
import math
from torch.backends.cuda import sdp_kernel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlashCrossAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.qkv_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.FFN = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        self.num_heads = num_heads
        self.embed_dimension = embed_dimension

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # 判断是否有SPDA属性
        if not self.flash:
            print("警告：闪光灯注意力需要 PyTorch >= 2.0")

    def forward(self, x, y):
        # BS-batch size, seq_len-sequence length, emb_dim-embedding dimensionality (embed_dimension)
        BS, seq_len, emb_dim = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 计算batch中所有头的q、k、v，并将头向前移动以成为batch 维
        # x_c_attn = self.c_attn(x)  # 得到(BS,1,embed_dimension*3)
        q1, k1, v1 = self.qkv_attn(x).split(self.embed_dimension, dim=2)
        k11 = k1.view(BS, seq_len, self.num_heads, emb_dim // self.num_heads).transpose(1, 2)  # (BS, nh, seq_len, hs)
        q11 = q1.view(BS, seq_len, self.num_heads, emb_dim // self.num_heads).transpose(1, 2)  # (BS, nh, seq_len, hs)
        v11 = v1.view(BS, seq_len, self.num_heads, emb_dim // self.num_heads).transpose(1, 2)  # (BS, nh, seq_len, hs)

        q2, _, _ = self.qkv_attn(y).split(self.embed_dimension, dim=2)
        q22 = q2.view(BS, seq_len, self.num_heads, emb_dim // self.num_heads).transpose(1, 2)  # (BS, nh, seq_len, hs)

        # (BS, nh, seq_len, hs) x (BS, nh, hs, seq_len) -> (BS, nh, seq_len, seq_len)
        if self.flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                # 使用Flash注意力CUDA核
                y1 = F.scaled_dot_product_attention(q22, k11, v11, attn_mask=None, dropout_p=self.dropout,
                                                    is_causal=False)
                y11 = y1.transpose(1, 2).contiguous().view(BS, seq_len, emb_dim)
                _, k2, v2 = self.qkv_attn(self.FFN(y11 + q2)).split(self.embed_dimension, dim=2)
                # (BS, nh, seq_len, hs)
                k22 = k2.view(BS, seq_len, self.num_heads, emb_dim // self.num_heads).transpose(1, 2)
                v22 = v2.view(BS, seq_len, self.num_heads, emb_dim // self.num_heads).transpose(1, 2)
                y2 = F.scaled_dot_product_attention(q11, k22, v22, attn_mask=None, dropout_p=self.dropout,
                                                    is_causal=False)
                y2 = y2 + q11
        else:
            print("Flash注意力仅在PyTorch>= 2.0环境下支持")
        y2 = y2.transpose(1, 2).contiguous().view(BS, seq_len, emb_dim)  # re-assemble all head outputs side by side

        # output projection
        y2 = self.attn_dropout(self.FFN(y2))
        return y2.squeeze(1)


if __name__ == '__main__':
    # 设置超参数：
    batch_size = 32
    max_sequence_len = 1
    num_heads = 2  # 头数
    heads_per_dim = 32  # 每个头的维度
    embed_dimension = num_heads * heads_per_dim  # 输入特征的维度
    # block_size = 1024

    # 实例化我们上面的 CausalSelfAttention 类
    FAt = FlashCrossAttention(num_heads=num_heads,
                              embed_dimension=embed_dimension,
                              bias=False,
                              dropout=0.1).to(device).eval()
    print(FAt)
    # 模拟数据
    x = torch.rand(batch_size,
                   max_sequence_len,
                   embed_dimension,
                   device=device,
                   )
    y = torch.rand(batch_size,
                   max_sequence_len,
                   embed_dimension,
                   device=device,
                   )
    print(x.shape)
    output = FAt(x, y)
    print(output.shape)
