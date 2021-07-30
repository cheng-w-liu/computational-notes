import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self, in_features=768, num_heads=8, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.):
        """
        :param in_features: input feature dimension
        :param num_heads: num of heads to use for the Multi-Head Attention
        :param qkv_bias: whether to introduce the bias terms in the qkv projection layer
        :param attn_drop_rate: drop out rate for the attention layer
        :param proj_drop_rate: drop out rate for the linear projection layer
        """
        super(Attention, self).__init__()
        assert in_features % num_heads == 0
        self.in_features = in_features
        self.num_heads = num_heads
        self.attn_dim = in_features // num_heads
        self.scale = self.attn_dim ** (-0.5)

        self.qkv_proj = nn.Linear(in_features, 3*in_features, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(in_features, in_features)
        self.proj_dropout = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        assert C == self.in_features
        qkv = self.qkv_proj(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.attn_dim) # (B, N, 3, num_heads, attn_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, attn_dim)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]  # (B, num_heads, N, attn_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)  # (B, num_heads, N, N)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, num_heads, N, attn_dim) ==> (B, N, num_heads * attn_dim))
        x = self.proj(x)  # (B, N, C)
        x = self.proj_dropout(x)
        return x


def test_attention():
    attn_layer = Attention(256, 8, False, 0.1, 0.1)
    x = torch.randn(1, 64, 256)
    print(f'input shape: {x.shape}')
    x = attn_layer(x)
    print(f'output shape: {x.shape}')


if __name__ == '__main__':
    test_attention()