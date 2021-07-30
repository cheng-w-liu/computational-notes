import torch
import torch.nn as nn

from patch_embedding import PatchEmbedding
from attention import Attention
from mlp import MLP

class ViTBlock(nn.Module):

    def __init__(self, dim=768, num_heads=8, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.,
                 mlp_ratio=4, mlp_drop_rate=0.):
        """
        :param dim: input feature dimension
        :param num_heads: num. of heads for the multi-head attention
        :param qkv_bias: whether to introduce the bias terms in the qkv projection layer
        :param attn_drop_rate: dropout rate for the attention layer
        :param proj_drop_rate: dropout rate for the linear projection layer right after attention
        :param mlp_ratio: hidden_dimenion of the MLP layer is dim * mlp_ratio
        :param mlp_drop_rate: dropout rate for the MLP layer
        """
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn_layer = Attention(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_layer = MLP(dim, int(dim * mlp_ratio), dim, mlp_drop_rate)

    def forward(self, x):
        x = x + self.attn_layer(self.norm1(x))
        x = x + self.mlp_layer(self.norm2(x))
        return x



class ViT(nn.Module):

    def __init__(self, num_classes=1000, img_size=224, patch_size=16, in_channels=3, embed_dim=768, pos_drop_rate=0., num_blocks=4,
                 num_heads=8, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.,
                 mlp_ratio=4, mlp_drop_rate=0.):
        """
        :param num_classes: num. of classes for the classification task
        :param img_size: input image size
        :param patch_size: patch size = convolution filter size
        :param in_channels: num. of channels in the input image
        :param embed_dim: num. of output channels in the convolution step = num. of conv filters
        :param pos_drop_rate: dropout rate after the position embedding
        :param num_blocks: num. of ViT blocks
        :param num_heads: num. of heads for the multi-head attention
        :param qkv_bias: whether to introduce the bias terms in the qkv projection layer
        :param attn_drop_rate: dropout rate after the attention layer
        :param proj_drop_rate: dropout rate after the linear projection layer right after attention
        :param mlp_ratio: hidden_dimenion of the MLP layer is dim * mlp_ratio
        :param mlp_drop_rate: dropout rate after the MLP layer
        """
        super(ViT, self).__init__()
        self.patch_embed_layer = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed_layer.num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(pos_drop_rate)
        self.encoders = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, mlp_ratio, mlp_drop_rate)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed_layer(x)
        x = torch.cat(
            (self.cls_token.expand(B, -1, -1), x),
            dim=1
        )
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for block in self.encoders:
            x = block(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0]

        x = self.head(x)
        return x



def test_ViT_block():
    print('Test ViT block:')
    x = torch.randn(1, 1+64, 256)
    block = ViTBlock(256, 8, False, 0.1, 0.1, 4, 0.1)
    print(f'input shape: {x.shape}')
    x = block(x)
    print(f'output shape: {x.shape}')
    print('\n')


def test_ViT_model():
    print('Test ViT model:')
    x = torch.randn(1, 3, 32, 32)
    model = ViT(
        num_classes=10,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
        pos_drop_rate=0.1,
        num_blocks=4,
        num_heads=8,
        qkv_bias=False,
        attn_drop_rate=0.1,
        proj_drop_rate=0.1,
        mlp_ratio=4,
        mlp_drop_rate=0.1
    )
    print(f'input shape: {x.shape}')
    x = model(x)
    print(f'output shape: {x.shape}')

if __name__ == '__main__':
    test_ViT_block()
    test_ViT_model()