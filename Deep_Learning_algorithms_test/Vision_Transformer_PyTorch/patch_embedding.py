import torch
import torch.nn as nn

from utils import to_2tuple


class PatchEmbedding(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        :param img_size: input image size
        :param patch_size: patch size = convolution filter size
        :param in_channels: num. of channels in the input image
        :param embed_dim: num. of output channels in the convolution step = num. of conv filters
        """
        super(PatchEmbedding, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape   # (B, 3, 224, 224)
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.conv2d(x)   # (B, 768, 14, 14)
        x = x.flatten(start_dim=2)  # (B, 768, 196)  14*14=196
        x = x.transpose(1, 2)  # (B, 196, 768)
        return x


def test_patch_embedding():
    x = torch.randn(1, 3, 32, 32)
    patch_embed = PatchEmbedding(32, 4, 3, 256)
    print(f'input shape: {x.shape}')
    x = patch_embed(x)
    print(f'output shape: {x.shape}')


if __name__ == '__main__':
    test_patch_embedding()