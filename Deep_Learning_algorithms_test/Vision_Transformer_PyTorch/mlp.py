import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_features=768, hidden_features=4*768, out_features=768, drop_rate=0., activation_layer=nn.GELU):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def test_mlp():
    x = torch.randn(1, 64, 256)
    mlp_layer = MLP(256, 4*256, 256, 0.1)
    print(f'input shape: {x.shape}')
    x = mlp_layer(x)
    print(f'output shape: {x.shape}')


if __name__ == '__main__':
    test_mlp()


