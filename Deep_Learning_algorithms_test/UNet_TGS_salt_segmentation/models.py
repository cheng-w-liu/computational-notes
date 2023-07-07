import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down (contracting) part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up part of UNet
        for feature in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # final output layer
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []
        for layer in self.downs:
            x = layer(x)
            # store the output _before_ down sampling
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = tvf.resize(x, size=skip_connection.shape[2:])

            x_concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x_concat_skip)

        return self.final_conv(x)

def test():
    x = torch.rand(7, 3, 101, 101)
    model = UNet(3, 1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert x.shape[0] == preds.shape[0] and x.shape[2] == preds.shape[2] and x.shape[3] == preds.shape[3]

if __name__ == '__main__':
    test()















