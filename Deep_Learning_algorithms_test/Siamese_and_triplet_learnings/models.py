import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EmbeddingNet(nn.Module):

    def __init__(self):
        """
             n x n    *    f x f     ===>  floor( (n + 2p -f) / s ) + 1  x  floor( (n + 2p -f) / s) + 1
           padding p      stride s


        input would be 1 x 28 x 28 images

        Conv2d
        28 x 28   *   5 x 5   ===>   floor( (28 + 0 - 5) / 1 ) + 1 = 23 + 1 = 24
         p = 0        s = 1

        MaxPool2d
        24 x 24   *   2 x 2   ===>   floor( (24 + 0 - 2)/2 ) + 1 = 12
         p = 0        s = 2

        Conv2d
        12 x 12   *   5 x 5   ===>   floor( (12 + 0 - 5)/1 ) + 1 = 8
         p = 0        s = 1

        MaxPool2d
        8 x 8    *   2 x 2   ===>  floor( (8 + 0 - 2)/2 ) + 1 = 4
        p = 0        s = 2

        """
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.PReLU(),
            nn.Linear(in_features=256, out_features=2)
        )


    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


    def get_embedding(self, x: torch.tensor) -> torch.tensor:
        return self.forward(x)


class ClassificationNet(nn.Module):

    def __init__(self, embedding_net: nn.Module, n_classes: int):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc = nn.Linear(in_features=2, out_features=n_classes)


    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        logits = F.log_softmax(self.fc(output), dim=1)
        # logits to be evaluated using NLLLoss
        return logits

    def get_embedding(self, x: torch.tensor) -> torch.tensor:
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):

    def __init__(self, embedding_net: nn.Module):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1: torch.tensor, x2: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        embeddings1 = self.embedding_net(x1)
        embeddings2 = self.embedding_net(x2)
        return (embeddings1, embeddings2)

    def get_embedding(self, x: torch.tensor) -> torch.tensor:
        return self.embedding_net(x)
