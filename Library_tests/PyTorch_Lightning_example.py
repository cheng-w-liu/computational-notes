import torch
from torch import nn
import lightning as L
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

import os

class MNISTDataModule(L.LightningDataModule):

   def __init__(self):
       super(MNISTDataModule, self).__init__()
       self.data_root_path = os.path.join(os.path.expanduser("~"), 'ml_datasets')
       self.batch_size = 128
       self.transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])

   def setup(self, stage=None):
       tmp_dataset = MNIST(self.data_root_path, train=True, download=True, transform=self.transform)
       self.train_dataset, self.val_dataset = random_split(tmp_dataset, [0.9, 0.1])
       self.test_dataset = MNIST(self.data_root_path, train=False, download=True, transform=self.transform)       
  
   def train_dataloader(self):
       return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
  
   def val_dataloader(self):
       return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
  
   def test_dataloader(self):
       return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class MNISTClassifier(L.LightningModule):

   def __init__(self):
       super().__init__()

       self.layer_1_size = 120
       self.layer_2_size = 84
       self.lr = 1e-3

       self.conv_blocks = nn.Sequential(
           nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), # (1, 28, 28) --> (6, 24, 24),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2), # (6, 24, 24) --> (6, 12, 12)
           nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), # (6, 12, 12) --> (16, 8, 8)
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2), # (16, 8, 8) --> (16, 4, 4)                       
       )

       self.fc_blocks = nn.Sequential(
           nn.Linear(256, self.layer_1_size),
           nn.ReLU(),
           nn.Linear(self.layer_1_size, self.layer_2_size),
           nn.ReLU(),
           nn.Linear(self.layer_2_size, 10),
       )

   def forward(self, x):
       batch_size = x.size(0)
       x = self.conv_blocks(x)
       x = x.view(batch_size, -1)
       x = self.fc_blocks(x)
       x = F.log_softmax(x, dim=1)
       return x
  
   def cross_entropy_loss(self, logits, labels):
       return F.nll_loss(logits, labels)
  
   def training_step(self, train_batch, batch_idx):
       x, y = train_batch
       logits = self.forward(x)
       loss = self.cross_entropy_loss(logits, y)
       self.log('train_loss', loss)
       return loss
  
   def validation_step(self, val_batch, batch_idx):
       x, y = val_batch
       logits = self.forward(x)
       loss = self.cross_entropy_loss(logits, y)
       self.log('val_loss', loss)

   def test_step(self, test_batch, batch_idx):
       x, y = test_batch
       logits = self.forward(x)
       loss = self.cross_entropy_loss(logits, y)
       self.log('val_loss', loss)

   def configure_optimizers(self):
       optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
       return optimizer


data_module = MNISTDataModule()
model = MNISTClassifier()

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, data_module)
