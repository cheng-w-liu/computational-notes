import os
import tempfile

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST
from torchvision import transforms

import ray.train.torch

def train_func():
   lr = 1e-3
   batch_size = 128
   num_epochs = 10

   # model, loss_func, optimizer
   model = resnet18(num_classes=10)
   model.conv1 = torch.nn.Conv2d(
       in_channels=1,
       out_channels=64,
       kernel_size=7,
       stride=2,
       padding=3,
       bias=False
   )

   # prepare/accelerate utility functions should be
   # called inside a training function executed by
   # `Trainer.run`
   model = ray.train.torch.prepare_model(model)

   loss_func = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=lr)

   # data
   transform = transforms.Compose(
       [
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,)),
       ]
   )
   data_root_dir = os.path.join(os.path.expanduser("~"), 'ml_datasets')
   train_dataset = FashionMNIST(
       root=data_root_dir,
       train=True,
       download=True,
       transform=transform
   )

   train_loader = DataLoader(
       dataset=train_dataset,
       batch_size=batch_size,
       shuffle=True,
   )

   # prepare/accelerate utility functions should be
   # called inside a training function executed by
   # `Trainer.run`
   train_loader = ray.train.torch.prepare_data_loader(train_loader)

   for epoch_idx in range(num_epochs):
       if ray.train.get_context().get_world_size() > 1:
           train_loader.sampler.set_epoch(epoch_idx)
      
       for images, labels in train_loader:
           outputs = model(images)
           loss = loss_func(outputs, labels)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
      
       metrics = {"loss": loss.item(), "epoch": epoch_idx}
       with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
           if ray.train.get_context().get_world_size() > 1:
               torch.save(
                   model.module.state_dict(), # .module: due to model wrapped in nn.DistributedDataParallel
                   os.path.join(temp_checkpoint_dir, "model.pt")
               )
           else:
               torch.save(
                   model.state_dict(), # .module: due to model wrapped in nn.DistributedDataParallel
                   os.path.join(temp_checkpoint_dir, "model.pt")
               )

           ray.train.report(
               metrics,
               checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
           )
      
       if ray.train.get_context().get_world_rank() == 0:
           print(f"epoch: {epoch_idx}, metrics: {metrics}")


scaling_config = ray.train.ScalingConfig(num_workers=1, use_gpu=True)

trainer = ray.train.torch.TorchTrainer(
   train_func,
   scaling_config=scaling_config
)
result = trainer.fit()


with result.checkpoint.as_directory() as checkpoint_dir:
   model_state_dict = torch.load(
       os.path.join(checkpoint_dir, "model.pt"),
       weights_only=True
   )
   model = resnet18(num_classes=10)
   model.conv1 = torch.nn.Conv2d(
       in_channels=1,
       out_channels=64,
       kernel_size=7,
       stride=2,
       padding=3,
       bias=False
   )
   model.load_state_dict(model_state_dict)
