# pytorch_lightning_and_ray_example.py
import os 

import lightning as L  
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchmetrics import Accuracy 
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST 
from torchvision import transforms 

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer
)
from ray import tune 
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig 
from ray.tune.schedulers import ASHAScheduler


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=128):
        super(MNISTDataModule, self).__init__()
        self.data_root_path = os.path.join(os.path.expanduser("~"), 'ml_datasets')
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        tmp_dataset = MNIST(self.data_root_path, train=True, download=True, transform=self.transform)
        self.train_dataset, self.val_dataset = random_split(tmp_dataset, [0.9, 0.1])
        self.test_dataset = MNIST(self.data_root_path, train=False, download=True, transform=self.transform)        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

class MNISTClassifier(L.LightningModule):
    def __init__(self, config):
        super(MNISTClassifier, self).__init__()
        
        self.accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)
        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config['lr']
    
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
            #nn.ReLU(),
        )

        self.eval_accuracy = []
        self.eval_loss = []

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_blocks(x)
        x = x.view(batch_size, -1)
        x = self.fc_blocks(x)
        x = F.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer 

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        accuracy = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch 
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.eval_loss.append(loss)        
        self.eval_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean().item()
        avg_acc = torch.stack(self.eval_accuracy).mean().item()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()

def train_func(config):
    data = MNISTDataModule(config["batch_size"])
    model = MNISTClassifier(config)

    trainer = L.Trainer(
        devices="auto",
        accelerator="cpu", #"auto",
        strategy=RayDDPStrategy(),
        #callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,        
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=data)

default_config = {"layer_1_size": 120, "layer_2_size": 84, "lr": 1e-3, "batch_size": 1024}
train_func(default_config)

#data = MNISTDataModule(default_config["batch_size"])
#model = MNISTClassifier(default_config)
#trainer = L.Trainer()
#trainer.fit(model, data)

search_space = {
    "layer_1_size": tune.choice([32, 64, 120, 128]),
    "layer_2_size": tune.choice([64, 84, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 256, 1024])
}

num_epochs = 5
num_samples = 10
scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

scaling_config = ScalingConfig(
    num_workers=3,
    use_gpu=True,
    resources_per_worker={"CPU": 1, "GPU": 1}   
)

checkpoint_config = CheckpointConfig(
    num_to_keep=2,
    checkpoint_score_attribute="ptl/val_accuracy",
    checkpoint_score_order="max",
)

run_config = RunConfig(
    checkpoint_config=checkpoint_config
)

ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config
)

def tune_mnist_asha(num_epochs=5, num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

results = tune_mnist_asha(num_epochs=num_epochs, num_samples=num_samples)

