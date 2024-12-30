import tempfile
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from ray.tune.schedulers import ASHAScheduler


import ray
from ray import train, tune




class MNISTClassifier(nn.Module):
   def __init__(self, config):
       super(MNISTClassifier, self).__init__()


       self.layer_1_size = config["layer_1_size"]
       self.layer_2_size = config["layer_2_size"]       


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
  
   def forward(self, x):
       batch_size = x.size(0)
       x = self.conv_blocks(x)
       x = x.view(batch_size, -1)
       x = self.fc_blocks(x)
       x = F.log_softmax(x, dim=1)
       return x




def load_dataset(mode="train"):
   assert mode in ["train", "test"], "mode should be either train or test"
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])
   data_root_path = os.path.join(os.path.expanduser("~"), 'ml_datasets')
   dataset = MNIST(
       data_root_path,
       train=mode == "train",
       download=True,
       transform=transform
   )
   return dataset




def make_model(config, device):
   model = MNISTClassifier(config).to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
   loss_func = F.nll_loss
   return model, optimizer, loss_func


def run_one_epoch(device, data_loader, model, optimizer, loss_func, training=True):
   if training:
       assert optimizer is not None
       model.train()
   else:
       assert optimizer is None
       model.eval()
  
   losses = []
   correct = 0
   total = 0
   with torch.set_grad_enabled(training):
       for x, y in data_loader:
           if training:
               optimizer.zero_grad()
           x, y = x.to(device), y.to(device)
           logits = model(x)
           loss = loss_func(logits, y)
           if training:
               loss.backward()
               optimizer.step()
           losses.append(loss.detach())
           correct += (torch.argmax(logits, dim=1) == y).sum().item()
           total += len(y)


   metrics = {
       "loss": torch.stack(losses).mean().item(),
       "accuracy": float(correct) / float(total),
   }
   return metrics


def train_one_mode(config):
   device = config["device"]
   model, optimizer, loss_func = make_model(config, device)
   tmp_dataset = load_dataset(mode="train")
   train_dataset, val_dataset = random_split(tmp_dataset, [0.9, 0.1])


   train_loader = DataLoader(
       train_dataset,
       batch_size=config["batch_size"],
       shuffle=True,
   )


   val_loader = DataLoader(
       val_dataset,
       batch_size=config["batch_size"],
       shuffle=True,
   )


   for epoch_idx in range(config["num_epochs"]):
       train_metrics = run_one_epoch(
           device, train_loader, model, optimizer, loss_func, training=True
       )
       train_loss = train_metrics["loss"]
       train_accuracy = train_metrics["accuracy"]


       val_metrics = run_one_epoch(
           device, val_loader, model, None, loss_func, training=False
       )
       val_loss = val_metrics["loss"]
       val_accuracy = val_metrics["accuracy"]


       print(f"Epoch {epoch_idx + 1}:")
       print(f"  Train: loss={train_loss:.5f}, accuracy={train_accuracy:.5f}")
       print(f"  Val: loss={val_loss:.5f}, accuracy={val_accuracy:.5f}")
       with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
           path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
           torch.save(
               (model.state_dict(), optimizer.state_dict()),
               path,
           )


           checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
           train.report(
               {"loss": val_loss, "accuracy": val_accuracy},
               checkpoint=checkpoint
           )
   print("Finished training")


def test_best_model(device, best_result: ray.train.Result) -> None:
   # model = MNISTClassifier(best_result.config)
   model, _optimizer, loss_func = make_model(best_result.config, device)
   model.to(device)


   checkpoint_path = os.path.join(
       best_result.checkpoint.to_directory(),
       "checkpoint.pt"
   )


   model_state, _optimizer_state = torch.load(checkpoint_path)
   model.load_state_dict(model_state)
  
   test_dataset = load_dataset(mode="test")
   test_loader = DataLoader(
       test_dataset,
       batch_size=best_result.config["batch_size"],
       shuffle=False,
   )


   metrics = run_one_epoch(
       device, test_loader, model, None, loss_func, training=False
   )
   loss = metrics["loss"]
   accuracy = metrics["accuracy"]
   print("Best config test set: loss={:.5f}, accuracy={:.5f}".format(loss, accuracy))



def main(num_samples=5, max_num_epochs=10):
   search_space = {
       "layer_1_size": tune.randint(80, 200),
       "layer_2_size": tune.randint(80, 200),
       "lr": tune.loguniform(1e-5, 1e-1),
       "batch_size": tune.choice([32, 64, 128]),
       "num_epochs": 10,
       "device": device,
   }


   scheduler = ASHAScheduler(
       max_t=max_num_epochs,
       grace_period=4,
       reduction_factor=2,
   )


   tuner = tune.Tuner(
       tune.with_resources(
           tune.with_parameters(train_one_mode),
           resources={"cpu": 2, "gpu": 1},
       ),
       tune_config=tune.TuneConfig(
           metric="loss",
           mode="min",
           num_samples=num_samples,
           scheduler=scheduler,
       ),
       param_space=search_space,
   )
   results = tuner.fit()
   return results
  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_config = {
   "layer_1_size": 120,
   "layer_2_size": 74,
   "lr": 1e-3,
   "batch_size": 128,
   "num_epochs": 10,
   "device": device
}

results = main(num_samples=8)

best_result = results.get_best_result("loss", "min")
print(f"Best trial config: {best_result.config}")
print("Best trial final validation loss: {:.5f}".format(best_result.metrics["loss"]))
print("Best trial final validation accuracy: {:.5f}".format(best_result.metrics["accuracy"]))

test_best_model(device, best_result)
