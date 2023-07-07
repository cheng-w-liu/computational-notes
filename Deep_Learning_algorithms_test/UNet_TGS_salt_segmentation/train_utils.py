import torch
import torch.nn as nn
from models import UNet

class Mean(object):

    def __init__(self):
        self.value = 0.
        self.n = 0

    def __call__(self, value):
        self.value += value
        self.n += 1

    def result(self):
        return self.value / float(self.n)

    def reset(self):
        self.value = 0.
        self.n = 0


def make_model(config_dict):
    model = UNet(
        in_channels=config_dict["in_channels"],
        out_channels=config_dict["out_channels"],
        features=config_dict["features"]
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config_dict['learning_rate'])
    loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    return model, optimizer, loss_func


def run_one_epoch(dataloader, model, optimizer, loss_func, training=True):
    if training:
        assert optimizer is not None

    loss_metric = Mean()
    for x, y in dataloader:
        if training:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                loss = loss_func(logits, y)
        loss_metric(loss.detach().data.item())

    return loss_metric.result()


def train_model(train_dataloader, test_dataloader, model, optimizer, loss_func, config_dict, path_to_model=None):

    best_loss = float('inf')

    tolerance = 0

    history = {
        'train': {'loss': []},
        'test': {'loss': []}
    }

    for epoch_idx in range(config_dict["num_epochs"]):

        model.train()
        train_loss = run_one_epoch(train_dataloader, model, optimizer, loss_func)

        model.eval()
        test_loss = run_one_epoch(test_dataloader, model, optimizer, loss_func)

        if test_loss < best_loss:
            best_loss = test_loss
            tolerance = 0

            if path_to_model is not None:
                torch.save(model.state_dict(), path_to_model)
                loaded_model, _, _ = make_model(config_dict)
                loaded_model.load_state_dict(torch.load(path_to_model))

        else:
            tolerance += 1

        history['train']['loss'].append(train_loss)
        history['test']['loss'].append(test_loss)

        if (epoch_idx + 1) % 5 == 0:
            print(
                f'finished epoch {epoch_idx}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}'
            )

        if tolerance >= config_dict["early_stop"]:
            break

    return history


def run(train_dataloader, test_dataloader, config_dict, path_to_model=None):

    model, optimizer, loss_func = make_model(config_dict)
    history = train_model(train_dataloader, test_dataloader, model, optimizer, loss_func, config_dict, path_to_model)
    return history


