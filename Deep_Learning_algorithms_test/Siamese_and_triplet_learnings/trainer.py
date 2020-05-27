import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from metrics import Metric
from typing import List


def fit(model: nn.Module, optimizer, scheduler, loss_func, train_dataloader, test_dataloader, metrics: List[Metric], device, epochs, log_every, start_epoch=0):

    train_losses = []
    train_metrics = []
    valid_losses = []
    valid_metrics = []

    if scheduler:
        for epoch_idx in range(0, start_epoch):
            scheduler.step()

    for epoch_idx in range(1, epochs+1):

        train_epoch_loss = train_epoch(model, optimizer, loss_func, train_dataloader, metrics, device, log_every)
        train_losses.append(train_epoch_loss)
        message = f'Epoch: {epoch_idx}/{epochs}. Train set: Average loss {train_epoch_loss:.3f}'
        for metric in metrics:
            message += f'\t Average {metric.name()}: {metric.value()}'
            train_metrics.append(metric.value())

        valid_epoch_loss = test_epoch(model, loss_func, test_dataloader, metrics, device)
        valid_losses.append(valid_epoch_loss)
        message += f'\nEpoch: {epoch_idx}/{epochs}. Valid. set: Average loss {valid_epoch_loss:.3f}'
        for metric in metrics:
            message += f'\t Average {metric.name()}: {metric.value()}'
            valid_metrics.append(metric.value())

        if scheduler:
            scheduler.step()

        print(message)
        print('\n ----- \n')

    loss_df = pd.DataFrame({'train': train_losses, 'valid': valid_losses})
    metrics_df = pd.DataFrame({'train': train_metrics, 'valid': valid_metrics})
    return loss_df, metrics_df


def train_epoch(model, optimizer, loss_func, train_dataloader, metrics, device, log_every):

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    train_loss = 0.

    for batch_idx, (data, targets) in enumerate(train_dataloader):
        if device == 'cuda':
            data = data.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(data)
        batch_loss = loss_func(logits, targets)
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()
        losses.append(batch_loss.item())

        for metric in metrics:
            metric(logits, targets, batch_loss)

        if batch_idx % log_every == 0:
            batch_size = data.size(0)
            n = len(train_dataloader.dataset)
            message = f'Train: {(batch_idx+1)*batch_size}/{n}, ({100. * (batch_idx+1) / len(train_dataloader) :.1f}%\tLoss: {np.mean(losses):.3f})'
            for metric in metrics:
                message += f'\t {metric.name()}: {metric.value():.2f}'

            print(message)
            losses = []

    train_loss /= float(len(train_dataloader))
    return train_loss


def test_epoch(model, loss_func, test_dataloader, metrics, device):

    for metric in metrics:
        metric.reset()

    model.eval()
    losses = []
    test_loss = 0.
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_dataloader):
            if device == 'cuda':
                data = data.to(device)
                targets = targets.to(device)

            logits = model(data)
            batch_loss = loss_func(logits, targets)
            test_loss += batch_loss.item()
            losses.append(batch_loss.item())

            for metric in metrics:
                metric(logits, targets, batch_loss)

    test_loss /= float(len(test_dataloader))
    return test_loss

