import torch
import wandb

from models import ViT
from utils import Mean, SparseCategoricalAccuracy

def make_model(config_dict):
    model = ViT(
        num_classes=config_dict['num_classes'],
        img_size=config_dict['img_size'],
        patch_size=config_dict['patch_size'],
        in_channels=config_dict['in_channels'],
        embed_dim=config_dict['embed_dim'],
        pos_drop_rate=config_dict['pos_drop_rate'],
        num_blocks=config_dict['num_blocks'],
        num_heads=config_dict['num_heads'],
        qkv_bias=config_dict['qkv_bias'],
        attn_drop_rate=config_dict['attn_drop_rate'],
        proj_drop_rate=config_dict['proj_drop_rate'],
        mlp_ratio=config_dict['mlp_ratio'],
        mlp_drop_rate=config_dict['mlp_drop_rate']
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=config_dict['base_lr'],
        max_lr=config_dict['max_lr'],
        mode='triangular2'
    )

    return model, optimizer, scheduler, loss_func


def run_one_epoch(dataloader, model, optimizer, scheduler, loss_func, training=True):
    loss_metric = Mean()
    acc_metric = SparseCategoricalAccuracy()
    for images, labels in dataloader:
        if training:
            optimizer.zero_grad()
            logits = model(images)
            batch_loss = loss_func(logits, labels)
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            with torch.no_grad():
                logits = model(images)
                batch_loss = loss_func(logits, labels)

        loss_metric(batch_loss)
        acc_metric(logits, labels)

    return loss_metric.result(), acc_metric.result()


def train_model(train_dataloader, test_dataloader, model, optimizer, scheduler, loss_func, config_dict):
    wandb.watch(model, loss_func, log="all", log_freq=1)

    for epoch_idx in range(config_dict['num_epochs']):
        model.train()
        train_loss, train_acc = run_one_epoch(train_dataloader, model, optimizer, scheduler, loss_func, True)

        model.eval()
        test_loss, test_acc = run_one_epoch(test_dataloader, model, None, None, loss_func, False)

        wandb.log(
            {'epoch': epoch_idx,
             'train_loss': train_loss,
             'train_acc': train_acc,
             'test_loss': test_loss,
             'test_acc': test_acc,
             'lr': scheduler.get_last_lr()[0]
             },
            step=epoch_idx
        )

        if (epoch_idx + 1) % 2 == 0:
            print(
                f'finished epoch {epoch_idx}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f},  test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}, last_lr: {scheduler.get_last_lr()[0]:.5f}'
            )


def run(train_dataloader, test_dataloader, config_dict, path_to_model=None):
    with wandb.init(project='CIFAR10-ViT-PyTorch', config=config_dict):
        model, optimizer, scheduler, loss_func = make_model(config_dict)
        train_model(train_dataloader, test_dataloader, model, optimizer, scheduler, loss_func, config_dict)

        if path_to_model is not None:
            torch.save(model.state_dict(), path_to_model)
            loaded_mode, _, _, _ = make_model(config_dict)
            loaded_mode.load_state_dict(torch.load(path_to_model))

