import argparse
import numpy as np
import os
import pandas as pd
import time
from tensorboardX import SummaryWriter
import torch

import data
import models
import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


parser = argparse.ArgumentParser(description='Variational Autoencoder for Collaborative Filtering, MovieLens 20m')
parser.add_argument('--data_dir_root', type=str, default='ml-20m',
                    help='location of the MovieLens-20m dataset')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1500,
                    help='batch size')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

args = parser.parse_args()

# --------------------------------------------------- #
# Load data
# --------------------------------------------------- #
data_loader = data.DataLoader(os.path.join(os.path.expanduser('~/ml_datasets'), args.data_dir_root))
n_items = data_loader.n_items
# each "data" object is a csr_matrix
train_data = data_loader.load_data('train')
test_pred_data, test_eval_data = data_loader.load_data('test')
valid_pred_data, valid_eval_data = data_loader.load_data('validation')

print(f'{n_items} items')
print(f'train: {train_data.shape[0]} users')
print(f'test: {test_pred_data.shape[0]} pred users, {test_eval_data.shape[0]} eval users')
print(f'valid: {valid_pred_data.shape[0]} pred users, {valid_eval_data.shape[0]} eval users')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter()

# --------------------------------------------------- #
# Build a VAE model for CF
# --------------------------------------------------- #
dims = [n_items, 600, 200]
model = models.MultiVAE(dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
loss_func = models.loss_function

# --------------------------------------------------- #
# Model training and evaluation
# --------------------------------------------------- #
def csr_matrix2tensor(csr_matrix_data):
    return torch.FloatTensor(csr_matrix_data.toarray())

train_data_size = train_data.shape[0]
n_train_batches = train_data_size // args.batch_size
indices = np.array(range(train_data_size))
np.random.shuffle(indices)
def train(epoch_idx, batch_size, n_batches, device, beta=1.0, log_interval=20):
    # turn on training mode
    model.train()
    start_time = time.time()
    train_loss = 0.0
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, train_data_size)
        x = csr_matrix2tensor(
            train_data[indices[start_idx: end_idx], :]
        ).to(device)

        optimizer.zero_grad()
        x_tilde, mu, logvar = model(x)
        loss = loss_func(x, x_tilde, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        train_loss += batch_loss

        elapsed_time = round(time.time() - start_time, 2)

        if batch_idx > 0 and batch_idx % log_interval == 0:
            print('epoch {:3d} | {:4d}/{:4d} batches | {:.2f} seconds / batch | '
                  ' batch loss: {:.2f}'.format(
                epoch_idx, batch_idx, n_batches, elapsed_time, batch_loss
            ))
            start_time = time.time()

    train_loss /= float(n_batches)
    return train_loss


def evaluate(pred_data, eval_data, batch_size, device, beta=1.0):
    model.eval()
    data_size = pred_data.shape[0]
    n_batches = data_size // batch_size
    total_loss = 0.0
    n100_list = []
    r20_list = []
    r50_list = []
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, data_size)

            data = pred_data[start_idx: end_idx, :]
            heldout_data = eval_data[start_idx: end_idx, :]

            x = csr_matrix2tensor(data).to(device)
            x_tilde, mu, logvar = model(x)
            loss = loss_func(x, x_tilde, mu, logvar, beta)
            batch_loss = loss.item()
            total_loss += batch_loss

            # exclude training examples
            x_tilde = x_tilde.cpu().numpy()
            #x_tilde[data.nonzero()] = -np.inf

            n100 = metrics.NDCG_binary_at_k_batch(x_tilde, heldout_data, 100)
            r20 = metrics.Recall_at_k_batch(x_tilde, heldout_data, 20)
            r50 = metrics.Recall_at_k_batch(x_tilde, heldout_data, 50)

            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    total_loss /= float(n_batches)
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)


best_n100 = -np.inf

train_loss_list = []
valid_loss_list = []
valid_n100_list = []
valid_r20_list = []
valid_r50_list = []
for epoch_idx in range(args.epochs):
    # train
    train_loss = train(epoch_idx, args.batch_size, n_train_batches, device)
    writer.add_scalars('loss',  {'train': train_loss}, epoch_idx)
    train_loss_list.append(train_loss)

    # evalute
    valid_loss, n100, r20, r50 = evaluate(valid_pred_data, valid_eval_data, args.batch_size, device)

    writer.add_scalars('loss', {'valid': valid_loss}, epoch_idx)
    writer.add_scalars('n100', {'n100': n100}, epoch_idx)
    writer.add_scalars('r20', {'r20': r20}, epoch_idx)
    writer.add_scalars('r50', {'r50': r50}, epoch_idx)

    if n100 > best_n100:
        with open(args.save, 'wb') as fh:
            torch.save(model, fh)
        best_n100 = n100

    valid_loss_list.append(valid_loss)
    valid_n100_list.append(n100)
    valid_r20_list.append(r20)
    valid_r50_list.append(r50)

stats = pd.DataFrame(
    {'epoch_idx': list(range(args.epochs)),
     'train_loss': train_loss_list,
     'valid_loss': valid_loss_list,
     'n100': valid_n100_list,
     'r20': valid_r20_list,
     'r50': valid_r50_list
     },
    columns=['epoch_idx', 'train_loss', 'valid_loss', 'n100', 'r20', 'r50']
)
stats.to_csv('stats.csv', index=False, header=True)


# ==================== #
#     plot stats       #
# ==================== #
FONTSIZE = 20
fig = plt.figure(figsize=(18, 6.5))

gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1])

ax = plt.subplot(gs[0, 0])
ax.plot(stats['epoch_idx'], stats['train_loss'], label='train')
ax.plot(stats['epoch_idx'], stats['valid_loss'], label='valid')
ax.set_xlabel('Epoch', fontsize=FONTSIZE, labelpad=15)
ax.set_ylabel('Loss', fontsize=FONTSIZE, labelpad=15)
ax.legend(loc='best', fontsize=FONTSIZE)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONTSIZE)
    tick.label.set_rotation(0)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONTSIZE)
    tick.label.set_rotation(0)
ax.set_title('Epoch loss', fontsize=0.8*FONTSIZE)


ax = plt.subplot(gs[0, 1])
ax.plot(stats['epoch_idx'], stats['n100'], label='n100')
ax.plot(stats['epoch_idx'], stats['r20'], label='r20')
ax.plot(stats['epoch_idx'], stats['r50'], label='r50')

ax.set_xlabel('Epoch', fontsize=FONTSIZE, labelpad=15)
ax.set_ylabel('Ranking metric', fontsize=FONTSIZE, labelpad=15)
ax.set_ylim(0)
ax.legend(loc='best', fontsize=FONTSIZE)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(FONTSIZE)
    tick.label.set_rotation(0)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(FONTSIZE)
    tick.label.set_rotation(0)
ax.set_title('Ranking metrics', fontsize=0.8*FONTSIZE)

plt.tight_layout(pad=0, w_pad=2.0, h_pad=1.0)
fig.suptitle('VAE for CF', fontsize=FONTSIZE)
plt.subplots_adjust(top=0.85)
plt.savefig('stats.png', bbox_inches='tight')