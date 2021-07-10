import numpy as np
import torch

from graph_sampling import random_sampling, topK_sampling
from metrics import Mean, SparseCategoricalAccuracy

def run_one_epoch(model, optimizer, loss_func,
                  V, selected_nodes, full_adjList, full_labels, embeddings,
                  k_layers=2, num_neighbors=5, topK=5, batch_size=32,
                  sampling_method='topK', n_iters=20, n_hops=7, training=True
                  ):
    loss_metric = Mean()
    acc_metric = SparseCategoricalAccuracy()

    num_nodes = len(selected_nodes)
    for batch_idx in range(num_nodes // batch_size):
        target_nodes = np.random.choice(selected_nodes, replace=False, size=(batch_size,))
        target_nodes_labels = torch.from_numpy(full_labels[target_nodes]).long()

        if sampling_method == 'topK':
            assert topK > 0 and n_iters > 0 and n_hops > 0
            batch_nodes, batch_adjList = topK_sampling(full_adjList, V, target_nodes, k_layers, topK, n_iters, n_hops)
        elif sampling_method == 'random':
            batch_nodes, batch_adjList = random_sampling(full_adjList, target_nodes, k_layers, num_neighbors)
        else:
            raise ValueError('sampling_method = topK or random')

        if training:
            optimizer.zero_grad()
            logits = model(batch_nodes, batch_adjList, embeddings)
            batch_loss = loss_func(logits, target_nodes_labels)
            batch_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch_nodes, batch_adjList, embeddings)
                batch_loss = loss_func(logits, target_nodes_labels)

        loss_metric(batch_loss.item())
        acc_metric(logits, target_nodes_labels)

    return loss_metric.result(), acc_metric.result()
