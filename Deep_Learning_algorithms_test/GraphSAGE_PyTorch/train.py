import numpy as np
import torch

from graph_sampling import generate_batch_nodes
from metrics import Mean, SparseCategoricalAccuracy


def run_one_epoch(model, optimizer, loss_func, nodes, full_labels, full_adjList, embeddings,
                  batch_size, num_sampled_neighbors, K, training=True):
    loss_metric = Mean()
    acc_metric = SparseCategoricalAccuracy()

    num_nodes = len(nodes)
    for batch_idx in range(num_nodes // batch_size):
        sampled_nodes = np.random.choice(nodes, replace=False, size=(batch_size,))
        sampled_nodes_targets = torch.from_numpy(full_labels[sampled_nodes]).long()

        batch_nodes, batch_adjList = generate_batch_nodes(sampled_nodes, full_adjList, num_sampled_neighbors, K)
        inputs = (batch_nodes, batch_adjList, embeddings)

        if training:
            optimizer.zero_grad()
            logits = model(inputs)
            batch_loss = loss_func(logits, sampled_nodes_targets)
            batch_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(inputs)
                batch_loss = loss_func(logits, sampled_nodes_targets)

        loss_metric(batch_loss.item())
        acc_metric(logits, sampled_nodes_targets)

    return loss_metric.result(), acc_metric.result()

