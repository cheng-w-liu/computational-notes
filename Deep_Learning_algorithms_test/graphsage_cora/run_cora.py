"""
doc

Usage:
    run_cora.py [options]

Options:
    --p-to-content=<file>               path to the cora.content file [default: ml_datasets/cora/cora.content]
    --p-to-citations=<fil>              path to the cora.cites file [default: ml_datasets/cora/cora.cites]
    --uniform-init=<float>              uniformly initialize all parameters [default: 0.1]
    --lr=<float>                        learning rate [default: 0.001]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
"""

from collections import defaultdict
from docopt import docopt
import numpy as np
import os
import random
from sklearn.metrics import f1_score
import time
import torch
import torch.nn as nn

from aggregator import Aggregator
from encoder import Encoder
from graph_sage import GraphSAGEclassifier


def get_cora_data(p_to_content: str, p_to_citations: str):
    num_nodes = 2708
    num_feat = 1433
    feat_data = np.zeros((num_nodes, num_feat), dtype=np.float32)
    labels = np.empty((num_nodes,))

    node2id = {}
    label2id = {}

    with open(p_to_content, 'r') as fh:
        for i, line in enumerate(fh):
            cols = line.strip().split()
            feat_data[i, :] = np.array([float(x) for x in cols[1:-1]])
            node2id[cols[0]] = i
            if cols[-1] not in label2id:
                label2id[cols[-1]] = len(label2id)
            labels[i] = label2id[cols[-1]]

    adj_list = defaultdict(set)
    with open(p_to_citations) as fh:
        for i, line in enumerate(fh):
            cols = line.strip().split()
            id1 = node2id[cols[0]]
            id2 = node2id[cols[1]]
            adj_list[id1].add(id2)
            adj_list[id2].add(id1)

    return feat_data, labels, adj_list, node2id, label2id

def main():
    args = docopt(__doc__)

    np.random.seed(2)
    random.seed(3)

    # load data
    p_to_content = os.path.join(os.path.expanduser('~'), args['--p-to-content'])
    p_to_citations = os.path.join(os.path.expanduser('~'), args['--p-to-citations'])
    feat_data, labels, adj_list, node2id, label2id = get_cora_data(p_to_content, p_to_citations)

    num_nodes, feat_dim = feat_data.shape
    num_classes = len(label2id)
    print(f'num of nodes: {num_nodes}')
    print(f'feature dim: {feat_dim}')
    print(f'num of classes: {num_classes}')

    embed_dim_1 = 128
    embed_dim_2 = 64

    # create GraphSAGE modules
    embeddings = nn.Embedding(num_nodes, feat_dim) #, padding_idx=num_nodes)
    embeddings.weight = nn.Parameter(
        torch.tensor(feat_data),
        requires_grad=False
    )

    agg1 = Aggregator(
        feature_map=embeddings,
        feat_dim=feat_dim,
        agg_type='mean',
        gpu=torch.cuda.is_available()
    )

    enc1 = Encoder(
        feature_map=embeddings,
        feat_dim=feat_dim,
        aggregator=agg1,
        adj_list=adj_list,
        embed_dim=embed_dim_1,
        base_model=None,
        num_sample=5,
        gpu=torch.cuda.is_available()
    )

    # tmp_graphsage = GraphSAGEclassifier(num_classes=num_classes,encoder=enc1)
    # print('parameters in k=1 GraphSAGE:')
    # for name, param in tmp_graphsage.named_parameters():
    #     if param.requires_grad:
    #         print(f'name: {name}, shape:{param.shape}')
    #         #print(param)
    # print('------------')

    agg2 = Aggregator(
        feature_map=lambda nodes: enc1(nodes),
        feat_dim=embed_dim_1,
        agg_type='mean',
        gpu=torch.cuda.is_available()
    )

    enc2 = Encoder(
        feature_map=lambda nodes: enc1(nodes),
        feat_dim=enc1.embed_dim,
        aggregator=agg2,
        adj_list=adj_list,
        embed_dim=embed_dim_2,
        base_model=enc1,
        num_sample=5,
        gpu=torch.cuda.is_available()
    )

    model = GraphSAGEclassifier(
        num_classes=num_classes,
        encoder=enc2
    )

    if torch.cuda.is_available():
        model = model.cuda()

    print('parameters in k=2 GraphSAGE:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'name: {name}, shape:{param.shape}')
    print('-'*30)

    print('all parameters in k=2 GraphSAGE:')
    for name, param in model.named_parameters():
        print(f'name: {name}, shape:{param.shape}')
    print('-'*30)

    rand_indices = np.random.permutation(num_nodes)
    test_size = 1000
    val_size = 500
    train_size = num_nodes - test_size - val_size
    batch_size = 256
    test = list(rand_indices[:test_size])
    val = list(rand_indices[test_size: test_size + val_size])
    train = list(rand_indices[test_size + val_size:])
    print(f'train size: {train_size}, test size: {test_size}, val size: {val_size}')
    print('-'*30)

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.0:
        print(f'uniformly initialize parameters: [-{uniform_init}, +{uniform_init}]')
        for param in model.parameters():
            if param.requires_grad:
                param.data.uniform_(-uniform_init, uniform_init)
    print('-'*30)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    clip_grad = float(args['--clip-grad'])

    for batch_idx in range(100):
        batch_nodes = train[:batch_size]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        batch_loss = model.loss(
            nodes=batch_nodes,
            labels=torch.tensor(labels[np.array(batch_nodes)], dtype=torch.long)
        )
        batch_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        end_time = time.time()
        time_elapsed = end_time - start_time
        #print(time_elapsed)
        #print(batch_idx, batch_loss.item())
        print(f'batch: {batch_idx}, training time: {time_elapsed:.6f}, batch_losss: {batch_loss.item():.5f}')
    print('-'*30)
    val_output = model.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))

if __name__ == '__main__':
    main()



