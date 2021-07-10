import numpy as np
import torch
from typing import Dict, List, Tuple


def get_unique_nodes(adjList, has_weight=False) -> List[int]:
    """
    :param adjList: Dict[int, List[int]] or Dict[int, List[Tuple[int, float]]] : adjacency-list representation of a (weighted) graph
    :param has_weight: bool: indicating if adjList carries non-trivial weights
    :return: List[int]: unique nodes extracted from the graph
    """
    nodes = set()
    for node, neighbors in adjList.items():
        if has_weight:
            nodes = nodes.union(set([node] + [idx for idx, _ in neighbors]))
        else:
            nodes = nodes.union(set([node] + neighbors))
    return list(nodes)


def random_sampling(adjList: Dict[int, List[int]], target_nodes: List[int], num_layers: int, num_neighbors: int):
    """
    :param adjList:
    :param target_nodes:
    :param num_layers:
    :param num_neighbors:
    :return:

    batch_nodes[k] = B^k
        B^{k} contains the nodes that are needed in order to generate embeddings of nodes in B^{k+1}
        B^K = target_nodes

    batch_adjList[k] = G^k
        G^k = graph associated with B^k
        G^K is a dummy entry

    B^k contains the nodes needed in order to generate embeddings of nodes in B^{k+1}
    G^k is the graph needed in order to generate embeddings of nodes in B^{k+1}

    """
    K = num_layers

    batch_nodes = [[] for _ in range(K+1)]
    batch_nodes[K] = target_nodes[:]

    batch_adjList = [{} for _ in range(K+1)]

    for k in range(K-1, -1, -1): # k = K-1, ..., 0
        for u in batch_nodes[k+1]:
            # batch_adjList[k][u] = list(np.random.choice(adjList[u], replace=True, size=(num_neighbors,)))
            batch_adjList[k][u] = [ (v, 1.0) for v in np.random.choice(adjList[u], replace=True, size=(num_neighbors,)) ]
        batch_nodes[k] = get_unique_nodes(batch_adjList[k], has_weight=True)

    return batch_nodes, batch_adjList


def topK_sampling_one_batch(adjList: Dict[int, List[int]], V: int, target_nodes: List[int], topK: int, n_iters: int, n_hops: int) -> Dict[int, List[Tuple[int, float]]]:
    n_nodes = len(target_nodes)

    traces = torch.zeros(n_nodes, n_iters, n_hops + 1)

    # Perform random walk
    for i, node in enumerate(target_nodes):
        for j in range(n_iters):
            u = node
            for k in range(1, n_hops + 1):
                u = np.random.choice(adjList[u])
                traces[i, j, k] = u

    # Collect counts of visits from the random walks
    visit_counts = traces[:, :, 1:].contiguous().view(n_nodes, -1).type(torch.int64)

    # Convert counts to probabilities
    visit_probs = torch.zeros(n_nodes, V)
    visit_probs.scatter_add_(
        dim=1,
        index=visit_counts,
        src=torch.ones_like(visit_counts, dtype=visit_probs.dtype)
    )

    visit_probs = visit_probs / visit_probs.sum(dim=1, keepdim=True)

    # For each of the target nodes, extract top-K visited neighbors along with the probabilities
    topK_results = visit_probs.topk(dim=1, k=topK)
    topK_neighbors = topK_results.indices
    topK_neighbor_weights = topK_results.values

    topK_adjList = {}
    for i, node in enumerate(target_nodes):
        topK_adjList[node] = [(idx.item(), prob.float().item()) for idx, prob in
                              zip(topK_neighbors[i], topK_neighbor_weights[i]) if idx != node and prob > 0.]

    return topK_adjList


def topK_sampling(adjList: Dict[int, List[int]], V: int, target_nodes: List[int], num_layers: int, topK, n_iters: int=10, n_hops: int=5):
    """
    :param adjList: adjacency-list representation of a graph
    :param V: num of nodes
    :param target_nodes:
    :param num_layers: num of Graph Neural Network layers to iterate over
    :param topK: topK neighbor nodes for each of the target nodes
    :param n_iters: num. of iterations of random walk
    :param n_hops: num of hops for each iteration of random walk
    :return:


    batch_nodes[k] = B^k
        B^{k} contains the nodes that are needed in order to generate embeddings of nodes in B^{k+1}
        B^K = target_nodes

    batch_adjList[k] = G^k
        G^k = graph associated with B^k
        G^K is a dummy entry

    B^k contains the nodes needed in order to generate embeddings of nodes in B^{k+1}
    G^k is the graph needed in order to generate embeddings of nodes in B^{k+1}

    """
    K = num_layers
    batch_nodes = [[] for _ in range(K+1)]
    batch_nodes[K] = target_nodes[:]

    batch_adjList = [{} for _ in range(K+1)]

    for k in range(K-1, -1, -1):
        batch_adjList[k] = topK_sampling_one_batch(adjList, V, batch_nodes[k+1], topK, n_iters, n_hops)
        batch_nodes[k] = get_unique_nodes(batch_adjList[k], has_weight=True)

    return batch_nodes, batch_adjList



def test_random_sampling():
    print('========================')
    print('  Test random sampling  ')
    print('========================')

    adjList = {
        0: [1, 2],
        1: [11, 0, 3],
        2: [0, 3, 7],
        3: [1, 2, 4, 9, 10],
        4: [3, 11],
        5: [9],
        6: [7, 10],
        7: [6, 10, 2],
        8: [9],
        9: [5, 3, 8],
        10: [3, 6, 7],
        11: [1, 4]
    }
    np.random.seed(42)

    nodes = get_unique_nodes(adjList)
    print(f'unique nodes: {nodes}')

    num_layers = 2
    num_neighbors = 3

    target_nodes = list(np.random.choice(nodes, replace=False, size=(2,)))
    batch_nodes, batch_adjList = random_sampling(adjList, target_nodes, num_layers, num_neighbors)

    print('batch_nodes:')
    print(batch_nodes)
    print('--')
    print('batch_adjList:')
    print(batch_adjList)
    print('====================================================\n')


def test_topK_sampling():
    print('========================')
    print('   Test topK sampling   ')
    print('========================')

    adjList = {
        0: [1, 2],
        1: [11, 0, 3],
        2: [0, 3, 7],
        3: [1, 2, 4, 9, 10],
        4: [3, 11],
        5: [9],
        6: [7, 10],
        7: [6, 10, 2],
        8: [9],
        9: [5, 3, 8],
        10: [3, 6, 7],
        11: [1, 4]
    }
    np.random.seed(42)

    nodes = get_unique_nodes(adjList, has_weight=False)
    print(f'unique nodes: {nodes}')

    num_layers = 2
    topK = 3

    target_nodes = list(np.random.choice(nodes, replace=False, size=(2,)))
    batch_nodes, batch_adjList = topK_sampling(adjList, len(nodes), target_nodes, num_layers, topK)

    print('batch_nodes:')
    print(batch_nodes)
    print('--')
    print('batch_adjList:')
    print(batch_adjList)
    print('====================================================\n')


if __name__ == '__main__':
    test_random_sampling()
    test_topK_sampling()

