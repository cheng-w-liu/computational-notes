import numpy as np
from typing import List, Dict


def get_unique_nodes(adjList: Dict[int, List[int]]) -> List[int]:
    """
    :param adjList: Dict[int, List[int]], adjacency-list representation of a graph
    :return:
      extracted the unique nodes the adjList, return as a list of nodes
    """
    nodes = set()
    for node, neighs in adjList.items():
        nodes = nodes.union(set([node] + neighs))
    nodes = list(nodes)
    return nodes


def generate_batch_nodes(target_nodes: List[int], full_adjList: Dict[int, List[int]], num_neigh_samples=5, K=2):
    """
    :param target_nodes: the nodes we like to generate representation for
    :param full_adjList: adjacency-list representation of the original, full graph
    :param num_neigh_samples:
    :param K:

    :return: a tuple of (batch_nodes, batch_adjList)
        batch_nodes: List[List[int]], outer list has length K.
                     Each list contains the nodes we want to generate representation for
        batch_adjList: List[Dict[int, List[int]]], length K.
                     Each entry is an adjacency-list representation of a sampled graph.

    Notes:
    batch_adjList[k]: the sampled graph needed in order to generate representations for the nodes
                      specified in batch_nodes[k+1]

    batch_nodes[0] corresponds to `B^0`, the initial nodes needed in order to generate
                     representation for the nodes in B^K

    adjList[K] is a dummy entry
    """

    batch_nodes = [[] for _ in range(K+1)]
    batch_nodes[K] = target_nodes[:]

    batch_adjList = [{} for _ in range(K+1)]

    for k in range(K-1, -1, -1):  # k = K-1, ..., 0
        batch_adjList[k] = {}
        for u in batch_nodes[k+1]:
            if num_neigh_samples is not None:
                batch_adjList[k][u] = list(np.random.choice(full_adjList[u], replace=True, size=(num_neigh_samples,)))
            else:
                batch_adjList[k][u] = full_adjList[u][:]
        batch_nodes[k] = get_unique_nodes(batch_adjList[k])

    return batch_nodes, batch_adjList


def test_graph_sampling():
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

    nodes = get_unique_nodes(adjList)

    sampled_nodes = np.random.choice(nodes, replace=False, size=(3,))

    num_neigh_samples = 3
    K = 3
    batch_nodes, batch_adjList = generate_batch_nodes(sampled_nodes, adjList, num_neigh_samples, K)

    print(f'target nodes: {sampled_nodes}')
    print(f'starting nodes: {batch_nodes[0]}')
    for k in range(K): # k = 0, ..., K-1
        print(batch_adjList[k])


if __name__ == '__main__':
    test_graph_sampling()
