import numpy as np
import os


class CoraData:

    def __init__(self, data_root: str, val_frac: float, test_frac: float):
        """
        :param data_root: path to the root folder of the Cora dataset
        :param val_frac: fraction of nodes that will be used for validation
        :param test_frac: fraction of nodes that will be used for testing
        """
        self.data_root = data_root
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.train_frac = 1. - self.val_frac - self.test_frac

        self.node2id = None
        self.id2node = None
        self.class2id = None
        self.id2class = None
        self.features = None
        self.labels = None
        self.adjList = None

        self.train_nodes = None
        self.val_nodes = None
        self.test_nodes = None

        self.process_data()

    def process_data(self):
        self.process_nodes()
        self.process_edges()
        self.sample_nodes()

    def process_nodes(self):
        """
        Parse node information
        """
        file_name = 'cora.content'
        p_to_file = os.path.join(self.data_root, file_name)
        nodes_features_labels = np.genfromtxt(p_to_file, dtype=np.dtype(str))

        # node and the corresponding id
        nodes = np.array(nodes_features_labels[:, 0], dtype=np.int32)
        self.node2id = {node: i for i, node in enumerate(nodes)}
        self.id2node = {i: node for node, i in self.node2id.items()}

        # node label
        classes = set(nodes_features_labels[:, -1])
        self.class2id = {c: i for i, c in enumerate(classes)}
        self.id2class = {i: c for c, i in self.class2id.items()}
        self.labels = np.array(list(map(self.class2id.get, nodes_features_labels[:, -1])), dtype=np.int32)

        # node features
        self.features = nodes_features_labels[:, 1:-1].astype(np.float32)

    def process_edges(self):
        """
        Parse edges info and use adjacency-list representation to represent the graph
        """
        file_name = 'cora.cites'
        p_to_file = os.path.join(self.data_root, file_name)
        raw_edges = np.genfromtxt(p_to_file, dtype=np.int32)
        edges = np.array(list(map(self.node2id.get, raw_edges.flatten()))).reshape(raw_edges.shape)

        # use adjancency list to represent edges
        adjList = {}
        for edge in edges:
            dst, src = edge

            if src in adjList:
                adjList[src].append(dst)
            else:
                adjList[src] = [dst]

            if dst in adjList:
                adjList[dst].append(src)
            else:
                adjList[dst] = [src]

        self.adjList = adjList

    def sample_nodes(self):
        """
        Split the nodes into train, val, test sets
        """
        V = len(self.node2id)
        split = np.random.choice(
            ['train', 'val', 'test'],
            replace=True,
            size=(V,),
            p=(self.train_frac, self.val_frac, self.test_frac)
        )

        self.train_nodes = np.where(split == 'train')[0]
        self.val_nodes = np.where(split == 'val')[0]
        self.test_nodes = np.where(split == 'test')[0]


def test_cora_data():
    data_root = os.path.join(os.path.expanduser('~/ml_datasets/'), 'cora')
    cora_data = CoraData(data_root, 0.15, 0.15)
    cora_data.process_data()
    print(f'num of nodes: {len(cora_data.node2id)}, with the following splits:')
    print(f'training: {len(cora_data.train_nodes)}, validation: {len(cora_data.val_nodes)}, testing: {len(cora_data.test_nodes)} ')

    n_edges = 0
    for node, neighbors in cora_data.adjList.items():
        n_edges += len(neighbors)
    print(f'num of edges: {n_edges}')

    print('Check adjacency-list')
    batch_size = 7
    sampled_nodes = list(np.random.choice(cora_data.train_nodes, replace=False, size=(batch_size,)))
    for u in sampled_nodes:
        print(f'adj[{u}]={cora_data.adjList[u]}')


if __name__ == '__main__':
    test_cora_data()




