import torch

class Mean(object):
    """
    Vanilla version of tf.keras.metrics.Mean
    """

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


class SparseCategoricalAccuracy:
    """
    Vanilla version of tf.keras.metrics.SparseCategoricalAccuracy
    """

    def __init__(self):
        self.hits = 0
        self.total = 0

    def __call__(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        self.hits += (preds == targets).sum().item()
        self.total += logits.size(0)

    def result(self):
        return float(self.hits) / float(self.total)

    def reset(self):
        self.hits = 0
        self.total = 0


def test_mean():
    m = Mean()
    m(1.)
    m(2.)
    m(3.)
    print(f'm.result: {m.result()}')
    assert m.result() == 2.0


def test_sparse_categorical_accuracy():
    logits = torch.tensor(
        [[1.3, -1.2, 1.29, -0.98],
         [0.01, -1.98, -0.45, 0.33],
         [1.0, 2.0, 3.0, 4.0],
         [-0.43, 1.0, 2.0, -2.4]]

    )
    actuals = torch.tensor([0, 0, 3, 1]).long()
    acc = SparseCategoricalAccuracy()
    acc(logits, actuals)
    print(f'accuracy: {acc.result()}')
    assert acc.result() == 0.5


if __name__ == '__main__':
    test_mean()
    test_sparse_categorical_accuracy()