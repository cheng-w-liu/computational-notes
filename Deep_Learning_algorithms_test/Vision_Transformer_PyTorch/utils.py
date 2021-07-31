import torch

# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py
from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)



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


class SparseCategoricalAccuracy(object):
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

