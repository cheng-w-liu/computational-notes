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

    def reset_state(self):
        self.value = 0.
        self.n = 0

    def result(self):
        return self.value / float(self.n)


class SparseCategoricalAccuracy:
    """
    Vanilla version of tf.keras.metrics.SparseCategoricalAccuracy
    """

    def __init__(self):
        self.hits = 0
        self.total = 0

    def __call__(self, logits, targets):
        self.hits += (torch.argmax(logits, dim=1) == targets).sum().item()
        self.total += logits.size(0)

    def reset_state(self):
        self.hits = 0
        self.total = 0

    def result(self):
        return float(self.hits) / float(self.total)
