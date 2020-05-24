from abc import ABC
import torch

class Metric(ABC):

    def __init__(self):
        pass

    def __call__(self, outputs, targets, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class BinaryClassificationAverageAccuracy(Metric):

    def __init__(self):
        self.hits = 0
        self.totals = 0

    def __call__(self, outputs: torch.tensor, targets: torch.tensor, loss: float):
        preds = torch.max(outputs, dim=1, keepdim=True)
        pred_classes = preds.indices
        self.hits += pred_classes.eq(targets.data.view_as(pred_classes)).cpu().sum().item()
        self.totals += targets.size(0)
        return self.value()

    def reset(self):
        self.hits = 0
        self.totals = 0

    def value(self):
        return 100 * float(self.hits) / float(self.totals)

    def name(self):
        return 'Accuracy'
