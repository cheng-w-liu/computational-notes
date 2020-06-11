from abc import ABC
import numpy as np
import pandas as pd
import torch
from typing import Tuple, List

class Metric(ABC):

    def __init__(self):
        pass

    def __call__(self, outputs, targets):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class BinaryClassificationAverageAccuracy(Metric):

    def __init__(self):
        super(BinaryClassificationAverageAccuracy, self).__init__()
        self.hits = 0
        self.totals = 0

    def __call__(self, outputs, targets):
        preds = torch.max(outputs[0].data, dim=1, keepdim=True)
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


class PrecisionRecall(Metric):

    def __init__(self, thresholds: List[float], metric_type: str = 'distance'):
        assert metric_type in ['distance', 'logistic'], 'metric_type = distance or logistic'
        super(PrecisionRecall, self).__init__()
        self.thresholds = thresholds[:]
        self.true_positives = np.zeros_like(self.thresholds)
        self.false_positives = np.zeros_like(self.thresholds)
        self.false_negatives = np.zeros_like(self.thresholds)
        self.metric_type = metric_type

    def get_true_positive_counts(selfs, y_true: torch.tensor, y_pred: torch.tensor) -> int:
        tp = ((y_true == 1) & (y_pred == 1)).int().sum().item()
        return tp

    def get_false_positive_counts(self, y_true: torch.tensor, y_pred: torch.tensor) -> int:
        fp = ((y_true == 0) & (y_pred == 1)).int().sum().item()
        return fp

    def get_false_negative_counts(self, y_true: torch.tensor, y_pred: torch.tensor) -> int:
        fn = ((y_true == 1) & (y_pred == 0)).int().sum().item()
        return fn

    def compute_similarity(self, outputs):
        raise NotImplementedError

    def get_prediction(self, similarity: torch.tensor, threshold: float):
        raise NotImplementedError

    def __call__(self, outputs, targets):
        if self.metric_type == 'distance':
            assert len(outputs) == 2
        elif self.metric_type == 'logistic':
            assert len(outputs) == 1
        similarity = self.compute_similarity(outputs)
        y_true = targets.view_as(similarity)
        for i, threshold in enumerate(self.thresholds):
            y_pred = self.get_prediction(similarity, threshold)
            self.true_positives[i] = self.get_true_positive_counts(y_true, y_pred)
            self.false_positives[i] = self.get_false_positive_counts(y_true, y_pred)
            self.false_negatives[i] = self.get_false_negative_counts(y_true, y_pred)

    def reset(self):
        self.true_positives = np.zeros_like(self.thresholds)
        self.false_positives = np.zeros_like(self.thresholds)
        self.false_negatives = np.zeros_like(self.thresholds)

    def value(self) -> pd.DataFrame:
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        df = pd.DataFrame(
            data={
                'threshold': self.thresholds,
                'precision': precision,
                'recall': recall
            },
            columns=['threshold', 'precision', 'recall']
        )
        return df

    def name(self):
        return 'Precision and Recall'


class DistanceLossPrecisionRecall(PrecisionRecall):

    def compute_similarity(self, outputs: Tuple[torch.tensor, torch.tensor]):
        assert self.metric_type == 'distance' and len(outputs) == 2
        distances = (outputs[0].data - outputs[1].data).pow(2).sum(dim=1).sqrt()
        return distances

    def get_prediction(self, similarity: torch.tensor, threhsold: float):
        y_pred = (similarity < threhsold).int()
        return y_pred


class LogisticLossPrecisionRecall(PrecisionRecall):

    def compute_similarity(self, outputs: Tuple[torch.tensor]):
        assert self.metric_type == 'logistic' and len(outputs) == 1
        probs = outputs[0].data.exp()[:, 1]
        return probs

    def get_prediction(self, similarity: torch.tensor, threshold: float):
        y_pred = (similarity > threshold).int()
        return y_pred



