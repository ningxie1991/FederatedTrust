import logging
import math
import numbers
import os.path
from math import e

import numpy as np
import shap
import torch.nn
from scipy.stats import entropy
from sklearn import preprocessing

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def get_normalized_score(score_key, score_map):
    score = 0;
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        keys = [key for key, value in score_map.items()]
        scores = np.array([value for key, value in score_map.items()])
        normalized_scores = preprocessing.normalize([scores])
        normalized_score_map = dict(zip(keys, normalized_scores[0]))
        score = normalized_score_map.get(score_key, np.nan)
    return score


def get_range_score(value, ranges, direction):
    score = 0
    if ranges is None:
        logger.warning("Score ranges are missing")
    else:
        total_bins = len(ranges) + 1
        bin = np.digitize(value, ranges, right=True)
        score = 1 - (bin / total_bins) if direction == 'desc' else bin / total_bins
    return score


def get_ranked_score(score_key, score_map, direction):
    score = 0
    if score_map is None:
        logger.warning("Score map is missing")
    else:
        sorted_score_map = dict(
            sorted(score_map.items(), key=lambda item: item[1], reverse=True if direction == 'desc' else False))
        for index, key in enumerate(sorted_score_map):
            if key == score_key:
                score = (index + 1) / len(sorted_score_map)
    return score


def get_true_score(value, direction):
    if value is True:
        return 1
    elif value is False:
        return 0
    else:
        if direction == 'desc':
            return 1 - value
        else:
            return value


def get_value(value):
    return value


def get_entropy(n, base=None):
    value, counts = np.unique([c for c in range(n)], return_counts=True)
    return entropy(counts, base=base)


def get_coefficient_variant(std, avg):
    return std / avg


def get_global_privacy_risk(dp, epsilon, n):
    if dp is True and isinstance(epsilon, numbers.Number):
        return 1 / (1 + (n-1) * math.pow(e, -epsilon))
    else:
        return 1


# supports PyTorch models
def get_feature_importance(dataloader, model, batch_size, device):
    shap_values = []
    if isinstance(model, torch.nn.Module):
        batch = next(iter(dataloader))
        batched_data, _ = batch

        n = batch_size
        m = math.floor(0.8 * n)

        background = batched_data[:m].to(device)
        test_data = batched_data[m:n].to(device)

        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(test_data)

        # plotting the graph
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_data.cpu().numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)

    return 0 if len(shap_values) == 0 else np.std(shap_values)


