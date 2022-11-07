import logging
import math
import numbers
import os.path
from math import e

import numpy as np
import shap
import torch.nn
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u, clever_t
from scipy.stats import entropy, variation
from sklearn import preprocessing
from torch import nn, optim

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

R_L1 = 40
R_L2 = 2
R_LI = 0.1


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
        sorted_scores = sorted(score_map.items(),
                               key=lambda item: item[1],
                               reverse=direction == 'desc')
        sorted_score_map = dict(sorted_scores)
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


def check_properties(*args):
    result = map(lambda x: x is not None and x != "", args)
    return np.mean(list(result))


def get_entropy(n, base=None):
    value, counts = np.unique([c for c in range(n)], return_counts=True)
    return entropy(counts, base=base)


def get_cv(std, avg):
    return std / avg


def get_global_privacy_risk(dp, epsilon, n):
    if dp is True and isinstance(epsilon, numbers.Number):
        return 1 / (1 + (n - 1) * math.pow(e, -epsilon))
    else:
        return 1


def get_feature_importance_cv(dataloader, model, batch_size, device):
    cv = 0
    if isinstance(model, torch.nn.Module):
        batch = next(iter(dataloader))
        batched_data, _ = batch

        n = batch_size
        m = math.floor(0.8 * n)

        background = batched_data[:m].to(device)
        test_data = batched_data[m:n].to(device)

        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(test_data)
        if shap_values is not None and len(shap_values) > 0:
            sums = np.array([shap_values[i].sum() for i in range(len(shap_values))])
            abs_sums = np.absolute(sums)
            cv = variation(abs_sums)
    return cv


def get_clever_score(dataloader, model, nb_classes, lr=0.01):
    batch = next(iter(dataloader))
    images, _ = batch
    background = images[-1]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 255.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=nb_classes,
    )
    # score_targeted = clever_t(classifier, background.numpy(), 2, 10, 5, R_L2, norm=2, pool_factor=3)
    score_untargeted = clever_u(classifier, background.numpy(), 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
    return score_untargeted