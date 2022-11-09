import logging
import math
import numbers
import os.path
from math import e

import numpy as np
import shap
import torch.nn
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u
from scipy.stats import variation
from sklearn import preprocessing
from torch import nn, optim

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)

R_L1 = 40
R_L2 = 2
R_LI = 0.1


def get_normalized_score(score_key, score_map):
    """Finds the score by the score_key in the score_map
        :param score_key: the key to look up in the score_map
        :param score_map: the defined score map
        :return: normalized score of [0, 1]
    """
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
    """Maps the value to a range and gets the score by the range and direction
        :param value: the input score
        :param ranges: the ranges defined
        :param direction: asc means the higher the range the higher the score, desc means otherwise
        :return: normalized score of [0, 1]
    """
    score = 0
    if ranges is None:
        logger.warning("Score ranges are missing")
    else:
        total_bins = len(ranges) + 1
        bin = np.digitize(value, ranges, right=True)
        score = 1 - (bin / total_bins) if direction == 'desc' else bin / total_bins
    return score


def get_ranked_score(score_key, score_map, direction):
    """Finds the score by the score_key in the score_map and returns the rank of the score
        :param score_key: the key to look up in the score_map
        :param score_map: the score map defined
        :param direction: asc means the higher the range the higher the score, desc means otherwise
        :return: normalized score of [0, 1]
    """
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
    """Returns the negative of the value if direction is 'desc', otherwise returns value
        :param value: the input value
        :param direction: asc means the higher the range the higher the score, desc means otherwise
        :return: object
    """
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
    """Returns the value
        :param value: the input value
        :return: the value object
    """
    return value


def check_properties(*args):
    """Check if all the arguments have values
        :param args: all the arguments
        :return: the mean of the binary array
    """
    result = map(lambda x: x is not None and x != "", args)
    return np.mean(list(result))


def get_entropy(n):
    """Calculates entropy
        :param n: number of data samples
        :return: entropy of the dataset
    """
    entropy = -1 * np.sum(np.log2(1/n)*(1/n))
    return entropy


def get_cv(std, avg):
    """Calculates the coefficient of variation
       :param std: the standard deviation
       :param avg: the mean
       :return: entropy of the dataset
   """
    return std / avg


def get_global_privacy_risk(dp, epsilon, n):
    """Calculates the global privacy risk by epsilon and the number of clients
       :param dp: True or False
       :param epsilon: the epsilon value
       :param n: number of clients
       :return: the global privacy risk
    """
    if dp is True and isinstance(epsilon, numbers.Number):
        return 1 / (1 + (n - 1) * math.pow(e, -epsilon))
    else:
        return 1


def get_feature_importance_cv(test_sample, model, cfg):
    """Calculates feature importance coefficient of variation
       :param test_sample: one test sample
       :param model: the model
       :param cfg: configs
       :return: the coefficient of variation of the feature importance scores, [0, 1]
    """
    cv = 0
    batch_size = cfg['batch_size']
    device = cfg['device']
    if isinstance(model, torch.nn.Module):
        batched_data, _ = test_sample

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


def get_clever_score(test_sample, model, cfg):
    """Calculates the CLEVER score
       :param test_sample: one test sample
       :param model: the model
       :param cfg: configs
       :return: the CLEVER score of type number
    """
    nb_classes = cfg['nb_classes']
    lr = cfg['lr']
    images, _ = test_sample
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
    score_untargeted = clever_u(classifier, background.numpy(), 10, 5, R_L2, norm=2, pool_factor=3, verbose=False)
    return score_untargeted