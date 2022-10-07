import logging
import os.path

import numpy as np
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


def get_std(list):
    return np.std(list)




