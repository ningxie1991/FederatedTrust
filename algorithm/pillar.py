import logging
import os

import numpy as np
from sklearn import preprocessing

dirname = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


class TrustPillar:
    def __init__(self, name, metrics, input_docs):
        self.name = name
        self.input_docs = input_docs
        self.metrics = metrics

    def evaluate(self):
        print(f"Assessing {self.name} pillar")
        score = 0
        avg_weight = 1 / len(self.metrics)
        for key, value in self.metrics.items():
            score += avg_weight * self.get_group_score(key, value)
        print(f"Pillar score: {score} \n--------------------")
        return score

    def get_group_score(self, name, metrics):
        group_score = 0
        avg_weight = 1 / len(metrics)
        for key, value in metrics.items():
            group_score += avg_weight * self.get_metric_score(key, value)
        print(f"{name} score: {group_score}")
        return group_score

    def get_metric_score(self, name, metric):
        score = 0
        try:
            input = metric.get('input')
            source_name = input.get('source', '')
            field_path = input.get('field_path', '')
            input_value = self.get_value_from_path(source_name, field_path)

            score_type = metric.get('type')
            if input_value is None:
                logger.warning(f"{field_path} in {source_name} is null")
            else:
                if score_type == 'true_score':
                    score = input_value
                elif score_type == 'score_mapping':
                    score = self.get_normalized_score(input_value, metric.get('score_map'))
                elif score_type == 'ranges':
                    score = self.get_range_score(input_value, metric.get('ranges'), metric.get('direction'))
                elif score_type == 'score_ranking':
                    score = self.get_ranked_score(input_value, metric.get('score_map'), metric.get('direction'))

            if score_type == 'factsheet_property_check':
                score = 0 if input_value is None else 1
        except KeyError:
            logger.warning(f"Null input for {name} metric")
        return score

    def get_value_from_path(self, source_name, path):
        input_doc = self.input_docs.get(source_name)
        if input_doc is None:
            logger.warning(f"{source_name} is null")
            return None
        else:
            d = input_doc
            for nested_key in path.split('/'):
                temp = d.get(nested_key)
                if isinstance(temp, dict):
                    d = d.get(nested_key)
                else:
                    return temp
        return None

    @staticmethod
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

    @staticmethod
    def get_range_score(value, ranges, direction):
        score = 0
        if ranges is None:
            logger.warning("Score ranges are missing")
        else:
            total_bins = len(ranges) + 1
            bin = np.digitize(value, ranges, right=True)
            score = 1 - (bin / total_bins) if direction == 'desc' else bin / total_bins
        return score

    @staticmethod
    def get_ranked_score(score_key, score_map, direction):
        score = 0
        if score_map is None:
            logger.warning("Score map is missing")
        else:
            sorted_score_map = dict(sorted(score_map.items(), key=lambda item: item[1], reverse=True if direction == 'desc' else False))
            for index, key in enumerate(sorted_score_map):
                if key == score_key:
                    score = (index + 1) / len(sorted_score_map)
        return score
