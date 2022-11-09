import logging

from federatedTrust import calculation
from federatedTrust.utils import get_input_value

logger = logging.getLogger(__name__)


class TrustPillar:
    def __init__(self, name, metrics, input_docs, use_weights=False):
        """TrustPillar class to represent a trust pillar
            :param name: name of the pillar
            :param metrics: metric definitions for the pillar
            :param input_docs: input documents
            :param use_weights: True to turn on the weights in the metric config file
        """
        self.name = name
        self.input_docs = input_docs
        self.metrics = metrics
        self.result = []
        self.use_weights = use_weights

    def evaluate(self):
        """Evaluate the trust score for the pillar
            :return score of [0, 1]
        """
        score = 0
        avg_weight = 1 / len(self.metrics)
        for key, value in self.metrics.items():
            weight = value.get('weight', avg_weight) if self.use_weights else avg_weight
            score += weight * self.get_notion_score(key, value.get('metrics'))
        score = round(score, 2)
        return score, {self.name: {"score": score, "notions": self.result}}

    def get_notion_score(self, name, metrics):
        """Evaluate the trust score for the notion
            :param name: name of the notion
            :param metrics: metrics definitions of the notion
            :return score of [0, 1]
        """
        notion_score = 0
        avg_weight = 1 / len(metrics)
        metrics_result = []
        for key, value in metrics.items():
            metric_score = self.get_metric_score(metrics_result, key, value)
            weight = value.get('weight', avg_weight) if self.use_weights else avg_weight
            notion_score += weight * float(metric_score)
        self.result.append({name: {"score": notion_score, "metrics": metrics_result}})
        return notion_score

    def get_metric_score(self, result, name, metric):
        """Evaluate the trust score for the metric
            :param result: the result object
            :param name: name of the metric
            :param metric: the metric definition
            :return score of [0, 1]
        """
        score = 0
        try:
            input_value = get_input_value(self.input_docs, metric.get('inputs'), metric.get('operation'))

            score_type = metric.get('type')
            if input_value is None:
                logger.warning(f"{name} input value is null")
            else:
                if score_type == 'true_score':
                    score = calculation.get_true_score(input_value, metric.get('direction'))
                elif score_type == 'score_mapping':
                    score = calculation.get_normalized_score(input_value, metric.get('score_map'))
                elif score_type == 'ranges':
                    score = calculation.get_range_score(input_value, metric.get('ranges'), metric.get('direction'))
                elif score_type == 'score_ranking':
                    score = calculation.get_ranked_score(input_value, metric.get('score_map'), metric.get('direction'))

            if score_type == 'property_check':
                score = 0 if input_value is None else input_value
        except KeyError:
            logger.warning(f"Null input for {name} metric")
        score = round(score, 2)
        result.append({name: {"score": score}})
        return score

