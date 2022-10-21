import logging
import numbers

from federatedTrust import calculation
from federatedTrust.utils import get_input_value

logger = logging.getLogger(__name__)


class TrustPillar:
    def __init__(self, name, metrics, input_docs):
        self.name = name
        self.input_docs = input_docs
        self.metrics = metrics
        self.result = {self.name: 0, "components": {}}

    def evaluate(self):
        score = 0
        avg_weight = 1 / len(self.metrics)
        for key, value in self.metrics.items():
            score += avg_weight * self.get_group_score(key, value)
        self.result[self.name] = score
        return self.result

    def get_group_score(self, name, metrics):
        group_score = 0
        avg_weight = 1 / len(metrics)
        for key, value in metrics.items():
            group_score += avg_weight * self.get_metric_score(key, value)
        self.result['components'][name] = group_score
        return group_score

    def get_metric_score(self, name, metric):
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

            if score_type == 'factsheet_property_check':
                score = 0 if input_value is None else 1
        except KeyError:
            logger.warning(f"Null input for {name} metric")
        return score

