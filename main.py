import json
import logging
import os

from federatedTrust.pillar import TrustPillar
from utils.file_handler import read_training_results_log

dirname = os.path.dirname(__file__)
# logging.basicConfig(filename='exp/log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def assess():
    print(f'Assessing model \n--------------------')  # Press Ctrl+F8 to toggle the breakpoint.
    factsheet_path = os.path.join(dirname, 'federatedTrust/example/factsheet.json')
    metrics_cfg_path = os.path.join(dirname, 'federatedTrust/evaluation_metrics.json')
    # training_results = os.path.join(dirname, 'results/training_results/eval_results.log')
    # read_training_results_log(training_results)

    with open(factsheet_path, 'r') as f, \
         open(metrics_cfg_path, 'r') as m:
        factsheet = json.load(f)
        metrics_cfg = json.load(m)
        input_docs = {'factsheet': factsheet}
        for key, value in metrics_cfg.items():
            pillar = TrustPillar(key, value, input_docs)
            score = pillar.evaluate()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    assess()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
