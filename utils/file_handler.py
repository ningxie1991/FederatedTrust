import ast
import json
import os

dirname = os.path.dirname(__file__)


def read_training_results_log(log_path):
    result = {}
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            json_dat = json.dumps(ast.literal_eval(line))
            dict_dat = json.loads(json_dat)
            if 'Server' in dict_dat.get('Role') and dict_dat.get('Round') == 'Final':
                result = dict_dat


