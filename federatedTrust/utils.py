import ast
import json
import logging
import os
from json import JSONDecodeError

import yaml
from dotmap import DotMap
from federatedTrust import calculation

logger = logging.getLogger(__name__)


def get_input_value(input_docs, inputs, operation):
    input_value = None
    args = []
    for i in inputs:
        source = i.get('source', '')
        field = i.get('field_path', '')
        input = get_value_from_path(input_docs, source, field)
        args.append(input)
    try:
        operationFn = getattr(calculation, operation)
        input_value = operationFn(*args)
    except TypeError as e:
        logger.warning(f"{operation} is not valid")

    return input_value


def get_value_from_path(input_docs, source_name, path):
    input_doc = input_docs[source_name]
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


def write_results(outdir, line):
    with open(os.path.join(outdir, 'federatedtrust_results.log'), "a") as out_file:
        out_file.write(line)


def update_frequency(map, members, n, round):
    for id in members:
        if round == -1:
            map[id] = 0
        else:
            if id in map:
                map[id] += 1 / n


def read_eval_results_log(outdir, file):
    result = None
    with open(os.path.join(outdir, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            json_dat = json.dumps(ast.literal_eval(line))
            dict_dat = json.loads(json_dat)
            if 'Server' in dict_dat.get('Role') and dict_dat.get('Round') == 'Final':
                result = DotMap(dict_dat['Results_raw'])
    return result


def read_file(outdir, file):
    result = None
    with open(os.path.join(outdir, file), 'r') as f:
        try:
            if file.lower().endswith('.yaml'):
                content = yaml.load(f, Loader=yaml.Loader)
            elif file.lower().endswith('.json'):
                content = json.load(f)
            else:
                content = f.read()
            result = DotMap(content)
        except JSONDecodeError as e:
            print(e)
    return result