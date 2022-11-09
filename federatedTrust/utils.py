import ast
import json
import logging
import os
from json import JSONDecodeError

import torch
import yaml
from dotmap import DotMap
from torch.utils.data import DataLoader
from hashids import Hashids
from federatedTrust import calculation

hashids = Hashids()
logger = logging.getLogger(__name__)


def get_input_value(input_docs, inputs, operation):
    """Gets the input value from input document and apply the metric operation on the value
       :param input_docs: the input document map
       :param inputs: all the inputs
       :param operation: the metric operation
       :return: the metric value
    """
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
    """Gets the input value from input document by path
       :param input_docs: the input document map
       :param source_name: the key of the input document from the map
       :param path: the field name of the input value of interest
       :return: the input value from the input document
    """
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


def write_results_json(out_file, dic):
    """Writes the result to JSON
       :param out_file: the output file
       :param dic: the object to be written into JSON
    """
    with open(out_file, "a") as f:
        json.dump(dic, f)


def update_frequency(map, members, n, round):
    """Updates the client selection rate in the map
       If the current round is -1, means before trainint starts,
       then initialize the selection rate to 0 for every client.
       :param map: the client selection map {key: client id, value: selection rate}
       :param members: the selected members
       :param n: the total number of rounds
       :param round: the current round
    """
    hashed_members = [hashids.encode(id) for id in members]
    for id in hashed_members:
        if round == -1:
            map[id] = 0
        else:
            if id in map:
                map[id] += 1 / n


def count_class_samples(map, data):
    """Counts the number of samples by class
         :param map: the class distribution map {key: class label, value: sample size}
         :param data: the training data of a client
      """
    for batch, labels in data:
        for b, label in zip(batch, labels):
            l = hashids.encode(label.item())
            if l in map:
                map[l] += 1
            else:
                map[l] = 0


def read_eval_results_log(outdir, file):
    """Reads the evaluation results log from FederatedScope
         :param outdir: the output directory where the file is
         :param file: the file name
         :return the final evaluation result
      """
    result = None
    with open(os.path.join(outdir, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            json_dat = json.dumps(ast.literal_eval(line))
            dict_dat = json.loads(json_dat)
            if 'Server' in dict_dat.get('Role') and dict_dat.get('Round') == 'Final':
                result = DotMap(dict_dat['Results_raw'])
    return result


def read_system_metrics_log(outdir, file):
    """Reads the system metrics log from FederatedScope
         :param outdir: the output directory where the file is
         :param file: the file name
         :return the system metrics
      """
    result = None
    with open(os.path.join(outdir, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            json_dat = json.dumps(ast.literal_eval(line))
            dict_dat = json.loads(json_dat)
            if dict_dat.get('id') == 'sys_avg':
                result = DotMap(dict_dat)
    return result


def read_file(outdir, file):
    """Reads the content of a file from a directory
         :param outdir: the directory where the file is
         :param file: the file name
         :return the file content
      """
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


def get_aux_data(data, x_aux, y_aux, class_num, class_sample_size):
    """Generates an auxilary dataset with balanced classes
         :param data: training data of one client
         :param x_aux: the auxiliary training set
         :param y_aux: the auxiliary labels
         :param class_num: the number of classes
         :param class_sample_size: the desired class sample size
         :return tuple of (x_aux, y_aux)
      """
    x, y = data
    for i in range(class_num):
        for (sample, label) in zip(x, y):
            l = len(x_aux)
            if (label == i and (len(x_aux) + 1 <= class_sample_size * (i + 1))):
                x_aux.append(sample)
                y_aux.append(label)

    return (x_aux, y_aux)


def get_aux_dataloader(x_aux, y_aux, batch_size, num_workers, shuffle=False):
    """Gets the auxiliary data loader
         :param x_aux: the auxiliary training set
         :param y_aux: the auxiliary labels
         :param batch_size: the batch size
         :param num_workers: the number of workers
         :param shuffle: should shuffle the data
         :return tuple of (x_aux, y_aux)
      """
    x_aux = torch.stack(x_aux)
    y_aux = torch.stack(y_aux)
    return DataLoader((x_aux, y_aux),
                     batch_size,
                     shuffle=shuffle,
                     num_workers=num_workers)
