import json
import os

from hashids import Hashids
from federatedTrust.utils import set_file
from json import JSONDecodeError

hashids = Hashids()


def populate_factsheet(server_cfg, client_cfgs):
    # to do
    pass


def register_selection(clients, total_round_num, out_dir):
    results_file = os.path.join(out_dir, "client_selection.json")
    set_file(results_file)
    with open(results_file, 'r+') as f:
        results = {}
        try:
            results = json.load(f)
            for client in clients:
                key = hashids.encode(client)
                if key in results:
                    results[key] += 1 / total_round_num
                else:
                    results[key] = 0
        except JSONDecodeError as e:
            print(e)
        f.seek(0)
        f.truncate()
        json.dump(results, f)
        f.close()