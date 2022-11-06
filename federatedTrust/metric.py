import json
import logging
import os
import shutil
import sys
from json import JSONDecodeError

import numpy as np
from hashids import Hashids

from federatedTrust.pillar import TrustPillar
from federatedTrust.utils import read_file, read_eval_results_log, update_frequency, write_results

dirname = os.path.dirname(__file__)
hashids = Hashids()
logger = logging.getLogger(__name__)


class TrustMetricManager:
    def __init__(self, outdir, ):
        self.outdir = outdir
        self.factsheet_file_nm = "factsheet.json"
        self.factsheet_template_file_nm = "factsheet_template.json"
        self.client_selection_file_nm = "client_selection.json"
        self.eval_results_file_nm = "eval_results.log"
        self.eval_metrics_file_nm = "eval_metrics_v1.json"
        self.model_map_file_nm = "model_map.json"
        self.log_nm = "federatedtrust_print.log"
        self.out_file_nm = "federatedtrust_results.log"
        self.register_logger()

    def register_logger(self):
        root_logger = logger
        root_logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(self.outdir, self.log_nm))
        fh.setLevel(logging.DEBUG)
        logger_formatter = logging.Formatter(
            "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        fh.setFormatter(logger_formatter)
        root_logger.addHandler(fh)

    def populate_factsheet(self, mode=None, cfg_file=None, trainer_context=None, eval_results_file=None,
                           client_selection_file=None):
        factsheet_file = os.path.join(self.outdir, self.factsheet_file_nm)
        factsheet_template = os.path.join(dirname, f"configs/{self.factsheet_template_file_nm}")
        model_map_file = os.path.join(dirname, f"configs/{self.model_map_file_nm}")

        # for development purpose
        if mode == "development":
            cfg_file = os.path.join(dirname, 'example/fs_config.yaml')
            eval_results_file = os.path.join(dirname, f"example/{self.eval_results_file_nm}")
            client_selection_file = os.path.join(dirname, f"example/{self.client_selection_file_nm}")
            model_context = {'trainable_para_names': ['a' for x in range(12)]}

        cfg = None if cfg_file is None else read_file(self.outdir, cfg_file)
        eval_results = None if eval_results_file is None else read_eval_results_log(self.outdir, eval_results_file)
        client_selection = None if client_selection_file is None else read_file(self.outdir, client_selection_file)

        if not os.path.exists(factsheet_file):
            shutil.copyfile(factsheet_template, factsheet_file)

        if cfg is not None and not os.path.exists(model_map_file):
            logger.error(f"{model_map_file} is missing! Please check documentation.")
            return

        with open(factsheet_file, 'r+') as f, open(model_map_file, 'r') as m:
            factsheet = {}
            try:
                factsheet = json.load(f)
                model_map = json.load(m)

                if cfg is not None:
                    logger.info("FactSheet: Populating configs")
                    # set project specifications
                    factsheet['project']['overview'] = cfg.expname
                    # set participants
                    factsheet['participants']['client_num'] = cfg.federate.client_num or 0
                    factsheet['participants']['sample_client_rate'] = cfg.federate.sample_client_rate or 0
                    factsheet['participants']['client_selector'] = cfg.federate.join_in_info or "random"
                    # set configuration
                    factsheet['configuration']['optimization_algorithm'] = cfg.federate.method or ""
                    factsheet['configuration']['training_model'] = model_map[cfg.model.type] or ""
                    factsheet['configuration']['personalization'] = cfg.personalization != {}
                    factsheet['configuration']['differential_privacy'] = cfg.nbafl.use or False
                    factsheet['configuration']['dp_epsilon'] = cfg.nbafl.epsilon or 0
                    factsheet['configuration']['total_round_num'] = cfg.federate.total_round_num or 0
                    factsheet['configuration']['learning_rate'] = cfg.train.optimizer.lr or 0
                    factsheet['configuration']['local_update_steps'] = cfg.train.local_update_steps or 0
                    # set data specifications
                    factsheet['data']['provenance'] = cfg.data.type
                    factsheet['data']['preprocessing'] = cfg.data.transform

                if trainer_context is not None:
                    logger.info("FactSheet: Populating shared client training model params")
                    factsheet['configuration']['trainable_param_num'] = len(trainer_context['trainable_para_names']) or 0

                if eval_results is not None:
                    logger.info("FactSheet: Populating model evaluation results")
                    factsheet['performance']['test_loss_avg'] = eval_results.client_summarized_avg.test_loss or 0
                    factsheet['performance']['test_acc_avg'] = eval_results.client_summarized_avg.test_acc or 0
                    factsheet['performance']['test_feature_importance_std'] = eval_results.client_summarized_avg.test_feature_importance_std or 0
                    factsheet['fairness']['test_acc_avg'] = eval_results.client_summarized_avg.test_acc or 0
                    factsheet['fairness']['test_acc_std'] = eval_results.client_summarized_fairness.test_acc_std or 0
                    factsheet['fairness']['class_imbalance'] = eval_results.client_summarized_avg.test_class_imbalance or sys.maxsize

                if client_selection is not None:
                    logger.info("FactSheet: Populating client selection results")
                    factsheet['fairness']['selection_avg'] = np.mean([x for x in client_selection.values()])
                    factsheet['fairness']['selection_std'] = np.std([x for x in client_selection.values()])
            except JSONDecodeError as e:
                logger.warning(f"Either {factsheet_file} or {model_map_file} is invalid")
                logger.error(e)
            f.seek(0)
            f.truncate()
            json.dump(factsheet, f)
            f.close()

    def register_selection(self, clients, total_round_num, round):
        client_selection_file = os.path.join(self.outdir, self.client_selection_file_nm)
        if round == -1:
            logger.info("Client selection: Setting up selection rate map")
        else:
            logger.info(f"Client selection: Updating selection rate after round {round}")
        hashed_clients = [hashids.encode(id) for id in clients]
        if not os.path.exists(client_selection_file):
            with open(client_selection_file, 'a+') as f:
                results = {}
                update_frequency(results, hashed_clients, total_round_num, round)
                f.seek(0)
                f.truncate()
                json.dump(results, f)
                f.close()
        else:
            with open(client_selection_file, 'r+') as f:
                try:
                    results = json.load(f)
                    update_frequency(results, hashed_clients, total_round_num, round)
                except JSONDecodeError as e:
                    logger.warning(f"{client_selection_file} is invalid")
                    logger.error(e)
                f.seek(0)
                f.truncate()
                json.dump(results, f)
                f.close()

    def evaluate(self):
        factsheet_file = os.path.join(self.outdir, self.factsheet_file_nm)
        metrics_cfg_file = os.path.join(dirname, f"configs/{self.eval_metrics_file_nm}")
        out_file = os.path.join(self.outdir, self.out_file_nm)

        if not os.path.exists(factsheet_file):
            logger.error(f"{factsheet_file} is missing! Please check documentation.")
            return

        if not os.path.exists(metrics_cfg_file):
            logger.error(f"{metrics_cfg_file} is missing! Please check documentation.")
            return

        with open(factsheet_file, 'r') as f, \
                open(metrics_cfg_file, 'r') as m:
            factsheet = json.load(f)
            metrics_cfg = json.load(m)
            input_docs = {'factsheet': factsheet}
            for key, value in metrics_cfg.items():
                pillar = TrustPillar(key, value, input_docs)
                result = pillar.evaluate()
                write_results(out_file, str(result) + "\n")