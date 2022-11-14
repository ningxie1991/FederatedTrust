import json
import logging
import os
import shutil
from json import JSONDecodeError

import numpy as np
from numpy import NaN
from tabulate import tabulate

from federatedTrust.calculation import get_cv, get_normalized_scores
from federatedTrust.pillar import TrustPillar
from federatedTrust.utils import read_file, read_eval_results_log, update_selection_rate, \
    read_system_metrics_log, write_results_json, count_class_samples

dirname = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


class TrustMetricManager:
    def __init__(self, outdir):
        """Manager class to help store the output directory
           and handle calls from the FL framework
           :param outdir: an ouput directory
        """
        self.outdir = outdir
        self.config_file_nm = "config.yaml"
        self.factsheet_file_nm = "factsheet.json"
        self.factsheet_template_file_nm = "factsheet_template.json"
        self.client_selection_file_nm = "client_selection.json"
        self.class_distribution_file_nm = "class_distribution.json"
        self.eval_results_file_nm = "eval_results.log"
        self.system_metrics_file_nm = "system_metrics.log"
        self.eval_metrics_file_nm = "eval_metrics_v1.json"
        self.model_map_file_nm = "model_map.json"
        self.log_nm = "federatedtrust_print.log"
        self.out_json_nm = "federatedtrust_results.json"
        self.stats_file_nm = "statistics.json"
        self.register_logger()

    def register_logger(self):
        """Confitures the logger
        """
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
                           system_metrics_file=None, stats_file=None):
        """Populates the factsheet with values
            :param mode: development or production mode
            :param cfg_file: the config file name
            :param trainer_context: trainer context
            :param eval_results_file: the evaluation results file name
            :param system_metrics_file: the system metric file name
            :param client_selection_file: the client selectio file name
            :param class_distribution_file: the class distributio file name
            :param feature_importance_cv: the feature importance coefficient of variation
            :param test_clever: the CLEVER score
        """
        factsheet_file = os.path.join(self.outdir, self.factsheet_file_nm)
        factsheet_template = os.path.join(dirname, f"configs/{self.factsheet_template_file_nm}")
        model_map_file = os.path.join(dirname, f"configs/{self.model_map_file_nm}")

        # for development purpose
        if mode == "development":
            cfg_file = os.path.join(dirname, f"example/{self.config_file_nm}")
            eval_results_file = os.path.join(dirname, f"example/{self.eval_results_file_nm}")
            system_metrics_file = os.path.join(dirname, f"example/{self.system_metrics_file_nm}")
            stats_file = os.path.join(dirname, f"example/{self.stats_file_nm}")
            trainer_context = {'trainable_para_names': ['a' for x in range(12)]}

        cfg = None if cfg_file is None else read_file(self.outdir, cfg_file)
        eval_results = None if eval_results_file is None else read_eval_results_log(self.outdir, eval_results_file)
        system_metrics = None if system_metrics_file is None else read_system_metrics_log(self.outdir, system_metrics_file)
        status = None if stats_file is None else read_file(self.outdir, stats_file)

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
                    factsheet['participants']['client_num'] = cfg.federate.client_num or NaN
                    factsheet['participants']['sample_client_rate'] = cfg.federate.sample_client_rate or NaN
                    factsheet['participants']['client_selector'] = cfg.federate.sampler or ""
                    # set configuration
                    factsheet['configuration']['optimization_algorithm'] = cfg.federate.method or ""
                    factsheet['configuration']['training_model'] = model_map[cfg.model.type] or ""
                    factsheet['configuration']['personalization'] = cfg.personalization != {} and cfg.personalization.local_param != []
                    factsheet['configuration']['differential_privacy'] = cfg.nbafl.use or False
                    factsheet['configuration']['dp_epsilon'] = cfg.nbafl.epsilon or NaN
                    factsheet['configuration']['total_round_num'] = cfg.federate.total_round_num or NaN
                    factsheet['configuration']['learning_rate'] = cfg.train.optimizer.lr or NaN
                    factsheet['configuration']['local_update_steps'] = cfg.train.local_update_steps or NaN
                    # set data specifications
                    factsheet['data']['provenance'] = cfg.data.type
                    factsheet['data']['preprocessing'] = cfg.data.transform

                if trainer_context is not None:
                    logger.info("FactSheet: Populating shared client training model params")
                    factsheet['configuration']['trainable_param_num'] = len(trainer_context['trainable_para_names']) or NaN

                if eval_results is not None:
                    logger.info("FactSheet: Populating model evaluation results")
                    factsheet['performance']['test_loss_avg'] = eval_results.client_summarized_avg.test_loss or NaN
                    factsheet['performance']['test_acc_avg'] = eval_results.client_summarized_avg.test_acc or NaN

                    test_acc_std = eval_results.client_summarized_fairness.test_acc_std or NaN
                    test_acc_avg = eval_results.client_summarized_avg.test_acc or NaN
                    test_acc_cv = get_cv(std=test_acc_std, mean=test_acc_avg)
                    factsheet['fairness']['test_acc_cv'] = 1 if test_acc_cv > 1 else test_acc_cv

                if system_metrics is not None:
                    factsheet['system']['avg_time_minutes'] = system_metrics['sys_avg/fl_end_time_minutes'] or NaN
                    factsheet['system']['avg_model_size'] = system_metrics['sys_avg/total_model_size'] or ""
                    factsheet['system']['avg_upload_bytes'] = system_metrics['sys_avg/total_upload_bytes'] or ""
                    factsheet['system']['avg_download_bytes'] = system_metrics['sys_avg/total_download_bytes'] or ""

                if status is not None:
                    class_distribution = status["class_distribution"]
                    if class_distribution is not None:
                        logger.info("FactSheet: Populating class distribution results")
                        class_samples_sizes = [x for x in class_distribution.values()]
                        class_imbalance = get_cv(list=class_samples_sizes)
                        factsheet['fairness']['class_imbalance'] = 1 if class_imbalance > 1 else class_imbalance

                    client_selection = status["client_selection"]
                    if client_selection is not None:
                        logger.info("FactSheet: Populating client selection results")
                        selections = [x for x in client_selection.values()]
                        selection_cv = get_cv(list=selections)
                        factsheet['fairness']['selection_cv'] = 1 if selection_cv > 1 else selection_cv

                    entropy_distribution = status["entropy_distribution"]
                    if entropy_distribution is not None:
                        logger.info("FactSheet: Populating client entropy results")
                        normalized_arr = get_normalized_scores([x for x in entropy_distribution.values()])
                        factsheet['data']['avg_entropy'] = np.mean(normalized_arr) or 0
            except JSONDecodeError as e:
                logger.warning(f"Either {factsheet_file} or {model_map_file} is invalid")
                logger.error(e)
            f.seek(0)
            f.truncate()
            json.dump(factsheet, f)
            f.close()

    def gather_stats(self, stats_key, stats_info):
        """Gather stats for the FL model
            :param stats_key: the key for the stats
            :param stats_info: the data for the stats
            e.g.
            key = class_distribution, stats_info = {"client_id": 1, "dataloader": {}}
            key = client_selection, stats_info = {"clients": [1,2,3..], "total_round_num": 25, "round": 1}
        """
        stats_file_path = os.path.join(self.outdir, self.stats_file_nm)
        with open(stats_file_path, 'a+') as f:
            if os.stat(stats_file_path).st_size == 0:
                results = {"class_distribution": {}, "entropy_distribution": {}, "client_selection": {}}
                self.update_stats(results, stats_key, stats_info)
                f.seek(0)
                f.truncate()
                json.dump(results, f)
                f.close()
            else:
                try:
                    f.seek(0)  # move file pointer to the beginning
                    results = json.load(f)
                    self.update_stats(results, stats_key, stats_info)
                    f.seek(0)
                    f.truncate()
                    json.dump(results, f)
                    f.close()
                except JSONDecodeError as e:
                    logger.warning(f"{stats_file_path} is invalid")
                    logger.error(e)

    @staticmethod
    def update_stats(results, stats_key, stats_info):
        if stats_key == "class_distribution":
            count_class_samples(results["class_distribution"], results["entropy_distribution"], stats_info)
        elif stats_key == "client_selection":
            update_selection_rate(results["client_selection"], stats_info)

    def evaluate(self, use_weights=False):
        """Evaluates the trustworthiness score
            :param use_weights: True to turn on the weights in the metric config file, default to False
            :return the result JSON
        """
        factsheet_file = os.path.join(self.outdir, self.factsheet_file_nm)
        metrics_cfg_file = os.path.join(dirname, f"configs/{self.eval_metrics_file_nm}")
        out_json = os.path.join(self.outdir, self.out_json_nm)

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
            metrics = metrics_cfg.items()
            input_docs = {'factsheet': factsheet}

            result_json = {"trust_score": 0, "pillars": []}
            final_score = 0
            avg_weight = 1 / len(metrics)
            result_print = []
            weights = {"robustness": 0.2, "privacy": 0.2, "fairness": 0.2, "explainability": 0.2, "accountability": 0.1, "architectural_soundness": 0.1}
            for key, value in metrics:
                pillar = TrustPillar(key, value, input_docs, use_weights)
                score, result = pillar.evaluate()
                weight = weights.get(key, avg_weight) if use_weights else avg_weight;
                final_score += weight * score
                result_print.append([key, score])
                result_json["pillars"].append(result)
            final_score = round(final_score, 2)
            result_json["trust_score"] = final_score
            write_results_json(out_json, result_json)
            print(tabulate(result_print, headers = ["trust_score", final_score], tablefmt='psql', showindex=False))
            return result_json