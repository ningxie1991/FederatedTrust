import json
import logging
import os
import shutil
import sys
from json import JSONDecodeError

from scipy.stats import variation
from tabulate import tabulate

from federatedTrust.calculation import get_cv, get_feature_importance_cv, get_clever_score
from federatedTrust.pillar import TrustPillar
from federatedTrust.utils import read_file, read_eval_results_log, update_frequency, \
    read_system_metrics_log, write_results_json, count_class_samples

dirname = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


class TrustMetricManager:
    def __init__(self, outdir ):
        """Manager class to help store the output directory
           and handle calls from the FL framework
           :param outdir: an ouput directory
        """
        self.outdir = outdir
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
                           system_metrics_file=None, client_selection_file=None, class_distribution_file=None,
                           feature_importance_cv=None, test_clever=None):
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
            cfg_file = os.path.join(dirname, 'example/fs_config.yaml')
            eval_results_file = os.path.join(dirname, f"example/{self.eval_results_file_nm}")
            system_metrics_file = os.path.join(dirname, f"example/{self.system_metrics_file_nm}")
            client_selection_file = os.path.join(dirname, f"example/{self.client_selection_file_nm}")
            class_distribution_file = os.path.join(dirname, f"example/{self.class_distribution_file_nm}")
            model_context = {'trainable_para_names': ['a' for x in range(12)]}

        cfg = None if cfg_file is None else read_file(self.outdir, cfg_file)
        eval_results = None if eval_results_file is None else read_eval_results_log(self.outdir, eval_results_file)
        system_metrics = None if system_metrics_file is None else read_system_metrics_log(self.outdir, system_metrics_file)
        client_selection = None if client_selection_file is None else read_file(self.outdir, client_selection_file)
        class_distribution = None if class_distribution_file is None else read_file(self.outdir, class_distribution_file)

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
                    factsheet['configuration']['personalization'] = cfg.personalization != {} and cfg.personalization.local_param != []
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

                    test_acc_std = eval_results.client_summarized_fairness.test_acc_std or 0
                    test_acc_avg = eval_results.client_summarized_avg.test_acc or 0
                    factsheet['fairness']['test_acc_cv'] = 1 if get_cv(test_acc_std, test_acc_avg) > 1 else get_cv(test_acc_std, test_acc_avg)

                if system_metrics is not None:
                    factsheet['system']['avg_time_minutes'] = system_metrics['sys_avg/fl_end_time_minutes'] or 0
                    factsheet['system']['avg_model_size'] = system_metrics['sys_avg/total_model_size'] or ""
                    factsheet['system']['avg_upload_bytes'] = system_metrics['sys_avg/total_upload_bytes'] or ""
                    factsheet['system']['avg_download_bytes'] = system_metrics['sys_avg/total_download_bytes'] or ""

                if client_selection is not None:
                    logger.info("FactSheet: Populating client selection results")
                    selections = [x for x in client_selection.values()]
                    selection_cv = variation(selections)
                    factsheet['fairness']['selection_cv'] = 1 if selection_cv > 1 else selection_cv

                if class_distribution is not None:
                    logger.info("FactSheet: Populating class distribution results")
                    class_samples_sizes = [x for x in class_distribution.values()]
                    class_imbalance = variation(class_samples_sizes)
                    factsheet['fairness']['class_imbalance'] = 1 if class_imbalance > 1 else class_imbalance

                if feature_importance_cv is not None:
                    factsheet['performance']['test_feature_importance_cv'] =1 if feature_importance_cv > 1 else feature_importance_cv

                if test_clever is not None:
                    factsheet['performance']['test_clever'] = test_clever or 0
            except JSONDecodeError as e:
                logger.warning(f"Either {factsheet_file} or {model_map_file} is invalid")
                logger.error(e)
            f.seek(0)
            f.truncate()
            json.dump(factsheet, f)
            f.close()

    def register_selection(self, clients, total_round_num, round):
        """Updates the client selection map
            :param clients: the selected client IDs
            :param total_round_num: the total number of rounds
            :param round: the round index
        """
        client_selection_file = os.path.join(self.outdir, self.client_selection_file_nm)
        if round == -1:
            logger.info("Client selection: Setting up selection rate map")
        else:
            logger.info(f"Client selection: Updating selection rate after round {round}")
        if not os.path.exists(client_selection_file):
            with open(client_selection_file, 'a+') as f:
                results = {}
                update_frequency(results, clients, total_round_num, round)
                f.seek(0)
                f.truncate()
                json.dump(results, f)
                f.close()
        else:
            with open(client_selection_file, 'r+') as f:
                try:
                    results = json.load(f)
                    update_frequency(results, clients, total_round_num, round)
                except JSONDecodeError as e:
                    logger.warning(f"{client_selection_file} is invalid")
                    logger.error(e)
                f.seek(0)
                f.truncate()
                json.dump(results, f)
                f.close()

    def register_class_distribution(self, data):
        """Updates the class distribution map
            :param data: the current training data in the client
        """
        class_distribution_file = os.path.join(self.outdir, self.class_distribution_file_nm)
        if not os.path.exists(class_distribution_file):
            with open(class_distribution_file, 'a+') as f:
                results = {}
                count_class_samples(results, data)
                f.seek(0)
                f.truncate()
                json.dump(results, f)
                f.close()
        else:
            with open(class_distribution_file, 'r+') as f:
                try:
                    results = json.load(f)
                    count_class_samples(results, data)
                except JSONDecodeError as e:
                    logger.warning(f"{class_distribution_file} is invalid")
                    logger.error(e)
                f.seek(0)
                f.truncate()
                json.dump(results, f)
                f.close()

    def evaluate(self, test_sample=None, model=None, cfg=None, use_weights=False):
        """Evaluates the trustworthiness score
            :param test_sample: test sample for calculating feature importance cv and class imbalance
            :param model: the global model
            :param cfg: configs
            :param use_weights: True to turn on the weights in the metric config file
            :return the result JSON
        """
        if test_sample is not None and model is not None and cfg is not None:
            feature_importance_cv = get_feature_importance_cv(test_sample, model, cfg)
            test_clever = get_clever_score(test_sample, model, cfg)
            self.populate_factsheet(feature_importance_cv=feature_importance_cv, test_clever=test_clever)

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
                pillar = TrustPillar(key, value, input_docs)
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