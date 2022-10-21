# FederatedTrust
An algorithm to calculate trustworthiness score of a given federated learning framework. In this study, `FederateScope` framework is chosen to demonstrate the usage of the algorithm. The current methods provided are written based on how `FederatedScope` works. For future work, more methods can be created for other types of federated learning frameworks. 

# Set up
1. install the package 
```
pip install federatedTrust
```

2. Before training starts, register the logger 
```
from federatedTrust.monitor import register_logger

out_dir = os.path.join(os.getcwd(), 'exp/results') # example output directory
register_logger(out_dir)
```

3. During training rounds, register the client selection from the server side (if selected client ids are available)
```
from federatedTrust.input import register_selection

out_dir = os.path.join(os.getcwd(), 'exp/results/') # example output directory
register_selection(out_dir, clients=[id_1, id_2, id_3,...,id_n], total_round_num=10)
```

4. Populate the FactSheet from the server side with server side configs and model evaluation results
```
from federatedTrust.input import populate_factsheet

out_dir = os.path.join(os.getcwd(), 'exp/results/') # example output directory
populate_factsheet(out_dir, server_cfg_file=<file>, model_context=<obj>, eval_results_file=<file>, client_selection_file=<file>)

# required param: 
#   out_dir: output file directory
# optional params:
#   server_cfg: server side configuration file, e.g., /example/fs_config.yaml
#   model_context: model context object
#   eval_results: model evaluation results file, e.g., /example/eval_results.log
#   client_selection: client selection file, e.g., client_selection.json file
```

5. Assess the trustworthiness:
```
from federatedTrust.output import assess

# make sure that the factsheet is populated
# check the federatedTrust/configs/eval_metrics.json for evaluation criteria

out_dir = os.path.join(os.getcwd(), 'exp/results') # example output file directory
assess(out_dir)

# result is logged in the federatedTrust_results.log file in the output directory
# operational logging is captures in federatedTrust_print.log file in the output directory
```

# Example assessment output
```
{'accountability': 1.0, 'components': {'project': 1.0, 'data': 1.0, 'participants': 1.0, 'configuration': 1.0000000000000002, 'performance': 1.0, 'fairness': 1.0}}
{'architectural_design': 0.5238095238095237, 'components': {'design_patterns': 1.0, 'noniid_data_handling': 0.0, 'optimization': 0.5714285714285714}}
{'reliability': 0.8239407546163156, 'components': {'client_reliability': 0.8181818181818181, 'model_reliability': 0.8296996910508131}}
{'explainability': 0.4948110714948221, 'components': {'algorithmic_transparency': 0.08962214298964417, 'interpretability': 0.9}}
{'fairness': 0.45833333333333337, 'components': {'selection_fairness': 0.08333333333333337, 'performance_fairness': 0.8333333333333334}}
{'privacy': 0.6418734532595862, 'components': {'technique': 1.0, 'uncertainty': 0.9166666666666666, 'indistinguishability': 0.008953693112092154}}
```