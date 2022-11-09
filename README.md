# FederatedTrust
An algorithm to calculate trustworthiness score of a given federated learning framework. In this study, `FederateScope` framework is chosen to demonstrate the usage of the algorithm. The current methods provided are written based on how `FederatedScope` works. For future work, more methods can be created for other types of federated learning frameworks. 

# Set up
1. Build the package inside the FederatedTrust workspace
```
python setup.py bdist_wheel
```

2. Install the package in the FederatedScope development framework
```
pip install ...\dist\FederatedTrust-0.1.0-py3-none-any.whl 
```
3. Before training starts, create the TrustMetricManager in the server side and client side.
```
from federatedTrust.metric import TrustMetricManager

# set up TrustMetricManager with the output directory
trust_metric_manager = TrustMetricManager(output_dier)
```
4. From the server side, set up the client selection map
```
# set up the client selection map with all client ids 
trust_metric_manager.register_selection(all_client_ids, total_round_num, -1)
```
5. From each client side, count the sample size by class and create the unified class distribution.
```
# pass the client data to count the sample size by class
trust_metric_manager.register_class_distribution(data)
```
6. Populate the FactSheet from the server side with configs from the config file.
```
trust_metric_manager.populate_factsheet(cfg_file="config.yaml")
```
7. During training rounds, register the client selection from the server side (if selected client ids are available)
```
trust_metric_manager.register_selection(selected_client_ids, total_round_num, round)
```
8. When training finishes, populate the FactSheet with the valuation result, the system metrics and the client selection file.
```
trust_metric_manager.populate_factsheet(eval_results_file="eval_results.log",
                                        system_metrics_file="system_metrics.log", 
                                        client_selection_file="client_selection.json")
```
9. Assess the trustworthiness:
```
# pass one test sample, the global model and some config data 
# to evaluate the entire trustworthiness score
self.trust_metric_manager.evaluate(test_sample, model, cfg)
```

# Example assessment output
```
+-------------------------+--------+
| trust_score             |   0.62 |
|-------------------------+--------|
| robustness              |   0.38 |
| privacy                 |   0.42 |
| fairness                |   0.49 |
| explainability          |   0.77 |
| accountability          |   0.9  |
| architectural_soundness |   0.78 |
+-------------------------+--------+
```
```
{
  "trust_score": 0.62,
  "pillars": [
    {
      "robustness": {
        "score": 0.38,
        "notions": [
          {
            "resilience_to_attacks": {
              "score": 0.05,
              "metrics": [
                {
                  "certified_robustness": {
                    "score": 0.05
                  }
                }
              ]
            }
          },
          {
            "algorithm_robustness": {
              "score": 0.18,
              "metrics": [
                {
                  "performance": {
                    "score": 0.36
                  }
                },
                {
                  "personalization": {
                    "score": 0
                  }
                }
              ]
            }
          },
          {
            "client_reliability": {
              "score": 0.91,
              "metrics": [
                {
                  "scale": {
                    "score": 0.91
                  }
                }
              ]
            }
          }
        ]
      }
    },
    {
      "privacy": {
        "score": 0.42,
        "notions": [
          {
            "technique": {
              "score": 1.0,
              "metrics": [
                {
                  "differential_privacy": {
                    "score": 1
                  }
                }
              ]
            }
          },
          {
            "uncertainty": {
              "score": 0.07,
              "metrics": [
                {
                  "entropy": {
                    "score": 0.07
                  }
                }
              ]
            }
          },
          {
            "indistinguishability": {
              "score": 0.2,
              "metrics": [
                {
                  "global_privacy_risk": {
                    "score": 0.2
                  }
                }
              ]
            }
          }
        ]
      }
    },
    {
      "fairness": {
        "score": 0.49,
        "notions": [
          {
            "selection_fairness": {
              "score": 0.88,
              "metrics": [
                {
                  "selection_variation": {
                    "score": 0.88
                  }
                }
              ]
            }
          },
          {
            "performance_fairness": {
              "score": 0.58,
              "metrics": [
                {
                  "accuracy_variation": {
                    "score": 0.58
                  }
                }
              ]
            }
          },
          {
            "class_distribution": {
              "score": 0.0,
              "metrics": [
                {
                  "class_imbalance": {
                    "score": 0
                  }
                }
              ]
            }
          }
        ]
      }
    },
    {
      "explainability": {
        "score": 0.77,
        "notions": [
          {
            "interpretability": {
              "score": 0.545,
              "metrics": [
                {
                  "algorithmic_transparency": {
                    "score": 0.09
                  }
                },
                {
                  "model_size": {
                    "score": 1.0
                  }
                }
              ]
            }
          },
          {
            "post_hoc_methods": {
              "score": 1.0,
              "metrics": [
                {
                  "feature_importance": {
                    "score": 1
                  }
                }
              ]
            }
          }
        ]
      }
    },
    {
      "accountability": {
        "score": 0.9,
        "notions": [
          {
            "factsheet_completeness": {
              "score": 0.9042857142857141,
              "metrics": [
                {
                  "project_specs": {
                    "score": 0.33
                  }
                },
                {
                  "participants": {
                    "score": 1.0
                  }
                },
                {
                  "data": {
                    "score": 1.0
                  }
                },
                {
                  "configuration": {
                    "score": 1.0
                  }
                },
                {
                  "performance": {
                    "score": 1.0
                  }
                },
                {
                  "fairness": {
                    "score": 1.0
                  }
                },
                {
                  "system": {
                    "score": 1.0
                  }
                }
              ]
            }
          }
        ]
      }
    },
    {
      "architectural_soundness": {
        "score": 0.78,
        "notions": [
          {
            "client_management": {
              "score": 1.0,
              "metrics": [
                {
                  "client_selector": {
                    "score": 1.0
                  }
                }
              ]
            }
          },
          {
            "optimization": {
              "score": 0.57,
              "metrics": [
                {
                  "algorithm": {
                    "score": 0.57
                  }
                }
              ]
            }
          }
        ]
      }
    }
  ]
}
```