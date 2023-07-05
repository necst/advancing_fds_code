from numpy import linspace, logspace, concatenate

res = 4

grids = {
    "initial_secai": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 8, base=2, num=res, dtype=int),
                "max_depth": linspace(5, 15, num=res, dtype=int),
                "class_weight": ["balanced"],
                "criterion": ["gini", "entropy"],
                "min_samples_split": linspace(2, 10, num=res, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1", "l2"],
                "tol": logspace(-3, -1, num=3),
                "C": logspace(-1, 2, num=2 * res),
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l2", "l1"],
                "loss": ["squared_hinge", "hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "C": logspace(-1, 2, num=res),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": linspace(2, 16, num=4, dtype=int),
                "learning_rate": linspace(0.1, 0.5, num=2),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=2),
                "subsample": logspace(-1, 0, num=2),
                "n_estimators": logspace(5, 7, base=2, num=4, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": logspace(8, 10, num=3, dtype=int, base=2),
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.1, 0.5, num=3),
            }
        ]
    },
    "final_secai": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 9, base=2, num=8, dtype=int),
                "max_depth": linspace(5, 15, num=4, dtype=int),
                "class_weight": ["balanced"],
                "criterion": ["entropy", "gini"],
                "min_samples_split": linspace(2, 15, num=3, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1", "l2"],
                "C": logspace(-1, 2, num=20),
                "class_weight": ["balanced"],
                "tol": logspace(-3, -1, num=2),
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l1", "l2"],
                "loss": ["squared_hinge", "hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-3, -1, num=2),
                "C": logspace(-1, 2, num=10),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": linspace(2, 16, num=4, dtype=int),
                "learning_rate": linspace(0.1, 0.5, num=3),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=4),
                "n_estimators": logspace(4, 7, base=2, num=5, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": [256],
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.0, 0.5, num=3),
            }
        ]
    },
    "final_lt_is": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 9, base=2, num=8, dtype=int),
                "max_depth": [5],
                "class_weight": ["balanced"],
                "criterion": ["entropy"],
                "min_samples_split": linspace(2, 15, num=8, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1", "l2"],
                "C": logspace(-1, 2, num=20),
                "class_weight": ["balanced"],
                "tol": [0.1],
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l1"],
                "loss": ["squared_hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "C": logspace(-1, 2, num=40),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": [2],
                "learning_rate": linspace(0.1, 0.5, num=10),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=4),
                "subsample": [1.0],
                "n_estimators": logspace(4, 7, base=2, num=5, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": [256],
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.3, 0.5, num=3),
            }
        ]
    },
    "final_lt_th": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 8, base=2, num=20, dtype=int),
                "max_depth": [8],
                "class_weight": ["balanced"],
                "criterion": ["entropy"],
                "min_samples_split": linspace(4, 10, num=res, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1"],
                "tol": logspace(-3, -1, num=3),
                "C": linspace(0.1, 3, num=10),
                "class_weight": ["balanced"],
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l1"],
                "loss": ["squared_hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "C": logspace(-1, 2, num=10),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": linspace(2, 6, num=5, dtype=int),
                "learning_rate": linspace(0.1, 0.5, num=3),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=3),
                "subsample": logspace(-1, 0, num=3),
                "n_estimators": logspace(5, 7, base=2, num=4, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": logspace(8, 10, num=3, dtype=int, base=2),
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.1, 0.5, num=3),
            }
        ]
    },
    "final_st_is": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 8, base=2, num=res, dtype=int),
                "max_depth": linspace(5, 15, num=res, dtype=int),
                "class_weight": ["balanced"],
                "criterion": ["gini", "entropy"],
                "min_samples_split": linspace(2, 10, num=res, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1", "l2"],
                "tol": logspace(-3, -1, num=3),
                "C": logspace(-1, 2, num=2 * res),
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l2", "l1"],
                "loss": ["squared_hinge", "hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "C": logspace(-1, 2, num=res),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": linspace(2, 16, num=4, dtype=int),
                "learning_rate": linspace(0.1, 0.5, num=2),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=2),
                "subsample": logspace(-1, 0, num=2),
                "n_estimators": logspace(5, 7, base=2, num=4, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": logspace(8, 10, num=3, dtype=int, base=2),
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.1, 0.5, num=3),
            }
        ]
    },
    "final_st_th": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 8, base=2, num=res, dtype=int),
                "max_depth": linspace(5, 15, num=res, dtype=int),
                "class_weight": ["balanced"],
                "criterion": ["gini", "entropy"],
                "min_samples_split": linspace(2, 10, num=res, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1", "l2"],
                "tol": logspace(-3, -1, num=3),
                "C": logspace(-1, 2, num=2 * res),
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l2", "l1"],
                "loss": ["squared_hinge", "hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "C": logspace(-1, 2, num=res),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": linspace(2, 16, num=4, dtype=int),
                "learning_rate": linspace(0.1, 0.5, num=2),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=2),
                "subsample": logspace(-1, 0, num=2),
                "n_estimators": logspace(5, 7, base=2, num=4, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": logspace(8, 10, num=3, dtype=int, base=2),
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.1, 0.5, num=3),
            }
        ]
    },
    "final_sf_is": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 8, base=2, num=res, dtype=int),
                "max_depth": linspace(5, 15, num=res, dtype=int),
                "class_weight": ["balanced"],
                "criterion": ["gini", "entropy"],
                "min_samples_split": linspace(2, 10, num=res, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1", "l2"],
                "tol": logspace(-3, -1, num=3),
                "C": logspace(-1, 2, num=2 * res),
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l2", "l1"],
                "loss": ["squared_hinge", "hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "C": logspace(-1, 2, num=res),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": linspace(2, 16, num=4, dtype=int),
                "learning_rate": linspace(0.1, 0.5, num=2),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=2),
                "subsample": logspace(-1, 0, num=2),
                "n_estimators": logspace(5, 7, base=2, num=4, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": logspace(8, 10, num=3, dtype=int, base=2),
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.1, 0.5, num=3),
            }
        ]
    },
    "final_sf_th": {
        "random_forest": [
            {
                "n_estimators": logspace(5, 8, base=2, num=res, dtype=int),
                "max_depth": linspace(5, 15, num=res, dtype=int),
                "class_weight": ["balanced"],
                "criterion": ["gini", "entropy"],
                "min_samples_split": linspace(2, 10, num=res, dtype=int),
            }
        ],
        "logistic_regression": [
            {
                "penalty": ["l1", "l2"],
                "tol": logspace(-3, -1, num=3),
                "C": logspace(-1, 2, num=2 * res),
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "solver": ["saga"],
                "max_iter": [500],
            }
        ],
        "support_vector_machine": [
            {
                "penalty": ["l2", "l1"],
                "loss": ["squared_hinge", "hinge"],
                "dual": [False],
                "class_weight": ["balanced"],
                "tol": logspace(-4, -1, num=4),
                "C": logspace(-1, 2, num=res),
                "max_iter": [7500],
            },
        ],
        "xgboost": [
            {
                "max_depth": linspace(2, 16, num=4, dtype=int),
                "learning_rate": linspace(0.1, 0.5, num=2),
                "booster": ["gbtree"],
                "gamma": logspace(-1, 1, num=2),
                "subsample": logspace(-1, 0, num=2),
                "n_estimators": logspace(5, 7, base=2, num=4, dtype=int),
            }
        ],
        "neural_network": [
            {
                "epochs": [15],
                "batch_size": logspace(8, 10, num=3, dtype=int, base=2),
                "fl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "sl_neurons": logspace(4, 6, num=3, dtype=int, base=2),
                "activation_funct": ["relu", "tanh"],
                "dropout_rate": linspace(0.1, 0.5, num=3),
            }
        ]
    }
}
