from pathlib import Path
import sys
import os
from time import time
from itertools import product
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from utils.utilities import split_train_test, calc_pos_weight, load_features_from_file
from FDS.model_creation import default_builder
from dataset import dataset
from model_selection.grids import grids
from model_selection.functions import cv_eval, save_time_info


USAGE = (
    "USAGE: python3 -m model_selection.grid_search MODEL DATASET_FOLDER"
    " FEATURE_FILE GRID_NAME"
)
OUTFILE = "%s_grid_search_%s.csv"

SIX_WEEK_TS = pd.Timestamp(year=2014, month=12, day=7, hour=23, minute=59)


if __name__ == "__main__":
    tic = time()
    # model_name, dataset_year = read_model_and_dataset()

    model_name = sys.argv[1]
    dataset_folder = sys.argv[2]
    dataset_year = "2014_15"
    feature_file = sys.argv[3]
    grid_name = sys.argv[4]

    calc_class_weight = model_name == "neural_network"
    ts_split = SIX_WEEK_TS

    results_dir = Path(os.getcwd()) / "secai" / "model_selection" / Path(dataset_folder).parts[-1]
    timings_txt = results_dir / "timings.txt"

    feature_set = load_features_from_file(feature_file)
    cols_to_read = feature_set + ["Fraud", "Timestamp"]

    df = dataset.load_aggregated(dataset_year, usecols=cols_to_read, dir=dataset_folder)  # type: ignore

    outfile = results_dir / (OUTFILE % (model_name, grid_name))

    x_train, y_train, x_test, y_test = split_train_test(df, ts_split)

    print(x_train.columns)
    pos_weight = calc_pos_weight(y_train)

    print("train set: %d, test set: %d" % (x_train.shape[0], x_test.shape[0]))
    print("ratio test set on original set: %f" % (x_test.shape[0] / df.shape[0]))
    print("using only training set")

    combos = []  # all combinations of model parameters
    models = []

    for g in grids[grid_name][model_name]:
        # get grid of all neural network parameter combinations
        combos += list(product(*g.values()))
        # turn into list of dictionaries with parameters as kwargs
        models += [dict(zip(g.keys(), v)) for v in [c for c in combos]]

    # x_train.reset_index(drop=True, inplace=True)
    # y_train.reset_index(drop=True, inplace=True)

    input_shape = (x_train.shape[1],)

    try:
        past_df_results = pd.read_csv(outfile, index_col=0)
        print("Found past results. Extracting tested hyper-parameters.")
        tested_params = [
            tuple(yaml.full_load(d).values())
            for d in past_df_results["params"].tolist()
        ]
    except FileNotFoundError:
        tested_params = []

    # make model,
    for params in models:

        cv_results = {
            "params": [],
            "mean_test_score": [],
            "mean_fit_time": [],
            "dataset_year": [],
            "dataset_folder": [],
            "ts_split": [],
            "feature_set": [],
            "grid_name": [],
        }
        print(params)

        if tuple(params.values()) in tested_params:
            print("Already analyzed. Skipping!")
            continue

        try:
            model = default_builder(
                params, model_name, input_shape=input_shape, scale_pos_weight=pos_weight
            )
        except Exception as e:
            print("Cannot build model with params: %s" % params)
            print(e)
            continue

        model_score = []
        fit_times = []

        try:
            mean_score, mean_fit_time, fold_scores, fold_fit_times = cv_eval(
                TimeSeriesSplit(n_splits=3),
                x_train,
                y_train,
                StandardScaler(),
                calc_class_weight,
                model,
            )
        except Exception as e:
            print("Cannot fit model with params: %s" % params)
            print(e)
            continue

        cv_results["params"].append(params)
        cv_results["mean_test_score"].append(mean_score)
        cv_results["dataset_year"].append(dataset_year)
        cv_results["dataset_folder"].append(dataset_folder)
        cv_results["ts_split"].append(ts_split)
        cv_results["feature_set"].append(str(feature_set))
        cv_results["grid_name"].append(grid_name)

        for i, x in enumerate(fold_scores):
            key = "test_score_fold_%d" % i
            if key not in cv_results.keys():
                cv_results[key] = []
            cv_results[key].append(x)

        for i, x in enumerate(fold_fit_times):
            key = "fit_time_fold_%d" % i
            if key not in cv_results.keys():
                cv_results[key] = []
            cv_results[key].append(x)

        cv_results["mean_fit_time"].append(mean_fit_time)

        results_df = pd.DataFrame(cv_results)
        put_header = not os.path.exists(outfile)
        results_df.to_csv(outfile, mode="a", header=put_header)

    results_df = pd.read_csv(outfile, index_col=0)
    results_df.sort_values(by="mean_test_score", ascending=False).reset_index(
        drop=True
    ).to_csv(outfile)

    elapsed = time() - tic
    save_time_info(timings_txt, ("%s_grid_search" % grid_name), elapsed)
