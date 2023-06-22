import os
from pathlib import Path
from time import time
import json
import sys

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import scipy.stats
from utils.utilities import load_features_from_file, split_train_test, calc_pos_weight
from FDS.model_creation import default_builder
from dataset import dataset
from model_selection.functions import cv_eval, save_time_info


USAGE = (
    "usage: python3 -m model_selection.fast_wrapper_method MODEL"
    " DATASET_FOLDER PARAMS_FILE FEATURE_FILE"
)
OUTFILE = "%s_fast_wrapper.csv"
SIX_WEEK_TS = pd.Timestamp(year=2014, month=12, day=7, hour=23, minute=59)

if __name__ == "__main__":

    tic = time()

    model_name = sys.argv[1]
    dataset_year = "2014_15"
    dataset_folder = sys.argv[2]
    params_file = sys.argv[3]
    feature_file = sys.argv[4]

    calc_class_weight = model_name == "neural_network"

    ts_split = SIX_WEEK_TS

    results_dir = Path(os.getcwd()) / "secai" / "model_selection" / Path(dataset_folder).parts[-1]
    timings_txt = results_dir / "timings.txt"

    feature_set = load_features_from_file(feature_file)
    cols_to_read = feature_set + ["Fraud", "Timestamp"]

    df = dataset.load_aggregated(dataset_year, usecols=cols_to_read, dir=dataset_folder)  # type: ignore

    outfile = results_dir / (OUTFILE % model_name)

    with open(params_file, "r") as f:
        fs_params = json.load(f)[model_name]

    x_train, y_train, _, _ = split_train_test(df, ts_split)

    pos_weight = calc_pos_weight(y_train)

    model = default_builder(
        fs_params,
        model_name,
        scale_pos_weight=pos_weight,
        input_shape=(1,),  # ann will get a single feature in input!
    )

    feature_scores = {}
    feature_iterator = tqdm(x_train.columns.tolist())

    for feature in feature_iterator:

        feature_iterator.set_description("%s" % feature)
        x_tmp = x_train[feature]

        mean_score, mean_fit_time, fold_scores, fold_fit_times = cv_eval(
            TimeSeriesSplit(n_splits=3),
            x_tmp.to_frame(),
            y_train,
            StandardScaler(),
            calc_class_weight,
            model,
        )

        feature_scores[feature] = mean_score

    feature_names = feature_scores.keys()
    np_feature_scores = np.array(list(feature_scores.values()))
    average_score = np_feature_scores.mean()
    std_score = np_feature_scores.std()

    features_over_avg = np.array(np_feature_scores > average_score, dtype=int)
    results = pd.DataFrame(
        {
            "feature_name": feature_names,
            "feature_score": np_feature_scores,
            "average_score": [average_score] * len(feature_names),
            "is_over_avg": features_over_avg,
            "fs_params": [fs_params] * len(feature_names),
            "dataset_year": [dataset_year] * len(feature_names),
            "dataset_folder": [dataset_folder] * len(feature_names),
            "ts_split": [ts_split] * len(feature_names),
        }
    )

    first_quartile = results.feature_score.quantile(0.25)
    first_quartile_from_gaussian = scipy.stats.norm(average_score, std_score).ppf(0.25)
    results["features_over_1st_quart"] = np.array(
        np_feature_scores > first_quartile, dtype=int
    )
    results["features_over_1st_quart_gauss"] = np.array(
        np_feature_scores > first_quartile_from_gaussian, dtype=int
    )

    results.sort_values(by="feature_score", ascending=False).to_csv(outfile)

    elapsed = time() - tic
    save_time_info(timings_txt, "fast_wrapper_method", elapsed)
