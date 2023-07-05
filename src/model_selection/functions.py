import sys
from os import makedirs
from time import time
from typing import Dict, Tuple, List, Callable
from pathlib import Path

from sklearn import metrics

import numpy as np
import pandas as pd

from config.config import MODEL_SELECTION_PATH
from utils.utilities import calc_proportional_class_weight, load_features_from_file
from utils.evaluation_functions import cost_accuracy, compute_matthews_weights
from FDS import Model


model_list = [m.value for m in Model]
dataset_list = ["2014_15"]


def cv_eval(
    cv_splitter,  # type please ?
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sc,  # : StandardScaler, ? No, any Sklearn Scaler
    calc_class_weight: bool,
    model,  # Union[SKLEARN MODEL, KERAS, XGBCLASSIFIER]
    predict_probs: bool = False,
    proba_threshold: float = 0.5,
    eval_metric: Callable = cost_accuracy
) -> Tuple[float, float, List[float], List[float]]:
    """
    Runs cross-validation with given model and dataset splitting strategy.
    Returns mean score (error, time) and score for each step.
    """
    fold_scores = []
    fold_fit_times = []

    for train_index, test_index in cv_splitter.split(x_train):

        x_train_part, x_val_part = (
            x_train.iloc[train_index],
            x_train.iloc[test_index],
        )

        y_train_part, y_val_part = (
            y_train.iloc[train_index],
            y_train.iloc[test_index],
        )

        reshape = len(x_train.shape) == 1

        if reshape:
            # print("reshaping", len(x_train.shape))
            x_train_part = x_train_part.to_numpy().reshape(-1, 1)  # type: ignore
            x_val_part = x_val_part.to_numpy().reshape(-1, 1)  # type: ignore


        x_train_part_scaled = sc.fit_transform(x_train_part)
        x_val_part_scaled = sc.transform(x_val_part)

        fit_tic = time()

        if calc_class_weight:  # only keras models ! :-)
            class_weight = calc_proportional_class_weight(y_train_part)
            model.fit(x_train_part_scaled, y_train_part, class_weight=class_weight)
        else:  # all the others
            model.fit(x_train_part_scaled, y_train_part)

        fit_elapsed = time() - fit_tic

        if predict_probs:
            predictions = model.predict_proba(x_val_part_scaled)[:, 1]
            predictions = predictions > proba_threshold
        else:
            predictions = model.predict(x_val_part_scaled)
            predictions = np.ravel(predictions)  # necessary for ANN

        score = eval_metric(y_val_part, predictions)

        fold_scores.append(score)
        fold_fit_times.append(fit_elapsed)

    mean_score = np.mean(fold_scores)
    mean_fit_time = np.mean(fold_fit_times)

    return float(mean_score), float(mean_fit_time), fold_scores, fold_fit_times


def evaluate_folds(
    cv_splitter,  # type please ?
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sc,  # : StandardScaler, ? No, any Sklearn Scaler
    calc_class_weight: bool,
    model,  # Union[SKLEARN MODEL, KERAS, XGBCLASSIFIER]
    predict_probs: bool = False,
    proba_threshold: float = 0.5,
) -> Dict:
    """
    Runs cross-validation with given model and dataset splitting strategy.
    Returns mean score (error, time) and score for each step.
    """
    
    folds = {
        "cost_accuracy": [],
        "precision": [],
        "f1_score": [],
        "recall": [],
        "tpr": [],
        "fpr": [],
        "tn": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "matthews_corr": [],
        "matthews_corr_weighted": [],
        "fit_times": [],
        "auc_roc": [],
        "auc_prc": [],
    }

    for train_index, test_index in cv_splitter.split(x_train):

        x_train_part, x_val_part = (
            x_train.iloc[train_index],
            x_train.iloc[test_index],
        )

        y_train_part, y_val_part = (
            y_train.iloc[train_index],
            y_train.iloc[test_index],
        )

        reshape = len(x_train.shape) == 1

        if reshape:
            # print("reshaping", len(x_train.shape))
            x_train_part = x_train_part.to_numpy().reshape(-1, 1)  # type: ignore
            x_val_part = x_val_part.to_numpy().reshape(-1, 1)  # type: ignore


        x_train_part_scaled = sc.fit_transform(x_train_part)
        x_val_part_scaled = sc.transform(x_val_part)

        fit_tic = time()

        if calc_class_weight:  # only keras models ! :-)
            class_weight = calc_proportional_class_weight(y_train_part)
            model.fit(x_train_part_scaled, y_train_part, class_weight=class_weight)
        else:  # all the others
            model.fit(x_train_part_scaled, y_train_part)

        fit_elapsed = time() - fit_tic

        if predict_probs:
            probs = model.predict_proba(x_val_part_scaled)[:, 1]
            predictions = probs > proba_threshold
        else:
            predictions = model.predict(x_val_part_scaled)
            predictions = np.ravel(predictions)  # necessary for ANN

        c_acc = cost_accuracy(y_val_part, predictions)
        
        conf_matrix = metrics.confusion_matrix(y_val_part, predictions)
        fpr_ax, tpr_ax, _ = metrics.roc_curve(y_val_part, probs)

        roc_auc = metrics.auc(fpr_ax, tpr_ax)

        precision, recall, _ = metrics.precision_recall_curve(
            np.array(y_val_part).astype(bool), probs, 
        )
        
        prc_auc = metrics.auc(recall, precision)

        TN, FP, FN, TP = conf_matrix.ravel()

        total_trans = TP + TN + FN + FP
        frauds = TP + FN
        legit = TN + FP
        ratio = frauds / legit

        tpr = TP / (TP+FN)
        fpr = FP / (FP+TN)

        matthews_correlation = metrics.matthews_corrcoef(y_val_part, predictions)

        c_matthews_corr = metrics.matthews_corrcoef(
            y_val_part, predictions, sample_weight=compute_matthews_weights(y_val_part)
        )

        report = metrics.classification_report(y_val_part, predictions, output_dict=True)

        precision, recall, f1_score, support = tuple(report['1'].values())

        folds["cost_accuracy"].append(c_acc)
        folds["precision"].append(precision)
        folds["f1_score"].append(f1_score)
        folds["recall"].append(recall)
        folds["tpr"].append(tpr)
        folds["fpr"].append(fpr)
        folds["tn"].append(TN)
        folds["tp"].append(TP)
        folds["fp"].append(FP)
        folds["fn"].append(FN)
        folds["matthews_corr"].append(matthews_correlation)
        folds["matthews_corr_weighted"].append(c_matthews_corr)
        folds["auc_roc"].append(roc_auc)
        folds["auc_prc"].append(prc_auc)
        folds["fit_times"].append(fit_elapsed)
    
    perf_metrics = list(folds.keys())
    for metric in perf_metrics:
        for i, val in enumerate(folds[metric]):
            folds["%s_fold%d" % (metric, i)] = val
        folds["mean_%s" % metric] = np.mean(folds[metric])
        del folds[metric]

    return folds


def save_time_info(time_file: Path, header: str, elapsed: float):
    """Appends time record to given file."""
    with open(time_file, "a") as f:
        f.write("%s: %f\n" % (header, elapsed))


def get_results_paths(model_name: str, folder_name: str) -> Tuple[Path, Path]:
    """
    Generates path where results are saved. Creates directory if it does exist.
    Returns also file with time records of model selection steps.
    """
    results_dir = MODEL_SELECTION_PATH / folder_name / model_name
    makedirs(results_dir, exist_ok=True)
    timings_txt = results_dir / ("timings_%s.txt" % model_name)
    return results_dir, timings_txt


def read_model_and_dataset() -> Tuple[str, str]:
    """Returns script input parameters (model name, dataset name)"""
    model_name = sys.argv[1]
    dataset_year = sys.argv[2]
    assert model_name in model_list
    assert dataset_year in dataset_list
    return model_name, dataset_year
