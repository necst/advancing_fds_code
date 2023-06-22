from sklearn import metrics
import numpy as np

from FDS import model_creation
from utils.utilities import split_train_test
from os import path, makedirs
from tqdm import tqdm


def cost_accuracy(y_true, y_predicted):
    """
    Implements a custom loss function, with a weighted missprediction cost that
    ranks higher FN rather than FP. The weight is the ratio between legit and
    fraudulent transactions

    TP: Frauds correctly predicted
    TN: Legit correctly predicted
    FN: Frauds predicted as legit
    FP: Legit predicted as frauds

    :param y_true: The true classification values ('Fraud' attribute)
    :param y_predicted: The classification values predicted by the model
    :return: The custom metric score
    """
    TP = np.sum(np.logical_and(y_predicted == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_predicted == 0, y_true == 0))
    FN = np.sum(np.logical_and(y_predicted == 0, y_true == 1))
    FP = np.sum(np.logical_and(y_predicted == 1, y_true == 0))
    return compute_metric_score(TP, TN, FN, FP)


def compute_metric_score(TP, TN, FN, FP, print_score=False):
    ratio_cost = (TN + FP) / (TP + FN)
    missprediction_cost = FN * ratio_cost + FP
    weighted_trans = TN + FP + ratio_cost * (TP + FN)
    score = 1 - missprediction_cost / weighted_trans
    if print_score:
        print(f"TP:{TP}, TN:{TN}, FN:{FN}, FP:{FP}")
        print("C-Accuracy: %f" % score)
    return score


def manual_compute_scores(TP, TN, FN, FP):
    score_dict = {"TP": TP, "TN": TN, "FN": FN, "FP": FP}
    TPR = float(TP) / (TP + FN)
    score_dict["TPR"] = TPR
    FPR = float(FP) / (FP + TN)
    score_dict["FPR"] = FPR
    precision = float(TP) / (TP + FP)
    score_dict["precision"] = precision
    accuracy = float(TP + TN) / (TP + TN + FP + FN)
    score_dict["accuracy"] = accuracy
    F1 = float(2 * TP) / (2 * TP + FP + FN)
    score_dict["F1"] = F1
    # todo add Weighted Matthews Coefficient
    custom_metric_score = compute_metric_score(TP, TN, FN, FP)
    score_dict["metric_score"] = custom_metric_score
    return score_dict


def compute_matthews_weights(y_true):
    fraud = sum(y_true)
    legit = len(y_true) - fraud
    total = len(y_true)
    legit_weight = fraud / total
    fraud_weight = legit / total
    weight_list = []
    for trans in y_true:
        if trans == 1:
            weight_list.append(fraud_weight)
        else:
            weight_list.append(legit_weight)
    weight_arr = np.array(
        weight_list,
    )
    return weight_arr


def compute_prec_rec_AUC(FP, FN, TP, TN):
    precision_1 = TP / (TP + FP)
    precision_0 = TN / (TN + FN)
    recall_1 = TP / (TP + FN)
    recall_0 = TN / (TN + FP)
    return metrics.auc(recall_0, precision_0), metrics.auc(
        recall_1, precision_1
    )


