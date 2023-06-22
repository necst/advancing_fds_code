from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing import const as data_info
import uuid
from typing import List, Dict, Union
import json


DATASET_SPLIT_TS = pd.Timestamp(year=2014, month=12, day=7, hour=23, minute=59)


def generate_transaction_id() -> str:
    """Returns a random string"""
    return uuid.uuid4().hex


def split_train_test(data, timestamp, timestamp_finish=None):
    """
    Splits the data into a training and a test sets, using 'timestamp'
    to perform the separation between them

    :param timestamp: A date used for splitting train and test
    :param data: The standardized DataFrame with all transactions
    :param timestamp_finish: The optional ending timestamp for selecting the test set
    :return: The data split in the various sets
    """
    df_train = data[data["Timestamp"] <= timestamp]
    if timestamp_finish is None:
        df_test = data[data["Timestamp"] > timestamp]
    else:
        df_test = data[
            (data["Timestamp"] > timestamp) & (
                data["Timestamp"] <= timestamp_finish)
        ]

    df_columns = data.columns.tolist()
    columns_to_remove = [
        column for column in df_columns if column in data_info.COLUMNS_BONIFICI_LIST
    ]
    df_train = df_train.drop(columns=columns_to_remove)
    df_test = df_test.drop(columns=columns_to_remove)

    y_train = df_train["Fraud"]
    x_train = df_train.drop(["Fraud"], axis=1)
    y_test = df_test["Fraud"]
    x_test = df_test.drop(["Fraud"], axis=1)

    return x_train, y_train, x_test, y_test


# def split_train_test_x_legit(data, timestamp, timestamp_finish=None):
#     df_train = data[data["Timestamp"] <= timestamp]
#     if timestamp_finish is None:
#         df_test = data[data["Timestamp"] > timestamp]
#     else:
#         df_test = data[
#             (data["Timestamp"] > timestamp)
#             & (data["Timestamp"] <= timestamp_finish)
#         ]
#
#     df_columns = data.columns.tolist()
#     columns_to_remove = [
#         column
#         for column in df_columns
#         if column in data_info.COLUMNS_BONIFICI_LIST
#     ]
#     df_train = df_train.drop(columns=columns_to_remove)
#     df_test = df_test.drop(columns=columns_to_remove)
#     df_train.reset_index(drop=True, inplace=True)
#     df_test.reset_index(drop=True, inplace=True)
#
#     y_train = df_train["Fraud"]
#     x_train = df_train.drop(["Fraud"], axis=1)
#     x_train_legit = df_train[df_train["Fraud"] == 0]
#     x_train_legit = x_train_legit.drop(["Fraud"], axis=1)
#     y_test = df_test["Fraud"]
#     x_test = df_test.drop(["Fraud"], axis=1)
#
#     return x_train_legit, x_train, y_train, x_test, y_test


def remove_features(df_trans, features_to_keep):
    """
    Removes columns from a DataFrame, keeping only the wanted ones

    :param df_trans: The DataFrame to modify
    :param features_to_keep: DataFrame columns to keep
    :return:
    """
    features_to_remove = [
        feature
        for feature in df_trans.columns.tolist()
        if feature not in features_to_keep
    ]
    df_trans = df_trans.drop(columns=features_to_remove, errors="ignore")
    return df_trans


def sort_and_reset(df: pd.DataFrame, by="Timestamp"):
    """
    Sorts a DataFrame by UserID and Timestamp, and then resets its indexes
    """
    df = df.sort_values(by=by)
    df = df.reset_index(drop=True)
    return df


def concat_df_list(dataframes) -> pd.DataFrame:
    if len(dataframes) == 0:
        return pd.DataFrame(columns=data_info.COLUMN_AGGREGATED_LIST)
    else:
        df = pd.concat(dataframes)
        df = sort_and_reset(df)
        return df


def remove_transactions(df: pd.DataFrame, to_remove: pd.DataFrame) -> pd.DataFrame:
    """Removes transactions in dataframe b from dataframe a and returns result"""
    transactions_in_b = to_remove.TransactionID
    out_df = df.loc[~df.TransactionID.isin(transactions_in_b)]
    return out_df


def calc_sample_weights(training_data, train_ts) -> List[float]:
    """Calculates sample weights"""
    df_train = training_data[training_data["Timestamp"] < train_ts]
    df_train = df_train.reset_index(drop=True)

    HALF_YEAR_H = 4380

    df_train["Timedelta_diff"] = train_ts - df_train["Timestamp"]
    df_train["diff_days"] = df_train["Timedelta_diff"].dt.days
    df_train["diff_hours"] = df_train["Timedelta_diff"].dt.seconds / 3600
    df_train["difference_h"] = df_train["diff_days"] * \
        24 + df_train["diff_hours"]
    df_train["weight"] = np.exp(-(df_train["difference_h"] / HALF_YEAR_H))
    weights_list = df_train["weight"].tolist()
    return weights_list


def calc_proportional_class_weight(y_train: pd.Series) -> Dict[int, float]:
    """Calculates proportional class weight"""
    frauds_number = y_train[y_train == 1].size
    ratio_frauds_total = frauds_number / y_train.size
    class_weight = {0: ratio_frauds_total, 1: (1 - ratio_frauds_total)}
    return class_weight


def calc_pos_weight(y_train: pd.Series) -> float:
    """Calculates class weight for XGBClassifier"""
    n_neg = y_train[y_train == 0].size
    n_pos = y_train[y_train == 1].size
    pos_weight = n_neg / n_pos
    return pos_weight


# def calc_sample_weights(training_data, _) -> List[float]:
#     weights = training_data[["Amount", "Fraud"]].apply(lambda x: x[0] if x[1]==1 else 100.0, axis=1).tolist()
#     return weights
# 
# def calc_proportional_class_weight(y_train: pd.Series) -> Dict[int, float]:
#     return {0: 1.0, 1: 1.0}
# 
# def calc_pos_weight(y_train: pd.Series) -> float:
#     return 1.0


def load_features_from_file(feature_file: Union[Path, str]) -> List[str]:
    """Loads features from given path."""
    with open(feature_file, "r") as f:
        lines = f.read().splitlines()
    features = [f.strip() for f in lines if f]
    return features


def load_thresholds_from_json(thresholds_file: Path) -> Dict[str, float]:
    """Loads thresholds file"""
    with open(thresholds_file, "r") as f:
        thresholds = json.load(f)
    return thresholds


def make_folder_name(model, scenario, strategy, policy, mitigation):
    """Creates a unique key for the attack configuration"""
    m = "_"+mitigation if mitigation is not None else ""
    folder_name = "%s%s%s%s%s" % (
        scenario[0], strategy[0], policy[0], model[0], m)
    return folder_name
