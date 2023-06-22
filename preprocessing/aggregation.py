# import sys
import os
import sys
from config.config import FRAUD_GENERATED_PATH
from utils.utilities import sort_and_reset
import pandas as pd
from dataset import dataset
from multiprocessing import Pool
from psutil import cpu_count
from itertools import repeat, product
from preprocessing.const import (
    AGGREGATED_FEATURES,
    AGGREGATION_FUNCTIONS,
    TIME_FRAMES_LIST,
    STRING_YEAR_H,
    PREPARED_FEATURES,
    RENAME_MAP,
)
from tqdm import tqdm
from typing import Tuple, Dict, Optional
import numpy as np
from collections import ChainMap
from pathlib import Path


# compatibility with old code
features_that_have_new = {
    "CC_ASN": "asn_cc",
    "IBAN": "iban",
    "IBAN_CC": "iban_cc",
    "IP": "ip",
}


# still for compatibility + smol trick for amount key
feature_map = {
    None: "amount",  # Amount has key=None since we do not group df by it
    "CC_ASN": "asn_cc",
    "IBAN": "iban",
    "IBAN_CC": "iban_cc",
    "IP": "ip",
    "IDSessione": "session",
}


# yet again for compatibility
rename_amount_to_global_amount_map = {
    "amount_mean8760h": "global_amount_mean8760h",
    "amount_std8760h": "global_amount_std8760h",
    "amount_sum8760h": "global_amount_sum8760h",
    "amount_count8760h": "global_amount_count8760h",
    "difference_from_amount_mean8760h": "difference_from_global_amount_mean8760h",
}


def __is_new_feature(
    df: pd.DataFrame, f_key: str, new_name: str
) -> pd.DataFrame:
    """Calculates for a given feature and for a given set of user transactions
    if the value of the feature is new or not.
    :param df: user transactions dataframe -> must be sorted by timestamp
    :param f_key: feature key on dataframe
    :param new_name: feature name on dataframe = 'is_new_' + new_name
    """
    # key accesses column in dataframe, value is only used for the new feature
    # name
    new_feature_name = "is_new_" + new_name
    # assign empty column to dataframe
    df = df.assign(**{new_feature_name: 0})
    # get the position of first occurence for each unique element
    pos_of_new_feature = list(
        map(lambda x: df[f_key].eq(x).idxmax(), df[f_key].unique())
    )
    # set True for "is_new_featurename"
    df.loc[pos_of_new_feature, new_feature_name] = 1
    return df


def __time_since_last_group(
    df: pd.DataFrame, new_attr_name: str, attribute: str = None
) -> pd.DataFrame:
    """
    Adds a new attribute that computes how much time (in hours) has passed from
    the previous transaction of a user and the current one, grouped by an
    attribute (e.g.: how much time has passed since the last transaction towards
    the same IBAN)

    :param data: A dataset with all transactions
    :param attribute: An attribute we want to group by (Iban, IP, ASN_CC, etc.)
    :param new_attr_name: The new attribute name
    :return:
    """

    # creates column with NaN values
    df = df.assign(**{new_attr_name: None})

    # When calculating time from global, there is no attribute to group by,
    # so we just create an array with a single value and use the same procedure
    if attribute is None:
        indices = [df.index[1:]]  # get indices shifted by one position
        ts_list_by_attr = [df.Timestamp]  # get list of timestamps
    else:
        grouped_by_attr = df.groupby(attribute)

        # we now get a list of indices  and timestamps for each dataframe
        # obtained by the groupby op
        indices = [x.index[1:] for _, x in grouped_by_attr]
        ts_list_by_attr = [x for _, x in grouped_by_attr.Timestamp]

    # let's zip them together so we get, for each unique attribute:
    # TS_0 TS_1
    # TS_1 TS_2
    # ...
    ts_list_zip = list(map(lambda x: list(zip(x, x[1:])), ts_list_by_attr))

    # Subtracts TS+1 - TS
    ts_diffs_by_attr = [
        list(map(lambda y: y[1] - y[0], x)) for x in ts_list_zip
    ]

    # Rounding operation
    ts_diffs_by_attr = [
        list(map(lambda y: y.total_seconds() / 3600, x))
        for x in ts_diffs_by_attr
    ]

    # Assign to dataframe, remember that the first transaction in timestamp
    # order will always be empty
    for i, j in enumerate(indices):
        df.loc[j, new_attr_name] = ts_diffs_by_attr[i]

    return df


def __compute_aggregate(
    df: pd.DataFrame,
    aggregation: list,
    time_frames: list,
    attributes_group: Optional[str],
    new_attribute_name: str,
) -> pd.DataFrame:
    """Computes aggregated features"""

    time_frames_by_agg_ops = list(product(time_frames, aggregation))

    column_names = list(
        map(
            lambda x: new_attribute_name + "_" + str(x[1]) + str(x[0]),
            time_frames_by_agg_ops,
        )
    )

    if attributes_group is not None:
        f = (
            lambda x: df.groupby(attributes_group)
            .rolling(x[0], on="Timestamp")["Amount"]
            .agg(x[1])
            .droplevel(0)
            .sort_index()
        )
    else:
        f = (
            lambda x: df.rolling(x[0], on="Timestamp")["Amount"]
            .agg(x[1])
            .sort_index()
        )

    rolling_by_time_frame = map(f, time_frames_by_agg_ops)

    # df = df.assign(**{k: v for k, v in zip(
    #     column_names, [None for _ in range(len(column_names))])})

    df = df.assign(
        **{k: v.values for k, v in zip(column_names, rolling_by_time_frame)}
    )

    return df


def __compute_user_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """ """

    # per user:
    # is_new_feature -> for each feature value count if it is the first
    # occurence
    for k, new_name in features_that_have_new.items():
        df = __is_new_feature(df, k, new_name)

    # time_since_last_global -> foreach user counts time since last user
    # transaction
    df = __time_since_last_group(df, "time_from_previous_trans_global")

    # compute actual aggregated features...
    for k2, new_name in feature_map.items():
        time_frames = [STRING_YEAR_H] + TIME_FRAMES_LIST

        df = __compute_aggregate(
            df, AGGREGATION_FUNCTIONS, time_frames, k2, new_name
        )

        if k2 is not None:
            df = __time_since_last_group(df, f"time_since_same_{new_name}", k2)

    return df


def __compute_distance_from_mean(
    data_by_feature: Tuple[pd.DataFrame, str]
) -> Dict[str, np.ndarray]:
    """
    Calculates mean difference from amount.
    """
    data, feature = data_by_feature
    amount_difference = round(data["Amount"] - data[feature], 2)
    return {f"difference_from_{feature}": amount_difference.values}


def __insert_time_into_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Insert an integer encoding for the hour of a transaction in a dataframe.

    :param data: the original dataframe
    :return: the modified dataframe
    """
    h = data["Timestamp"].dt.hour
    m = data["Timestamp"].dt.minute
    s = data["Timestamp"].dt.second
    data["Time"] = h * 3600 + m * 60 + s
    return data


def __cyclic_encode(
    data: pd.DataFrame, feature_name: str, max_value: float
) -> pd.DataFrame:
    """
    Adds two column, respectively sin and cos encoding of a cyclic feature.

    :param max_value: The max value that the feature can have
    :param data: The data we want to apply the encoding to
    :param feature_name: the column name
    :return: the modified dataframe
    """
    data = __insert_time_into_df(data)
    name_x = f"{feature_name}_x"
    name_y = f"{feature_name}_y"
    data[name_y] = np.sin(2 * np.pi * data[feature_name] / max_value)
    data[name_x] = np.cos(2 * np.pi * data[feature_name] / max_value)
    data = data.drop(["Time"], axis=1)

    return data


def __create_bool_attribute(
    data: pd.DataFrame, column_name: str, true_val: str
) -> pd.DataFrame:
    """
    Creates a boolean attribute on column with column_name and assigns True
    where value corresponds to the given true_val.

    :param new_name: The new name to be given to the column
    :param data: A dataframe of transactions
    :param column_name: The column from which take information
    :param og_values: What values needs to be seen in order to change the value
    :return the modified dataframe:
    """
    true_values = data[column_name] == true_val
    data.loc[true_values, column_name] = 1
    data.loc[~true_values, column_name] = 0
    data[column_name] = data[column_name].astype("int")
    return data


def __is_international(data: pd.DataFrame) -> pd.DataFrame:
    """
    Checks whether a transaction is international or not based on CC of IBAN and
    CC of ASN

    :param data: the transaction dataframe
    :return: a modified dataframe with an additional column stating if the
            transaction is international
    """
    temp = data["CC_ASN"].str.split(",", n=1, expand=True)
    data["CC"] = temp[0]
    data["is_international"] = 1
    data.loc[(data["CC"] == data["IBAN_CC"]), "is_international"] = 0
    data = data.drop(["CC"], axis=1)
    return data


def __prepare_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops NaN (at this stage we're dropping only non aggregated features),
    renames columns so there are no misunderstandings everywhere, encodes
    Time_x, Time_y, bool attributes {is_national_iban, confirm_SMS,
    is_international}
    """

    num_of_not_existing_attrs = len(
        set(PREPARED_FEATURES).difference(set(raw_df.columns))
    )

    if num_of_not_existing_attrs == 0:
        return raw_df
    else:
        # DROP ALL NAN!
        raw_df = raw_df.dropna()

        if raw_df.empty:
            return raw_df

        # rename columns if necessary
        raw_df = raw_df.rename(columns=RENAME_MAP)
        raw_df = __create_bool_attribute(raw_df, "confirm_SMS", "Si")
        # cycling encoding of time (instead of hours minutes seconds...)
        raw_df = __cyclic_encode(raw_df, "Time", 86400)
        # set flag
        raw_df = __is_international(raw_df)
        # is_national_iban
        raw_df["is_national_iban"] = raw_df["IBAN_CC"]
        raw_df = __create_bool_attribute(raw_df, "is_national_iban", "IT")
        raw_df = sort_and_reset(raw_df)
        return raw_df


def __get_dataset_with_aggr_features(
    data_grouped_by_user, use_pool, n_jobs, disable_tqdm
):
    if use_pool:
        p = Pool(n_jobs)
        df = list(
            tqdm(
                p.imap_unordered(
                    __compute_user_aggregated_features,
                    [x for _, x in data_grouped_by_user],
                ),
                total=len(data_grouped_by_user),
                disable=disable_tqdm,
                desc="User aggregated features",
            )
        )
        p.close()
        p.join()
    else:
        df = list(
            map(
                __compute_user_aggregated_features,
                tqdm(
                    [x for _, x in data_grouped_by_user],
                    total=len(data_grouped_by_user),
                    disable=disable_tqdm,
                    desc="User aggregated features",
                ),
            )
        )

    df = pd.concat(df)
    df = sort_and_reset(df)
    return df


def __get_dataset_with_distance_from_mean(df, use_pool, n_jobs, disable_tqdm):
    time_frames = [STRING_YEAR_H] + TIME_FRAMES_LIST
    features = list(feature_map.values())
    features_by_tf = list(product(features, time_frames))

    column_names = list(
        map(lambda x: str(x[0]) + "_mean" + str(x[1]), features_by_tf)
    )

    if use_pool:
        p = Pool(n_jobs)
        new_columns = list(
            tqdm(
                p.imap_unordered(
                    __compute_distance_from_mean,
                    zip(repeat(df), column_names),
                ),
                total=len(column_names),
                disable=disable_tqdm,
                desc="Distance from mean",
            )
        )
        p.close()
        p.join()
    else:
        new_columns = list(
            map(
                __compute_distance_from_mean,
                tqdm(
                    zip(repeat(df), column_names),
                    total=len(column_names),
                    disable=disable_tqdm,
                    desc="Distance from mean",
                ),
            )
        )

    new_columns_dict = dict(ChainMap(*new_columns))
    df = df.assign(**new_columns_dict)
    df = sort_and_reset(df)
    return df


def preprocess_dataset(
    df: pd.DataFrame, n_jobs: int = -1, disable_tqdm: bool = False
) -> pd.DataFrame:
    """
    Computes global features and aggregated ones.
    """

    if df.empty:
        return df

    # if dataset has been already preprocessed, then drop the columns
    df = df.drop(columns=AGGREGATED_FEATURES, errors="ignore")

    df["Fraud"] = df["Fraud"].fillna(0)
    columns_where_nan = df.isna().any()

    # if there is any NaN, anywhere, drop it and print warning
    if columns_where_nan.any():
        print(
            "[WARNING] Detected NaN values in dataset before processing."
            " Default action = drop"
        )

        first_len = df.shape[0]
        df = df.dropna()
        last_len = df.shape[0]
        print("[!] Dropped %d NaN" % (first_len - last_len))

    # We always need the dataset to be sorted at least by Timestamp
    df = sort_and_reset(df)

    # if dataset has only one user, then we dont need to open pool
    n_jobs = n_jobs if df.UserID.unique().shape[0] > 1 else 0

    # we may do not want to use multiprocessing
    use_pool = True  # use_pool

    if n_jobs == -1:
        # Open pool with size given by architecture simultaneous threads
        n_jobs = cpu_count()
    elif n_jobs == 0:
        use_pool = False

    all_legit = df.loc[df["Fraud"] == 0]

    last_fraud_per_user = df.loc[df.Fraud == 1].groupby("UserID").tail(1)
    table_join = pd.merge(
        df, last_fraud_per_user, on="UserID", suffixes=("", "_y")
    )
    frauds_with_past_transactions = table_join.loc[
        table_join.Timestamp <= table_join.Timestamp_y
    ][all_legit.columns]

    preprocessed_datasets = []

    for d in [all_legit, frauds_with_past_transactions]:
        # If it does not have the set of "global" features, then calculate them
        # If it has them, then it's not necessary to recalculate them since they're
        # calculated for each transaction
        if d.empty:
            preprocessed_datasets.append(pd.DataFrame([]))
            continue

        prepared = __prepare_dataset(d)
        grouped_by_user = prepared.groupby("UserID")
        w_aggr_features = __get_dataset_with_aggr_features(
            grouped_by_user, use_pool, n_jobs, disable_tqdm
        )
        final = __get_dataset_with_distance_from_mean(
            w_aggr_features, use_pool, n_jobs, disable_tqdm
        )
        preprocessed_datasets.append(final)

    legit_preprocessed, frauds_w_past_preprocessed = preprocessed_datasets

    if not frauds_w_past_preprocessed.empty:
        frauds_preprocessed = frauds_w_past_preprocessed.loc[
            frauds_w_past_preprocessed["Fraud"] == 1
        ]
    else:
        frauds_preprocessed = frauds_w_past_preprocessed

    # more processing steps
    merged = pd.concat([legit_preprocessed, frauds_preprocessed])
    sorted_data = sort_and_reset(merged)
    no_nan_data = sorted_data.fillna(-1)

    # just for compatibility with old code
    final_dataset = no_nan_data.rename(
        columns=rename_amount_to_global_amount_map
    )

    return final_dataset


def preprocess_from_past_raw_data(
    transactions: pd.DataFrame, previous_transactions: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """
    Preprocess transactions given the previous ones
    """
    if previous_transactions.empty:
        return preprocess_dataset(transactions, **kwargs)

    assert previous_transactions.Timestamp.max() < transactions.Timestamp.min()

    t_ids = transactions.TransactionID
    previous_data_same_col = previous_transactions[transactions.columns]
    data = pd.concat([previous_data_same_col, transactions])
    data = preprocess_dataset(data, **kwargs)
    processed_transactions = data.loc[data.TransactionID.isin(t_ids)]
    return processed_transactions


if __name__ == "__main__":
    usage = "usage: python3 -m preprocessing.aggregation [-h] [--resume] DATASET_NAME [FOLDER])"
    help = "Calculates aggregated features for given augmented datasets.\n"
    help += "example: python3 -m preprocessing.aggregation 2014_15 dataset/fraud_generated/campaign_name"
    
    CHUNKS_NUMBER = 1000

    args = sys.argv

    if len(args) < 2:
        print(usage)
        sys.exit(-1)

    if args[1] == "-h":
        print(usage)
        print(help)
        sys.exit(0)

    # if args[1] == "--resume":
    #     dataset_name = args[2]
# 
    #     if len(args) > 3:
    #         dir_aug = args[3]
    #         prefix_aug = args[4]
    #         dir_aggr = dir_aug
    #     else:
    #         dir_aug = FRAUD_GENERATED_PATH
    #         dir_aggr = AGGREGATED_DATA_PATH
# 
    #     tmp_files = [f for f in os.listdir(dir_aggr) if "tmp" in f]
    #     start_n = max([int(f.split("_")[1]) for f in tmp_files])+1
    #     chunks = ["_".join(f.split("_")[:2]) + "_" for f in tmp_files]
    #     all_df = [
    #         dataset.load_aggregated(dataset_name, dir=dir_aggr, prefix=c) for c in chunks
    #     ]
    #     sorted_df = sort_and_reset(pd.concat(all_df))
    #     users_already_proc = sorted_df.UserID.unique()
    #     df = dataset.load_augmented(dataset_name, dir=dir_aug)
    #     users = df.loc[~df.UserID.isin(users_already_proc)]["UserID"].unique()
    #     chunks = None
    #     all_df = None
    #     sorted_df = None
    #     chunks_n = CHUNKS_NUMBER - start_n
    #     print("Restarting from %d, missing %d users." % (start_n, len(users)))
    # else:
    #     dataset_name = args[1]

    dataset_name = args[1]

    dir_aug = args[2]
    pref_aug = args[3]
    dir_aggr = dir_aug
    pref_aggr = args[4]


    df = dataset.load_augmented(dataset_name, dir=dir_aug, prefix=pref_aug)
    users = df["UserID"].unique()
    start_n = 0
    chunks_n = CHUNKS_NUMBER

    # splits user array in evenly sized lists so we have no memory issues
    # on <8gb ram
    # also, datasets are preprocessed using only user information
    user_chunks = np.array_split(users, chunks_n)
    for i, user_list in enumerate(user_chunks, start=start_n):
        df_chunk = df.loc[df.UserID.isin(user_list)]
        prep_chunk = preprocess_dataset(df_chunk, n_jobs=-1)
        # preprocessed_datasets/tmp_N_dataset_name.csv
        dataset.save_aggregated(
            dataset_name, prep_chunk, dir=dir_aggr, prefix=f"tmp_{i}_"
        )

    tmp_files = [f for f in os.listdir(dir_aggr) if "tmp" in f]
    chunks = ["_".join(f.split("_")[:2]) + "_" for f in tmp_files]
    all_df = [
        dataset.load_aggregated(dataset_name, dir=dir_aggr, prefix=c) for c in chunks
    ]
    sorted_df = sort_and_reset(pd.concat(all_df))
    dataset.save_aggregated(dataset_name, sorted_df, dir=dir_aggr, prefix=pref_aggr)
    for f in tmp_files:
        tmp_file_path = Path(dir_aggr) / f
        print("Removing %s" % (tmp_file_path))
        os.remove(tmp_file_path)
