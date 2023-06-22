from typing import List, Optional, Union, Dict
from pathlib import Path
from config import config as c
from utils.utilities import sort_and_reset

import pandas as pd
import dask.dataframe as dd
import os
from enum import Enum
import gc


TS_SPLIT_2014_15 = pd.Timestamp(year=2015, month=1, day=5)


class Dataset(Enum):
    """List of all available datasets"""
    DATASET_2014_15 = "2014_15"


class DatasetNotFound(Exception):
    """Raised when dataset cannot be found in the selected directory."""
    pass


def load_dataset(
    dir: Union[Path, str],
    dataset: Union[Dataset, str],
    prefix: str = "",
    sort_reset: bool = True,
    start_ts=None,
    end_ts=None,
    usecols=None,
    users: Optional[List[str]] = None,
    target_column: str = "Fraud",
    timestamp_column: str = "Timestamp",
    load_with_target_and_ts: bool = True,
    quiet=False,
) -> pd.DataFrame:
    """
    Dataset loading function. Reads a csv file and outputs a pandas DataFrame.
    **Don't use this one! Use load_original, load_augmented, etc.**

    :param dir: directory of the dataset
    :type dir: Union[Path, str]
    :param dataset: Name or constant associated to name
    :type dataset: Union[Dataset, str]
    :param prefix: prefix of the file, optional
    :type prefix: str
    :param sort_reset: sort by timestamp and reset indices of csv file
    :type sort_reset: bool
    :param start_ts: get only data after given timestamp
    :type start_ts: pandas.Timestamp
    :param end_ts: get only data before given timestamp
    :type end_ts: pandas.Timestamp
    :param usecols: filter columns
    :type usecols: List[str]
    :param users: filter by user id
    :type users: Optional[List[str]]
    :param get_hash: return hash with dataset
    :type get_hash: bool
    :param target_column: column of class label
    :type target_column: str
    :param timestamp_column: column with timestamp
    :type timestamp_column: str
    :param load_with_target_and_ts: load dataset with target column and timestamp
    :type load_with_target_and_ts: bool
    :return: Dataset or dataset and hash
    :rtype: Union[pd.DataFrame, Tuple[pd.DataFrame, str]]
    """
    dir = Path(dir)
    files = os.listdir(dir)

    dataset = Dataset(dataset)

    dataset_name = prefix + dataset.value + ".csv"

    if dataset_name in files:
        df_file = dir / dataset_name

        if not quiet:
            print("LOAD: %s" % df_file)

        if usecols is not None and load_with_target_and_ts:
            usecols = list(usecols)
            usecols += [target_column, timestamp_column]
            usecols = list(set(usecols))

        df = dd.read_csv(df_file, usecols=usecols, dtype={"TransactionID": "object"})  # type: ignore
        df = df.compute()

        try:
            df = df.drop(columns=["Unnamed: 0"])
        except Exception:
            pass

        df.Timestamp = pd.to_datetime(df.Timestamp, dayfirst=True)

        if users is not None:
            df = df[df.UserID.isin(users)]

        if start_ts is not None:
            df = df[df[timestamp_column] >= start_ts]

        if end_ts is not None:
            df = df[df[timestamp_column] < end_ts]

        if sort_reset:
            df = sort_and_reset(df)

        return df
    else:
        raise DatasetNotFound("Dataset %s not found in %s" % (dataset_name, dir))


def save_dataset(
    dir: Union[Path, str],
    dataset: Union[Dataset, str],
    df: pd.DataFrame,
    prefix: str = "",
    sort_reset: bool = True,
    quiet=False,
) -> None:
    """
    Saves pandas dataframe into a csv file.

    :param dir: Directory where to save dataset file
    :type dir: Union[Path, str]
    :param dataset: Name of the dataset
    :type dataset: Union[Dataset, str]
    :param df: Actual dataset
    :type df: pd.DataFrame
    :param prefix: Prefix of name of dataset
    :type prefix: str
    :param sort_reset: Sort and reset before saving
    :type sort_reset: bool
    """
    dataset = Dataset(dataset)
    dir = Path(dir)
    os.makedirs(dir, exist_ok=True)

    if sort_reset:
        df = sort_and_reset(df)

    dataset_name = prefix + dataset.value + ".csv"

    df_file = dir / dataset_name

    if not quiet:
        print("SAVE: %s" % df_file)
    df.to_csv(df_file)


def load_original(
    dataset: Union[Dataset, str], dir=c.ORIGINAL_DATA_PATH, with_fraud_column=True, **kwargs
) -> pd.DataFrame:
    df = load_dataset(dir, dataset, **kwargs)
    if with_fraud_column:
        df["Fraud"] = 0
    return df


def load_synthetic_frauds(
    dataset: Union[Dataset, str], dir=c.FRAUD_GENERATED_PATH, **kwargs
) -> pd.DataFrame:
    df = load_dataset(
        dir,
        dataset,
        c.SYNTHETIC_FRAUDS_DATA_PREFIX,
        **kwargs,
    )
    return df


def load_augmented(
    dataset: Union[Dataset, str], dir=c.TRAINING_DATASET_PATH, prefix=c.AUGMENTED_DATA_PREFIX, **kwargs
) -> pd.DataFrame:
    df = load_dataset(dir, dataset, prefix=prefix, **kwargs)
    return df


def load_removed_legit_transactions(
    dataset: Union[Dataset, str], dir=c.FRAUD_GENERATED_PATH, **kwargs
) -> pd.DataFrame:
    df = load_dataset(dir, dataset, c.LEGIT_REMOVED_DATA_PREFIX, **kwargs)
    return df


def load_aggregated(
    dataset: Union[Dataset, str], dir=c.TRAINING_DATASET_PATH, prefix=c.AGGREGATED_DATA_PREFIX, **kwargs
) -> pd.DataFrame:
    df = load_dataset(dir, dataset, prefix=c.AGGREGATED_DATA_PREFIX, **kwargs)
    return df


def save_original(
    dataset: Union[Dataset, str], df: pd.DataFrame, dir=c.ORIGINAL_DATA_PATH, **kwargs
) -> None:
    save_dataset(dir, dataset, df, **kwargs)


def save_synthetic_frauds(
    dataset: Union[Dataset, str],
    df: pd.DataFrame,
    dir=c.FRAUD_GENERATED_PATH,
    **kwargs,
) -> None:
    save_dataset(
        dir,
        dataset,
        df,
        c.SYNTHETIC_FRAUDS_DATA_PREFIX,
        **kwargs,
    )


def save_augmented(
    dataset: Union[Dataset, str], df: pd.DataFrame, dir=c.TRAINING_DATASET_PATH, prefix=c.AUGMENTED_DATA_PREFIX, **kwargs
) -> None:
    save_dataset(dir, dataset, df, prefix=prefix, **kwargs)


def save_removed_legit_transactions(
    dataset: Union[Dataset, str],
    df: pd.DataFrame,
    dir=c.FRAUD_GENERATED_PATH,
    **kwargs,
) -> None:
    save_dataset(
        dir,
        dataset,
        df,
        c.LEGIT_REMOVED_DATA_PREFIX,
        **kwargs,
    )


def save_aggregated(
    dataset: Union[Dataset, str], df: pd.DataFrame, dir=c.TRAINING_DATASET_PATH, **kwargs
) -> None:
    save_dataset(dir, dataset, df, **kwargs)
