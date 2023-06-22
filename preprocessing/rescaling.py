import pandas as pd
import os
from preprocessing.const import ATTRIBUTES_TO_NOT_STANDARDIZE
import config.config as c
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import ChainMap
from multiprocessing import Pool
from tqdm import tqdm
from dataset import dataset
from itertools import repeat
from psutil import cpu_count
from typing import List, Tuple, Optional
from joblib import dump, load


def __load_scaler(save_folder: str, name: str) -> StandardScaler:
    """ """

    file_name = name + ".bin"
    scaler_dir = os.path.join(c.STD_SCALER_DIR, save_folder)

    if not os.path.exists(scaler_dir):
        raise Exception("Scaler path does not exist")

    if file_name not in os.listdir(scaler_dir):
        raise Exception("Scaler not found.")

    file_name = os.path.join(scaler_dir, file_name)
    scaler = load(file_name)
    return scaler


def __save_scaler(save_folder: str, scaler: StandardScaler, name: str) -> None:
    """ """
    file_name = name + ".bin"
    scaler_dir = os.path.join(c.STD_SCALER_DIR, save_folder)

    # print("Scaler path created: %s" % scaler_dir)
    os.makedirs(scaler_dir, exist_ok=True)

    file_name = os.path.join(scaler_dir, file_name)
    dump(scaler, file_name, compress=True)


def __get_features_to_std(df: pd.DataFrame) -> List[str]:
    """ """
    return [f for f in df.columns if f not in ATTRIBUTES_TO_NOT_STANDARDIZE]


def __standardize_column(
    data: Tuple[pd.DataFrame, str, bool, Optional[str]]
) -> np.ndarray:
    """
    Standardizes the data (not boolean attributes)
    x_standardized = (x - mean) / std

    :param data: A dataframe with ALL transactions, both legit and frauds
    :return: The standardized data
    """
    df, col, fit, save_folder = data
    x: np.ndarray = df[col].values.reshape(-1, 1)

    if fit:
        scaler = StandardScaler().fit(x)
        if save_folder:
            __save_scaler(save_folder, scaler, col)
    else:
        if not save_folder:
            # scalers need to be loaded from somewhere then
            raise Exception("Save folder must be provided if not fitting data.")
        scaler = __load_scaler(save_folder, col)

    standardized = scaler.transform(x)
    return {col: standardized}


def __manage_null_std(df):
    df = df.fillna(-1)
    return df


def standardize_dataset(
    df: pd.DataFrame,
    fit: bool = False,
    save_folder: str = None,
    disable_tqdm: bool = False,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """ """

    if df.empty:
        return df

    features = __get_features_to_std(df)

    df = __manage_null_std(df)

    if not disable_tqdm:
        print("Standardizing ... (fit: %s, folder: %s)" % (fit, save_folder))

    # we may do not want to use multiprocessing
    use_pool = True  # use_pool

    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs == 0:
        use_pool = False

    if use_pool:
        with Pool(n_jobs) as p:
            std_cols = list(
                tqdm(
                    p.imap_unordered(
                        __standardize_column,
                        zip(
                            map(lambda x: df[x].to_frame(), features),
                            features,
                            repeat(fit),
                            repeat(save_folder),
                        ),
                    ),
                    total=len(features),
                    disable=disable_tqdm,
                )
            )
    else:
        std_cols = list(
            map(
                __standardize_column,
                tqdm(
                    zip(
                        map(lambda x: df[x].to_frame(), features),
                        features,
                        repeat(fit),
                        repeat(save_folder),
                    ),
                    total=len(features),
                    disable=disable_tqdm,
                ),
            )
        )

    new_columns = dict(ChainMap(*std_cols))
    df = df.assign(**new_columns)
    return df


if __name__ == "__main__":
    import sys

    usage = "USAGE: python3 -m preprocessing.rescaling DATASET_NAME..."
    args = sys.argv

    if len(args) < 2:
        print(usage)
        sys.exit(-1)

    datasets = args[1:]

    for dataset_name in datasets:
        df = dataset.load_aggregated(dataset_name)
        df = df.fillna(-1)
        df = standardize_dataset(df, fit=True)
        dataset.save_standardized(dataset_name, df)
