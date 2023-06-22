import uuid

import pandas as pd
import numpy as np

from utils.utilities import sort_and_reset
from .loss import single_loss
from FDS import FraudDetectionSystem, UpdatePolicy, FDSDataset
from config.config import CUSTOM_CAMPAIGN_PATH, FEATURES_DIR, HPARAMS_DIR, THRESHOLDS_DIR
from dataset import dataset

from preprocessing.rescaling import standardize_dataset
from preprocessing.aggregation import preprocess_dataset

SIX_WEEK_TS = pd.Timestamp(year=2014, month=12, day=7,
                           hour=23, minute=59, second=0)


def train_models(models):
    trained_models = {}
    fit_ts = SIX_WEEK_TS + pd.Timedelta(minutes=1)
    cache_rand = uuid.uuid4().hex[:6]
    train_datasets = ["lt_is", "lt_th", "st_is", "st_th", "sf_is", "sf_th"]
    for campaign in train_datasets:
        dataset_folder = CUSTOM_CAMPAIGN_PATH / ("%s_train" % campaign)
        df_train = dataset.load_aggregated(
            "2014_15", dir=dataset_folder, end_ts=SIX_WEEK_TS
        )

        # X_train = df_train.drop(columns=["Fraud"])
        # X_test = df_test.drop(columns=["Fraud"])
        # y_train = df_train["Fraud"]
        # y_test = df_test["Fraud"]

        for m in models:
            hparam_file = (
                HPARAMS_DIR / ("%s.json" % campaign)
            )  # dict with models as keys and hparams as values
            thresholds_file = (
                THRESHOLDS_DIR / campaign / "thresholds.json"
            )
            feature_dir = FEATURES_DIR / campaign

            print("fitting %s model with %s and %s" %
                  (m, hparam_file, feature_dir))

            fds_name = "%s_%s" % (m, campaign)
            cache_dir = cache_rand + fds_name

            fds = FraudDetectionSystem(
                m,
                UpdatePolicy.TWO_WEEKS,
                "",
                thresholds_file=thresholds_file,
                hyperparam_file=hparam_file,
                feature_dir=feature_dir,
                scaler_cache_dir=cache_dir,
            )
            fds.fit(FDSDataset(None, df_train, None), fit_ts)  # type: ignore
            trained_models[fds_name] = fds
    return trained_models


def eval_dataset(X_test, models):
    all_predictions = {}
    for fds_name, fds in models.items():
        df_test_ft = X_test[fds.features]  # type: ignore

        df_test_std = standardize_dataset(
            df_test_ft,
            fit=False,
            save_folder=fds.scaler_cache_dir,
            disable_tqdm=True,
            n_jobs=-1,
        )
        predictions, _ = fds.predict(df_test_std)
        all_predictions[fds_name] = predictions
    return all_predictions


def calc_loss(y_true, y_pred, amounts):
    """"""
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    amounts = np.array(amounts).reshape(-1, 1)

    M = np.hstack([y_true, y_pred, amounts])
    loss_vector = np.apply_along_axis(
        lambda x: single_loss(x[0], x[1], x[2]), 1, M)
    return np.cumsum(loss_vector)


def augment_dataset(original_df_part, frauds_in_range, to_remove_in_range):
    # remove hijacked transactions in campaign range
    initial_shape = original_df_part.shape[0]
    original_df_part = original_df_part.loc[~original_df_part.TransactionID.isin(
        to_remove_in_range.TransactionID)]
    after_remove_shape = original_df_part.shape[0]
    assert after_remove_shape == initial_shape - \
        to_remove_in_range.shape[0]

    dataset_part = pd.concat([original_df_part, frauds_in_range])
    # assert final_dataset.shape[0] > initial_shape
    dataset_part = sort_and_reset(dataset_part)
    return dataset_part


def augment_dataset_chosen_campaigns(original_df, campaign_data, weekly_campaigns, i):
    """"""
    augmented_df = original_df
    for i, campaign in enumerate(weekly_campaigns):
        frauds_part_df = campaign_data[campaign]["frauds"][i]
        to_remove_part_df = campaign_data[campaign]["to_remove"][i]
        augmented_df = augment_dataset(
            augmented_df, frauds_part_df, to_remove_part_df)
    return augmented_df

