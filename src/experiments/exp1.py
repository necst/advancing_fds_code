import uuid
import sys
from datetime import datetime

import pandas as pd
import numpy as np

from config.config import EXPERIMENT_1_RESULT_PATH, RESULTS_PATH, CUSTOM_CAMPAIGN_PATH, FEATURES_DIR, HPARAMS_DIR
from dataset import dataset
from preprocessing.rescaling import standardize_dataset
import pickle
from .functions import eval_dataset

SIX_WEEK_TS = pd.Timestamp(year=2014, month=12, day=7, hour=23, minute=59, second=0)


USAGE = "python3 -m experiments.exp_1 [FOLDER]"


if __name__ == "__main__":

    output_df_name = sys.argv[1] if len(sys.argv) > 1 else "%s_results.csv" % (datetime.now().isoformat())
    outfile = EXPERIMENT_1_RESULT_PATH / output_df_name

    datasets = ["lt_is", "lt_th", "sf_is", "sf_th", "st_is", "st_th"]
    models = [
        "logistic_regression",
        "neural_network",
        "random_forest",
        "support_vector_machine",
        "xgboost",
    ]

    df_test_aggr = dataset.load_aggregated(
        "2014_15", CUSTOM_CAMPAIGN_PATH / "secai_test", start_ts=SIX_WEEK_TS
    )

    X_test = df_test_aggr.drop(columns=["Fraud"])
    y_test = df_test_aggr["Fraud"]

    fit_ts = SIX_WEEK_TS + pd.Timedelta(minutes=1)
    all_predictions = {}

    trained_models_pickle = RESULTS_PATH / "trained_models.pkl"

    # we generate a random string so that consecutive tests do not clash...
    # i dont remember all of the parts of the library. probably there is some
    # line of code that deletes the folders but whatever
    try:
        with open(trained_models_pickle, "rb") as f:
            trained_models = pickle.load(f)
    except Exception as e:
        print("Exc: %s" % e)
        trained_models = {}

    all_predictions = eval_dataset(X_test, trained_models)

    RAW_FEATURES = [
        "TransactionID",
        "IP",
        "IDSessione",
        "Timestamp",
        "Amount",
        "UserID",
        "IBAN",
        "confirm_SMS",
        "IBAN_CC",
        "CC_ASN",
    ]
    columns = RAW_FEATURES + ["true"] + list(all_predictions.keys())

    x = np.append(X_test[RAW_FEATURES].values, y_test.values.reshape(-1, 1), axis=1)  # type: ignore
    data = np.append(x, np.array(list(all_predictions.values())).T, axis=1)

    output_df = pd.DataFrame(
        data,
        columns=columns,
    )

    output_df = output_df.astype({"TransactionID": "string"})
    output_df = output_df.astype(
        {k: "int32" for k in ["true"] + list(all_predictions.keys())}
    )
    
    output_df.to_csv(outfile)
