import pandas as pd
import numpy as np

from config.config import RESULTS_PATH, CUSTOM_CAMPAIGN_PATH
from dataset import dataset
import pickle
from .functions import eval_dataset
from sklearn.metrics import f1_score


SIX_WEEK_TS = pd.Timestamp(year=2014, month=12, day=7, hour=23, minute=59, second=0)

USAGE = "python3 -m experiments.performance_table"

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

if __name__ == "__main__":

    
    outfile_dir = RESULTS_PATH / "final_perf_eval"
    datasets = ["all_lt_is", "all_lt_th", "all_sf_is", "all_sf_th", "all_st_is", "all_st_th"]
    models = [
        "logistic_regression",
        "neural_network",
        "random_forest",
        "support_vector_machine",
        "xgboost",
    ]

    campaign_data = {d: dataset.load_aggregated(
        "2014_15", CUSTOM_CAMPAIGN_PATH / d, start_ts=SIX_WEEK_TS) for d in datasets}
    
    fit_ts = SIX_WEEK_TS + pd.Timedelta(minutes=1)
    all_predictions = {}

    trained_models_pickle = RESULTS_PATH / "trained_models.pkl"
    
    try:
        with open(trained_models_pickle, "rb") as f:
            trained_models = pickle.load(f)
    except Exception as e:
        print("Exc: %s" % e)
        trained_models = {}
    
    f1scores = {
        "model": [],
        "campaign": [],
        "score": []
    }

    for campaign in campaign_data:
        df_test_aggr = campaign_data[campaign]
        X_test = df_test_aggr.drop(columns=["Fraud"])
        y_test = df_test_aggr["Fraud"]

        all_predictions = eval_dataset(X_test, trained_models)

        for m in all_predictions:
            f1scores["model"].append(m)
            f1scores["campaign"].append(campaign)
            f1scores["score"].append(f1_score(y_test, all_predictions[m]))

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
        
        output_df.to_csv(outfile_dir / (campaign + ".csv"))

    f1scores_df = pd.DataFrame(f1scores)
    f1scores_df.to_csv(outfile_dir / "scores.csv")