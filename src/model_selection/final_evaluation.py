import sys, os
import numpy as np
import pandas as pd
import uuid
from pathlib import Path

from FDS import UpdatePolicy, Model, FDSDataset
from FDS.FDS import FraudDetectionSystem
from utils import evaluation_functions
from config.config import RESULTS_PATH
from dataset import dataset
from preprocessing.rescaling import standardize_dataset
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns


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


SIX_WEEK_TS = pd.Timestamp(year=2014, month=12, day=7, hour=23, minute=59)


if __name__ == "__main__":

    model_name = sys.argv[1]
    model = Model(model_name)
    feature_dir = sys.argv[2]
    hparams_file = sys.argv[3]
    thresholds_file = sys.argv[4]
    dataset_train = sys.argv[5]
    dataset_test = sys.argv[6]

    calc_class_weight = model_name == "neural_network"
    ts_split = SIX_WEEK_TS
    train_dataset_name = Path(dataset_train).parts[-1]
    test_dataset_name = Path(dataset_test).parts[-1]

    results_dir = RESULTS_PATH / "model_selection" / train_dataset_name
    timings_txt = results_dir / "timings.txt"

    df_train = dataset.load_aggregated("2014_15", dir=dataset_train)  # type: ignore
    df_test = dataset.load_aggregated("2014_15", dir=dataset_test)  # type: ignore

    outfile = results_dir / ("%s_final_eval.csv" % model_name)

    df_train = df_train[df_train["Timestamp"] <= ts_split]
    df_test = df_test[df_test["Timestamp"] > ts_split]

    final_results = {}

    cache_dir = uuid.uuid4().hex[:6] + model_name
    fit_ts = ts_split + pd.Timedelta(minutes=1)

    fds = FraudDetectionSystem(                
        model_name,
        UpdatePolicy.TWO_WEEKS,
        "",
        thresholds_file=thresholds_file,
        hyperparam_file=hparams_file,
        feature_dir=Path(feature_dir),
        scaler_cache_dir=cache_dir,
    )
    print(fds.threshold)
    fds.fit(FDSDataset(None, df_train, None), fit_ts)
    y_test = df_test["Fraud"]
    df_test = df_test[fds.features]
    df_test_std = standardize_dataset(
        df_test,
        fit=False,
        save_folder=cache_dir,
        disable_tqdm=True,
        n_jobs=-1,
    )

    predictions, probs = fds.predict(df_test_std)

    final_score = evaluation_functions.metric_score(y_test, predictions)
    breakpoint()
    fpr, tpr, _ = metrics.roc_curve(np.array(y_test).astype(bool), probs)
    roc_auc = metrics.auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.4f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    fig.savefig(results_dir / ("%s_%s_roc.png" % (model_name, test_dataset_name)))

    # computes and plot PRC
    precision, recall, _ = metrics.precision_recall_curve(
        np.array(y_test).astype(bool), probs, 
    )
    no_skill = sum(y_test) / len(y_test)
    prc_auc = metrics.auc(recall, precision)
    fig = plt.figure()
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label="PRC curve (area = %0.4f)" % prc_auc,
    )
    plt.plot([0, 1], [no_skill, no_skill], color="navy", lw=2, linestyle="--")
    plt.plot([0.0125, 0.0125], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower right")
    fig.savefig(results_dir / ("%s_%s_prc.png" % (model_name, test_dataset_name)))

    # computes and plot confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, predictions)
    fig = plt.figure()
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Pastel2",
        linecolor="white",
        linewidths=0.5,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.savefig(results_dir / ("%s_%s_conf_matrix.png" % (model_name, test_dataset_name)))

    TN, FP, FN, TP = conf_matrix.ravel()
    total_trans = TP + TN + FN + FP
    frauds = TP + FN
    legit = TN + FP
    ratio = frauds / legit

    matthews_correlation = metrics.matthews_corrcoef(y_test, predictions)

    c_matthews_corr = metrics.matthews_corrcoef(
        y_test, predictions, sample_weight=compute_matthews_weights(y_test)
    )

    report = metrics.classification_report(y_test, predictions, output_dict=True)

    precision, recall, f1_score, support = tuple(report['1'].values())

    final_results = {
        "c_accuracy" : [final_score],
        "w_matthews_corr": [c_matthews_corr],
        "matthews_corr": [matthews_correlation],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1_score],
        "support": [support],
        "tp": [TP],
        "fp": [FP],
        "tn": [TN],
        "fn": [FN],
        "ratio": [ratio],
        "test_dataset_name": [test_dataset_name],
        "train_dataset_name": [train_dataset_name],
        "feature_dir": [feature_dir],
        "hparams_file": [hparams_file],
        "thresholds_file": [thresholds_file],
    }

    results_df = pd.DataFrame(final_results)
    put_header = not os.path.exists(outfile)
    results_df.to_csv(outfile, mode="a", header=put_header)
    print("results at %s" % outfile)