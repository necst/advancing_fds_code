
import random
import sys
import pickle
from datetime import datetime
import os

import pandas as pd
import numpy as np

from dataset import dataset
from config.config import CUSTOM_CAMPAIGN_PATH, RESULTS_PATH, EXPERIMENT_2_RESULT_PATH
from utils.utilities import sort_and_reset
from .functions import eval_dataset, calc_loss, train_models
from FDS.FDS import MAX_TS_2014_15


USAGE = "python3 -m experiments.exp_2 [FOLDER]"

# 2014-10-26 23:59:00 - 2014-11-02 23:59:00
# 2014-11-02 23:59:00 - 2014-11-09 23:59:00
# 2014-11-09 23:59:00 - 2014-11-16 23:59:00
# 2014-11-16 23:59:00 - 2014-11-23 23:59:00
# 2014-11-23 23:59:00 - 2014-11-30 23:59:00
# 2014-11-30 23:59:00 - 2014-12-07 23:59:00
# 2014-12-07 23:59:00 - 2014-12-14 23:59:00
# 2014-12-14 23:59:00 - 2014-12-21 23:59:00
# 2014-12-21 23:59:00 - 2014-12-28 23:59:00
# 2014-12-28 23:59:00 - 2015-01-04 23:59:00
# 2015-01-04 23:59:00 - 2015-01-11 23:59:00
# 2015-01-11 23:59:00 - 2015-01-18 23:59:00
# 2015-01-18 23:59:00 - 2015-01-25 23:59:00
# 2015-01-25 23:59:00 - 2015-02-01 23:59:00
# 2015-02-01 23:59:00 - 2015-02-08 23:59:00
# 2015-02-08 23:59:00 - 2015-02-15 23:59:00
# 2015-02-15 23:59:00 - 2015-02-22 23:59:00
# 2015-02-22 23:59:00 - 2015-03-01 23:59:00


def evaluate_campaign_losses(campaign_data, campaign, start, end, trained_models):

    print("testing fraud campaign %s" % campaign)

    data = campaign_data[campaign]
    next_data_in_week = data.loc[(
        data.Timestamp >= start) & (data.Timestamp < end)]

    X_test = next_data_in_week.drop(columns=["Fraud"])
    y_test = next_data_in_week["Fraud"]

    print(start, end)

    # all model evaluations of the new data
    all_predictions = eval_dataset(X_test, trained_models)
    losses = {}
    for m in trained_models.keys():
        losses[m] = calc_loss(
            y_test.to_numpy(), all_predictions[m], X_test["Amount"])

    return losses, all_predictions


if __name__ == "__main__":

    datasets = ["all_lt_is", "all_lt_th", "all_sf_is",
                "all_sf_th", "all_st_is", "all_st_th"]
    models = [
        "logistic_regression",
        "neural_network",
        "random_forest",
        "support_vector_machine",
        "xgboost",
    ]

    outfile = "results.csv"
    folder_name = "adversarial_campaign_%s" % (datetime.now().isoformat())
    folder_name = sys.argv[1] if len(sys.argv) > 1 else folder_name
    output_folder = EXPERIMENT_2_RESULT_PATH / folder_name
    os.makedirs(output_folder, exist_ok=True)

    # list of timestamps of the start of each week
    ts_weeks = [MAX_TS_2014_15 -
                pd.Timedelta(weeks=i) for i in reversed(range(19))]

    start = ts_weeks[6]
    end = ts_weeks[-1]
    # end = ts_weeks[-1]
    tss = []
    t = start
    i = 0
    while t < end:
        t = start + pd.Timedelta(weeks=1*i)
        tss.append(t)
        i+=1

    # generate pairs of week start and week end
    # [(w0_start, w0_end), ..., (wn_start, wn_end)]
    ts_week_start_end = list(zip(tss, tss[1:]))
    from_week_6 = ts_week_start_end

    # from_week_6 = ts_week_start_end[6:]
    # breakpoint()
    # print information
    txt_report = "Adversarial campaign (AGGREGATED SAMPLES)\n"

    # count legitimate transactions in each range
    # print("Counting transactions in ranges")
    # counts = []
    # for start, end in from_week_6:
    #     n = df.loc[(df.Timestamp >= start) & (df.Timestamp < end)].shape[0]
    #     counts.append(n)

    trained_models_pickle = RESULTS_PATH / "trained_models.pkl"
    campaign_data = {d: dataset.load_aggregated(
        "2014_15", CUSTOM_CAMPAIGN_PATH / d, start_ts=from_week_6[0][0]) for d in datasets}

    try:
        with open(trained_models_pickle, "rb") as f:
            trained_models = pickle.load(f)
    except Exception as e:
        print("Exc: %s" % e)
        trained_models = train_models(models)

    with open(trained_models_pickle, "wb") as f:
        pickle.dump(trained_models, f)

    test_ts_start = from_week_6[0][0]

    # for each time window we will save the name of the chosen campaign
    weekly_chosen_campaigns = []
    weekly_loss = []
    weekly_predictions = []
    weekly_best = []

    # random_initial_campaign = "st_th"
    # weekly_chosen_campaigns.append(random_initial_campaign)

    # for the remaining 12 weeks...
    for i, j in enumerate(from_week_6):

        start, end = j
        print(i, start, end)

        # first campaign is chosen randomly
        if i == 0:
            random_initial_campaign = random.choice(datasets)
            weekly_chosen_campaigns.append(random_initial_campaign)

            losses, all_predictions = evaluate_campaign_losses(
                campaign_data, random_initial_campaign, start, end, trained_models)

            final_sum = {k: v[-1] for k, v in losses.items()}
            best_predictions = all_predictions
            best_model = min(final_sum, key=final_sum.get)
            best_loss = final_sum[best_model]
            best = (best_model, best_loss)
        else:
            # we record the loss for each campaign. from this dict we will extract the campaign that
            # increases the most the loss for the best performing fraud detector
            loss_per_campaign = {}
            predictions_per_campaign = {}
            for c in datasets:
                losses, all_predictions = evaluate_campaign_losses(
                    campaign_data, c, start, end, trained_models)
                loss_per_campaign[c] = losses
                predictions_per_campaign[c] = all_predictions

            # for each recorded loss per campaign find the one where the previous
            # best model goes the worst...
            # we save all the cumulated losses in a vector
            prev_best_model_loss = np.array(
                [l_vec[prev_best[0]][-1] for l_vec in loss_per_campaign.values()])
            # subtract previous loss
            # prev_best_model_loss -= prev_best[1]
            # get index of campaign associated to max loss increase
            index_best_campaign = np.argmax(prev_best_model_loss, axis=0)
            best_campaign = datasets[index_best_campaign]
            weekly_chosen_campaigns.append(best_campaign)
            # we now need to re-evaluate all losses and get the best performing model
            # at each iteration this should change
            final_sum = {k: v[-1]
                         for k, v in loss_per_campaign[best_campaign].items()}
            last_week_sum = weekly_loss[-1]
            final_sum = {k: final_sum.get(k, 0) + last_week_sum.get(k, 0) for k in list(final_sum.keys())}
            best_model = min(final_sum, key=final_sum.get)
            best_predictions = predictions_per_campaign[best_campaign]
            best_loss = final_sum[best_model]
            best = (best_model, best_loss)

        weekly_loss.append(final_sum)
        weekly_predictions.append(best_predictions)
        weekly_best.append(best_model)
        print("CURRENT BEST MODEL: %s" % best[0])
        prev_best = best

    dfs = []
    # augment dataset with previously chosen campaigns
    for week_n, campaign in enumerate(weekly_chosen_campaigns):
        data = campaign_data[campaign]
        dfs.append(data.loc[(data.Timestamp >= from_week_6[week_n][0]) & (
            data.Timestamp < from_week_6[week_n][1])])

    current_df = pd.concat(dfs)
    final_dataset = sort_and_reset(current_df)

    dateinfo = datetime.strftime(datetime.now(), "%y%m%d_%H%M%S")
    dataset.save_augmented("2014_15", final_dataset, dir=output_folder)

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

    final_predictions = {k: np.array([]) for k in list(trained_models.keys())}
    for w in weekly_predictions:
        for k,v in w.items():
            final_predictions[k] = np.concatenate((final_predictions[k], v))

    # all_predictions = predictions_per_campaign[weekly_chosen_campaigns[-1]]
    columns = RAW_FEATURES + ["true"] + list(final_predictions.keys())
    X_test = current_df.drop(columns="Fraud")
    y_test = current_df["Fraud"]

    x = np.append(X_test[RAW_FEATURES].values,
                  y_test.values.reshape(-1, 1), axis=1)  # type: ignore
    data = np.append(x, np.array(list(final_predictions.values())).T, axis=1)

    output_df = pd.DataFrame(
        data,
        columns=columns,
    )

    output_df = output_df.astype({"TransactionID": "string"})
    output_df = output_df.astype(
        {k: "int32" for k in ["true"] + list(final_predictions.keys())}
    )

    output_df.to_csv(output_folder / outfile)

    global_frauds_count = current_df.loc[current_df.Fraud == 1].shape[0]
    global_ratio = (global_frauds_count / current_df.shape[0]) * 100

    num_frauds_info = "Frauds Count: %d\nFrauds Ratio:%f\n" % (
        global_frauds_count,
        global_ratio,
    )

    txt_report += num_frauds_info

    data = {
        "start": [],
        "end": [],
        "fraud_type": [],
        "n_frauds": [],
        "n_transactions": [],
        "ratio_frauds": [],
        "best": [],
    }

    for model in trained_models.keys():
        data.update({"loss_"+model: []})

    for i, week in enumerate(from_week_6):
        start, end = week
        c = weekly_chosen_campaigns[i]
        losses = weekly_loss[i]
        best_model = weekly_best[i]
        count = current_df.loc[
            (current_df.Timestamp >= start) & (
                current_df.Timestamp < end)
        ].shape[0]
        count_frauds = current_df.loc[
            (current_df.Fraud == 1) &
            (current_df.Timestamp >= start) &
            (current_df.Timestamp < end)
        ].shape[0]
        perc_frauds = round(count_frauds / count * 100, 4) if count > 0 else 0
        data["start"].append(start)
        data["end"].append(end)
        data["n_frauds"].append(count_frauds)
        data["n_transactions"].append(count)
        data["ratio_frauds"].append(perc_frauds)
        data["fraud_type"].append(c)
        data["best"].append(best_model)
        for model, cum_loss in weekly_loss[i].items():
            data["loss_"+model].append(cum_loss)

    info_data = pd.DataFrame(data)
    info_data.to_csv(output_folder / "info.csv")

    txt_report += info_data.to_string()
    print(txt_report)

    with open(output_folder / "info.txt", "w") as f:
        f.write(txt_report)
