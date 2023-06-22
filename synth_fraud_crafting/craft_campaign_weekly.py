import pandas as pd
from FDS.FDS import MAX_TS_2014_15
from dataset import dataset
from config.config import FRAUD_GENERATED_PATH
from utils.utilities import sort_and_reset
import sys
from math import floor

USAGE = "python3 -m synth_fraud_crafting.craft_temporal_mix CAMPAIGN_GEN_STR [OUTPUT_FOLDER] [--up-to-week6]"

CAMPAIGNS = {
    "s": "short_term_info_stealing",
    "S": "short_term_transaction_hijacking",
    "l": "long_term_info_stealing",
    "L": "long_term_transaction_hijacking",
    "f": "single_fraud_info_stealing",
    "F": "single_fraud_transaction_hijacking",
    "0": "no_frauds",
}

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


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(USAGE)

    xs = sys.argv[1]
    folder_name = xs if len(sys.argv) < 3 else sys.argv[2]

    if folder_name in CAMPAIGNS.values():
        print("Name not available!")

    if len(xs) == 1 and xs != "0":
        xs = xs * 18

    if len(xs) < 18:
        xs = xs.ljust(18, xs[-1])

    if len(xs) > 18:
        xs = xs[:18]

    for x in xs:
        if x not in CAMPAIGNS.keys():
            print("ALLOWED CHARS:")
            print(CAMPAIGNS)
            sys.exit(-1)

    # list of timestamps of the start of each week
    ts_weeks = [MAX_TS_2014_15 - pd.Timedelta(weeks=i) for i in reversed(range(19))]

    # generate pairs of week start and week end
    # [(w0_start, w0_end), ..., (wn_start, wn_end)]
    ts_week_start_end = list(zip(ts_weeks, ts_weeks[1:]))

    # for ts in ts_week_start_end:
    #     print(ts)

    # get campaign timestamp boundaries, merging adjacent weeks
    # (c0, (c0start, c0end), ... ())
    # breakpoint()

    campaigns_per_week = [
        (CAMPAIGNS[xs[i]], x) for i, x in enumerate(ts_week_start_end)
    ]

    # i = 0
    # j = 0
    # while i < len(xs):
    #     c = CAMPAIGNS[xs[i]]
    #     start, end = ts_week_start_end[i]
    #     j = i + 1
    #     while j < len(xs):
    #         if CAMPAIGNS[xs[j]] != c:
    #             break
    #         j += 1
    #     i = j
    #     end = ts_week_start_end[j - 1][1]
    #     campaigns_per_week.append((c, (start, end)))

    # print information
    txt_report = "custom fraud campaign\n"
    txt_report += "input string: %s\n" % xs

    # load original dataset with no frauds
    df = dataset.load_original("2014_15")
    df["Fraud"] = 0

    selected_frauds = []
    transactions_to_remove = []

    # for each campaign
    for c in CAMPAIGNS.values():

        if c == "no_frauds":
            continue

        print("Getting frauds from campaign %s" % c)
        # get ranges of each campaign
        ranges = [r for cc, r in campaigns_per_week if cc == c]

        # count legitimate transactions in each range
        print("Counting transaction in ranges")
        counts = []
        for start, end in ranges:
            n = df.loc[(df.Timestamp >= start) & (df.Timestamp < end)].shape[0]
            counts.append(n)

        synth_fraud_folder = FRAUD_GENERATED_PATH / c

        # load synthetic frauds associated to campaign
        fraud_df = dataset.load_synthetic_frauds("2014_15", dir=synth_fraud_folder)

        # load hijacked original transactions to be removed from dataset
        to_remove_df = dataset.load_removed_legit_transactions(
            "2014_15", dir=synth_fraud_folder
        )

        # for each range, count of original transactions
        for r, count in zip(ranges, counts):
            print("Locating transactions in range %s" % str(r))
            start, end = r
            print(c, start, end)
            # locate synthetic frauds in time interval
            t_in_range = fraud_df.loc[
                (fraud_df.Timestamp >= start) & (fraud_df.Timestamp < end)
            ]

            # locate hijacked transactions in time interval
            to_remove_in_range = to_remove_df.loc[
                (to_remove_df.Timestamp >= start) & (to_remove_df.Timestamp < end)
            ]

            # number of frauds in range
            n_frauds_in_range = t_in_range.shape[0]

            # max number of frauds to sample
            one_perc = floor(count / 100)

            # if less than target for one percent, load all
            t_to_sample = (
                n_frauds_in_range if n_frauds_in_range < one_perc else one_perc
            )

            # sample frauds
            print("Sampling %d frauds in range %s" % (t_to_sample, r))

            frauds_in_range = t_in_range.sample(t_to_sample)
            assert frauds_in_range.Timestamp.min() >= start
            assert frauds_in_range.Timestamp.max() < end

            # append in selection
            selected_frauds.append(frauds_in_range)

            to_remove_in_range = to_remove_in_range.loc[
                to_remove_in_range.hijacked_by.isin(t_in_range.TransactionID)
            ]

            # append list of transaction ids to remove
            transactions_to_remove.append(to_remove_in_range)

    transactions_to_remove_df = sort_and_reset(pd.concat(transactions_to_remove))
    final_frauds_df = sort_and_reset(pd.concat(selected_frauds))

    # remove hijacked transactions in campaign range
    initial_shape = df.shape[0]
    df = df.loc[~df.TransactionID.isin(transactions_to_remove_df.TransactionID)]
    after_remove_shape = df.shape[0]
    assert after_remove_shape == initial_shape - transactions_to_remove_df.shape[0]
    final_dataset = pd.concat([df, final_frauds_df])
    # assert final_dataset.shape[0] > initial_shape
    final_dataset = sort_and_reset(final_dataset)
    output_folder = FRAUD_GENERATED_PATH / folder_name

    dataset.save_augmented("2014_15", final_dataset, dir=output_folder)
    dataset.save_synthetic_frauds("2014_15", final_frauds_df, dir=output_folder)
    dataset.save_removed_legit_transactions(
        "2014_15", transactions_to_remove_df, dir=output_folder
    )

    global_frauds_count = final_dataset.loc[final_dataset.Fraud == 1].shape[0]
    global_ratio = (global_frauds_count / final_dataset.shape[0]) * 100

    data = {
        "start": [],
        "end": [],
        "fraud_type": [],
        "n_frauds": [],
        "n_transactions": [],
        "ratio_frauds": [],
    }

    num_frauds_info = "Frauds Count: %d\nFrauds Ratio:%f\n" % (
        global_frauds_count,
        global_ratio,
    )
    txt_report += num_frauds_info

    for c, week in campaigns_per_week:
        start, end = week
        count = final_dataset.loc[
            (final_dataset.Timestamp >= start) & (final_dataset.Timestamp < end)
        ].shape[0]
        count_frauds = final_frauds_df.loc[
            (final_frauds_df.Timestamp >= start) & (final_frauds_df.Timestamp < end)
        ].shape[0]
        perc_frauds = round(count_frauds / count * 100, 4) if count > 0 else 0
        data["start"].append(start)
        data["end"].append(end)
        data["n_frauds"].append(count_frauds)
        data["n_transactions"].append(count)
        data["ratio_frauds"].append(perc_frauds)
        data["fraud_type"].append(c)

    info_data = pd.DataFrame(data)

    txt_report += info_data.to_string()
    print(txt_report)

    with open(output_folder / "info.txt", "w") as f:
        f.write(txt_report)
