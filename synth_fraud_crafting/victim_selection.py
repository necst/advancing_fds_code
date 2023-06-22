import json
import os
from dataset import dataset
from config.config import CANDIDATE_VICTIMS_LIST, VICTIM_PROFILES_INFO_FILE


def profile_victims(
    legit_dataset, profile_dict, pre_selected_users, already_selected=[]
):
    """
    Returns the potential victims exhibiting the spending pattern described in 'profile_dict'

    :param legit_dataset: Original DataFrame containing all the legit transactions
    :param profile_dict:
    :param pre_selected_users:
    :param already_selected:
    """

    min_count = profile_dict.get("min_count")
    max_count = profile_dict.get("max_count")
    min_mean = profile_dict.get("min_mean")
    max_mean = profile_dict.get("max_mean")

    # get all users with number of transactions in between min and max
    count_list = legit_dataset["UserID"].value_counts().tolist()
    count_list = [count for count in count_list if min_count <= count <= max_count]
    user_count = (
        legit_dataset["UserID"].value_counts()[: len(count_list)].index.tolist()
    )

    # same but with transaction mean
    grouped_mean = legit_dataset.groupby("UserID").mean()
    grouped_mean = grouped_mean[
        (grouped_mean["Amount"] >= min_mean) & (grouped_mean["Amount"] <= max_mean)
    ]
    user_mean = list(grouped_mean.index.values)
    pre_selected_users = set(pre_selected_users)

    victims = set(user_count + user_mean)
    victims = list(victims.intersection(pre_selected_users))
    victims = [victim for victim in victims if victim not in already_selected]

    return victims


def select_candidates_victims(legit_dataset, year, df_segnal=None):
    """
    Selects fraud victims.
    Candidates are user with at least 3 transactions

    :param legit_dataset: a transaction dataframe
    :return: a dataframe containing all possible victims
    """
    min_trans = 3

    users = legit_dataset.groupby(["UserID"]).size().reset_index(name="Count")

    # select all users with more than 3 transactions as valid users
    valid_users = users.loc[users["Count"] > min_trans]["UserID"].drop_duplicates()

    if df_segnal is not None:
        real_fraud_victims = df_segnal["UserID"].unique().tolist()
        users = legit_dataset.loc[
            (legit_dataset["UserID"].isin(valid_users))
            & (~legit_dataset["UserID"].isin(real_fraud_victims))
        ]["UserID"].drop_duplicates()
    else:
        users = legit_dataset.loc[(legit_dataset["UserID"].isin(valid_users))][
            "UserID"
        ].drop_duplicates()

    users = users.values.tolist()

    with open(VICTIM_PROFILES_INFO_FILE, "r") as victims_profiles:
        victims_prof_dict = json.load(victims_profiles)

    high_profile_params = victims_prof_dict.get("high_profiles")
    medium_profile_params = victims_prof_dict.get("medium_profiles")
    low_profile_params = victims_prof_dict.get("low_profiles")

    high_victims = profile_victims(legit_dataset, high_profile_params, users)
    medium_victims = profile_victims(
        legit_dataset, medium_profile_params, users, high_victims
    )
    low_victims = profile_victims(
        legit_dataset, low_profile_params, users, high_victims + medium_victims
    )

    assert len(set.intersection(set(high_victims), set(medium_victims), set(low_victims))) == 0

    final_victims_data = {
        f"high_vict_{year}": high_victims,
        f"medium_vict_{year}": medium_victims,
        f"low_vict_{year}": low_victims,
    }

    if os.path.exists(CANDIDATE_VICTIMS_LIST):
        with open(CANDIDATE_VICTIMS_LIST, "r") as f:
            file_data = json.load(f)
    else:
        file_data = {}

    file_data.update(final_victims_data)

    with open(CANDIDATE_VICTIMS_LIST, "w") as f:
        json.dump(file_data, f)


if __name__ == "__main__":
    import sys

    USAGE = "USAGE: python3 -m dataset.synth_fraud_crafting.victim_selection"

    df = dataset.load_original("2014_15")
    df_segnal_2014_15 = dataset.load_original("segnalaz_2014_15")
    select_candidates_victims(df, "2014_15", df_segnal_2014_15)
