import pandas as pd
import random

import math
import json
from datetime import timedelta

from config import config as c
from .random_attribute_gen import *
from utils.utilities import generate_transaction_id



def select_victims(victim_number, year):
    """
    Selects fraud victims.

    :param victim_number: Number of victims
    :param year: Either '2012_13' or '2014_15'
    :return: A list of all the selected victims
    """
    print(f"Selecting victims for {year}")
    with open(c.VICTIM_PROFILES_INFO_FILE, "r") as vict_profiles_file:
        vict_profiles = json.load(vict_profiles_file)
    with open(c.CANDIDATE_VICTIMS_LIST, "r") as potential_victims_file:
        potential_victims = json.load(potential_victims_file)

    high_numb = round(vict_profiles.get("perc_high") * victim_number)
    medium_numb = round(vict_profiles.get("perc_medium") * victim_number)
    low_numb = round(vict_profiles.get("perc_low") * victim_number)

    high_vict = random.sample(potential_victims.get(f"high_vict_{year}"), high_numb)
    medium_vict = random.sample(
        potential_victims.get(f"medium_vict_{year}"), medium_numb
    )
    low_vict = random.sample(potential_victims.get(f"low_vict_{year}"), low_numb)

    all_victs = high_vict + medium_vict + low_vict
    random.shuffle(all_victs)
    assert len(set(all_victs)) == len(all_victs)
    return all_victs


def pick_amount(strategy, total_amount, fraud_number, min_amount, max_amount):
    """
    Computes an amount to steal, based on the strategy

    :param strategy: 1) rand_uniform: value chosen uniformly between two values
                     2) rand_gauss: value chosen as a gaussian distribution
    :param total_amount: The total value to steal (rand_gauss)
    :param fraud_number: The number of frauds to perform
    :param min_amount: The minimum amount to steal at each transaction
    :param max_amount: The maximum amount to steal at each transaction
    :return: A list containing all the amount to steal
    """

    amount_list = []
    if strategy == "rand_uniform":
        for _ in range(0, fraud_number):
            amount = round(random.uniform(min_amount, max_amount), 2)
            amount_list.append(amount)
    elif strategy == "rand_gauss":
        mean = total_amount / fraud_number
        std = 0.2 * mean
        for _ in range(0, fraud_number):
            amount = round(random.gauss(mean, std), 2)
            if amount > max_amount:
                amount = max_amount + round(random.randint(-200, 200))
            if amount < min_amount:
                amount = min_amount + round(random.randint(-200, 200))
            # amount = amount if amount > 0 else -amount
            assert amount > 0
            amount_list.append(amount)
    else:
        raise Exception("The passed strategy is invalid")
    return amount_list


def generate_timestamp_attack(starting_ts, fraud_number, duration_h, strategy):
    """
    Generates a list of timestamp for a complete attack against a User

    :param starting_ts: The starting time of attack
    :param fraud_number: The number of transactions in the attack
    :param duration_h: The duration (in hour) of the attack
    :param strategy: The attacker strategy
    :return:
    """
    ts_list = []
    delta_h = duration_h / fraud_number
    temp_ts = starting_ts

    for _ in range(0, fraud_number):
        if strategy == "stealthy":
            new_ts = generate_ts_hour_working_hour(temp_ts)
        else:
            new_ts = generate_ts_hour(temp_ts)
        ts_list.append(new_ts)
        minutes = round(random.uniform(0, 59), 4)
        temp_ts = round_dt_to_second(new_ts + timedelta(hours=delta_h, minutes=minutes))

    ts_list.sort()
    return ts_list


def craft_frauds_info_stealing(data, userID, scenario_dict, year):
    """
    Craft frauds against a victim, with information stealing.

    :param data: A dataframe containing all legit transactions
    :param userID: The victim ID
    :param scenario_dict: A dictionary containing all the information about the
    attack scenario
    :param year: 2012_13 or 2014_15
    :return: A List with frauds against a victim
    """
    user_transactions = data.loc[data["UserID"] == userID]
    db_finish_time = data.Timestamp.max()

    # take a random moment for the starting of our attack frauds
    starting_timestamp = generate_random_starting_ts(user_transactions, db_finish_time)

    attacker_iban, attacker_iban_cc = generate_attacker_iban()
    attacker_ip = generate_random_hashed_IP()
    attacker_sessionID = generate_random_SessionID()

    attacker_cc_asn = (
        generate_ASN_CC_2012_13(data)
        if year == "2012_13"
        else generate_ASN_CC_2014_15(data)
    )

    min_frauds = scenario_dict.get("min_frauds", 1)
    max_frauds = scenario_dict.get("max_frauds", 1)
    min_amount = scenario_dict.get("min_amount", 0)
    max_amount = scenario_dict.get("max_amount", 50000)
    total_amount = scenario_dict.get("total_amount", 0)
    duration_h = scenario_dict.get("duration_h", 0)
    strategy = scenario_dict.get("strategy")

    if strategy == "context":
        mean = user_transactions["Amount"].mean()
        std = user_transactions["Amount"].std()
        min_std_coeff = scenario_dict.get("min_std_coeff")
        max_std_coeff = scenario_dict.get("max_std_coeff")
        min_mean_coeff = scenario_dict.get("min_mean_coeff")
        max_mean_coeff = scenario_dict.get("max_mean_coeff")
        count_coeff = scenario_dict.get("count_coeff")
        min_amount = max(mean + min_std_coeff * std, min_mean_coeff * mean)
        max_amount = max(mean + max_std_coeff * std, max_mean_coeff * mean)
        if max_amount > 50000:
            max_amount = 50000
        strategy = "rand_uniform"
        if count_coeff > 0:
            delta_h = timedelta(hours=duration_h)
            starting_count_ts = starting_timestamp - delta_h
            last_period_trans = user_transactions[
                (user_transactions["Timestamp"] <= starting_timestamp)
                & (user_transactions["Timestamp"] >= starting_count_ts)
            ]
            count = last_period_trans.shape[0]
            if count < 2:
                count = 2
            min_frauds = count_coeff * count
            max_frauds = count_coeff * count

    num_frauds = random.randint(min_frauds, max_frauds)

    ts_list = generate_timestamp_attack(
        starting_timestamp, num_frauds, duration_h, strategy
    )

    if strategy == "stealthy":
        strategy = "rand_uniform"

    amount_list = pick_amount(
        strategy, total_amount, num_frauds, min_amount, max_amount
    )

    crafted_list = []

    for iteration in range(num_frauds):
        timestamp = ts_list[iteration]
        amount = amount_list[iteration]
        if timestamp > db_finish_time:
            break
        fraud = pd.DataFrame()
        fraud["TransactionID"] = [generate_transaction_id()]
        fraud["IP"] = [attacker_ip]
        fraud["IDSessione"] = [attacker_sessionID]
        fraud["Timestamp"] = [timestamp]
        fraud["Amount"] = [amount]
        fraud["UserID"] = [userID]
        fraud["IBAN"] = [attacker_iban]
        fraud["confirm_SMS"] = [generate_num_conferma()]
        fraud["IBAN_CC"] = [attacker_iban_cc]
        fraud["CC_ASN"] = [attacker_cc_asn]
        fraud["Fraud"] = [1]
        fraud.reset_index(drop=True)
        crafted_list.append(fraud)

        # Variable updating
        if len(ts_list) > 1 and iteration < num_frauds - 1:
            delta_h = ts_list[iteration + 1] - timestamp
            if delta_h > pd.Timedelta(minutes=30):
                attacker_sessionID = generate_random_SessionID()

        if random.randint(1, 100) > 95:
            attacker_ip = generate_random_hashed_IP()
            attacker_cc_asn = (
                generate_ASN_CC_2012_13(data)
                if year == "2012_13"
                else generate_ASN_CC_2014_15(data)
            )

        if random.randint(1, 100) > 98:
            attacker_iban, _ = generate_attacker_iban()

    return crafted_list


def craft_frauds_trans_hijack(data, userID, scenario_dict):
    """
    Craft frauds against a victim, with transaction hijacking

    :param data: A dataframe containing all legit transactions
    :param userID: The victim ID
    :param scenario_dict: A dictionary containing all the information about the
    attack scenario
    :return: A List with frauds against a victim
    """
    user_transactions = data.loc[data["UserID"] == userID]
    min_victim_transactions = scenario_dict.get("min_victim_transactions", 0)

    if user_transactions.shape[0] < min_victim_transactions:
        return [], []

    half: int = math.floor(len(user_transactions) / 2)
    if half > 0:
        index = random.randint(0, half)

    # IP, IDSession and ASN_CC copied from an existing trans
    fraud_copy = user_transactions.iloc[index].copy()
    starting_ts = fraud_copy["Timestamp"]

    attacker_iban, attacker_iban_cc = generate_attacker_iban()

    crafted_list = []
    to_remove = []

    min_frauds = scenario_dict.get("min_frauds", 1)
    max_frauds = scenario_dict.get("max_frauds", 1)
    min_amount = scenario_dict.get("min_amount", 0)
    max_amount = scenario_dict.get("max_amount", 50000)
    total_amount = scenario_dict.get("total_amount", 0)
    strategy = scenario_dict.get("strategy")
    duration_h = scenario_dict.get("duration_h")

    if duration_h > 0:
        delta = timedelta(hours=duration_h)
        trans_in_next_period = user_transactions[
            (user_transactions["Timestamp"] > starting_ts)
            & (user_transactions["Timestamp"] <= starting_ts + delta)
        ]
        count = trans_in_next_period["Timestamp"].count()
        min_frauds = count
        max_frauds = count

    num_frauds = random.randint(min_frauds, max_frauds)
    num_frauds = 1 if num_frauds == 0 else num_frauds
    amount_list = pick_amount(
        strategy, total_amount, num_frauds, min_amount, max_amount
    )

    for counter in range(min(num_frauds, len(user_transactions) - 1 - index)):
        fraud = pd.DataFrame()
        random_transaction_id = generate_transaction_id()
        fraud["TransactionID"] = [random_transaction_id]
        fraud["IP"] = [fraud_copy["IP"]]
        fraud["IDSessione"] = [fraud_copy["IDSessione"]]
        max_minutes_td = 10
        fraud["Timestamp"] = round_dt_to_second(
            fraud_copy["Timestamp"]
            + timedelta(minutes=round(random.uniform(0, max_minutes_td), 2))
        )

        fraud["Amount"] = [amount_list[counter]]
        fraud["UserID"] = [userID]
        fraud["IBAN"] = [attacker_iban]
        fraud["confirm_SMS"] = [generate_num_conferma()]
        fraud["IBAN_CC"] = [attacker_iban_cc]
        fraud["CC_ASN"] = [fraud_copy["CC_ASN"]]
        fraud["Fraud"] = [1]

        crafted_list.append(fraud)

        if random.randint(0, 100) < 75:
            legit_to_remove = pd.DataFrame()
            legit_to_remove["TransactionID"] = [fraud_copy["TransactionID"]]
            legit_to_remove["IP"] = [fraud_copy["IP"]]
            legit_to_remove["IDSessione"] = [fraud_copy["IDSessione"]]
            legit_to_remove["Timestamp"] = [fraud_copy["Timestamp"]]
            legit_to_remove["Amount"] = [fraud_copy["Amount"]]
            legit_to_remove["UserID"] = [userID]
            legit_to_remove["IBAN"] = [fraud_copy["IBAN"]]
            legit_to_remove["confirm_SMS"] = [fraud_copy["confirm_SMS"]]
            legit_to_remove["IBAN_CC"] = [fraud_copy["IBAN_CC"]]
            legit_to_remove["CC_ASN"] = [fraud_copy["CC_ASN"]]
            legit_to_remove["hijacked_by"] = [random_transaction_id]
            to_remove.append(legit_to_remove)

        index += 1
        fraud_copy = user_transactions.iloc[index].copy()

    return crafted_list, to_remove


def split_percentage(percentage, victim_number, current_victim_index):
    """
    Update the victim index

    :param percentage: The percentage of victims to take
    :param victim_number: The total number of victims
    :param current_victim_index: The current victim index
    :return: The updated victim index
    """
    num_victims_to_take = math.floor(victim_number * percentage / 100)
    new_current_victim_index = current_victim_index + num_victims_to_take
    return new_current_victim_index


def craft_frauds(data, scenario_dict, starting_index, victims_number, victims, year):
    """
    Craft all the frauds of the same strategy

    :param data: Dataset of legit transactions
    :param scenario_dict: Dictionary with parameters for the scenario
    :param starting_index: Starting victim index
    :param victims_number: Number of victims for the scenario
    :param victims: List of all possible victims
    :param year: Either '2012_13' or '2014_15'
    :return:
    """
    fraud_crafted = []

    percentage = scenario_dict.get("percentage")
    new_index = split_percentage(percentage, victims_number, starting_index)
    hijack = scenario_dict.get("hijack", False)
    to_remove = []

    for index in range(starting_index, new_index):
        victimID = victims[index]
        if hijack:
            crafted, temp_remove = craft_frauds_trans_hijack(
                data, victimID, scenario_dict
            )
            fraud_crafted += crafted
            to_remove += temp_remove
        else:
            fraud_crafted += craft_frauds_info_stealing(
                data, victimID, scenario_dict, year
            )
    return new_index, fraud_crafted, to_remove
