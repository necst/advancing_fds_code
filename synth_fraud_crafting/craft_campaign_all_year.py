import json
import os
import pandas as pd

from dataset import dataset
from config.config import FRAUD_GENERATED_PATH, SYNTH_FRAUDS_REPORT

from .campaign_generator import (
    select_victims,
    craft_frauds,
)
from utils.utilities import sort_and_reset
import sys
from datetime import datetime
from pathlib import Path


USAGE = "USAGE: python3 -m synth_fraud_crafting.craft DATASET STRATEGY_PATH"


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(-1)

    dataset_name = sys.argv[1]
    strategy_file = sys.argv[2]

    df_legit = dataset.load_original(dataset_name)

    print(f"CRAFTING FRAUDS FOR {dataset_name}")
    df_legit = df_legit.sort_values(by=["UserID", "Timestamp"])

    print("Reading crafting parameters...")
    with open(strategy_file, "r") as param_file:
        params = json.load(param_file)

    victims_number = params.get(f"victims_number_{dataset_name}")
    victims = select_victims(victims_number, dataset_name)
    campaign_name = params["campaign_name"]

    crafted = []
    to_remove = []
    next_index = 0
    now = datetime.now().isoformat

    report_content = (
        f"SYNTHETIC FRAUD GENERATION REPORT DATASET {dataset_name} - {now}\n"
    )

    for i, strategy_dict in enumerate(params.get("strategies")):
        old_index = next_index
        next_index, temp_craft, temp_remove = craft_frauds(
            df_legit,
            strategy_dict,
            next_index,
            victims_number,
            victims,
            dataset_name,
        )
        crafted += temp_craft
        to_remove += temp_remove
        print(f"strategy: {i}")
        print(f"frauds: {len(temp_craft)} victims: {next_index - old_index}")
        print(f"total fraud generated: {len(temp_craft)}")

        report_content += (
            f"Strategy {i}:\t\t\t Frauds generated: "
            f"{len(temp_craft)}\t\t\tVictims: "
            f"{next_index-old_index}\n"
        )
        report_content += f"Total fraud generated: {len(temp_craft)} \n\n"

    synth_fraud_folder = FRAUD_GENERATED_PATH / campaign_name
    print(synth_fraud_folder, Path(synth_fraud_folder))
    os.makedirs(synth_fraud_folder, exist_ok=True)

    synth_frauds_df = pd.concat(crafted)
    synth_frauds_df = sort_and_reset(synth_frauds_df)
    dataset.save_synthetic_frauds(dataset_name, synth_frauds_df, dir=synth_fraud_folder)

    try:
        df_to_remove = pd.concat(to_remove)
        df_to_remove["Fraud"] = 0
        df_to_remove.reset_index(drop=True, inplace=True)
        dataset.save_removed_legit_transactions(
            dataset_name, df_to_remove, dir=synth_fraud_folder
        )
    except ValueError:  # skip if no objects to concatenate
        dataset.save_removed_legit_transactions(
            dataset_name,
            pd.DataFrame(columns=synth_frauds_df.columns),
            dir=synth_fraud_folder,
        )

    fraud_report = synth_fraud_folder / SYNTH_FRAUDS_REPORT
    with open(fraud_report, "w") as report_file:
        report_file.write(report_content)

    print("Report written at %s" % fraud_report)
