import sys

import pandas as pd
import numpy as np

from dataset import dataset
from utils.utilities import DATASET_SPLIT_TS
from config.config import TRAINING_DATASET_PATH


def remove_correlated_pair(ignore_set, C_CF, corr, threshold):

    F = [f for f in corr.columns if f not in ignore_set]
    C = corr[F].filter(items=F, axis=0)
    pairs = C.mask(np.triu(np.ones(C.shape)).astype(bool)).stack()

    R = set()

    for i in pairs.index:
        coef = abs(pairs[i])
        if coef > threshold:
            if i[0] in R or i[1] in R:
                continue
            target_coefs = abs(C_CF[i[0]]), abs(C_CF[i[1]])
            worst = 0 if target_coefs[0] < target_coefs[1] else 1
            print("Correlated features: %s, %s" % i)
            print("Remove: %s, Correlation Coefficient: %f, Target Corr.: %f" %
                  (i[worst], coef, target_coefs[worst]))
            R.add(i[worst])

    final_set = (set(F).difference(R))
    return final_set


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    outfile = sys.argv[2]

    dataset_directory = TRAINING_DATASET_PATH

    ts_split = DATASET_SPLIT_TS

    df_aggr = dataset.load_aggregated(dataset_name, end_ts=ts_split)
    R = [f for f in df_aggr.columns if "8760h" in f] + ["confirm_SMS"]

    df_aggr = df_aggr[set(df_aggr.columns).difference(R)]
    corr = df_aggr.corr()
    target = ["Fraud"]

    C_CF: pd.Series = corr["Fraud"]
    C_FF: pd.DataFrame = corr[[f for f in corr.columns if f not in target]]

    I = set(["Amount", "is_international", "is_national_iban",
            "is_new_asn_cc", "is_new_iban", "is_new_iban_cc", "is_new_ip"])
    
    ignore_set = I.union(set(target))

    F = set(corr.columns).difference(ignore_set)

    feature_set = remove_correlated_pair(ignore_set, C_CF, C_FF, 0.95)
    feature_set = feature_set.union(set(I))

    with open(outfile, "w") as f:
        for feature in sorted(list(feature_set)):
            f.write("%s\n" % feature)