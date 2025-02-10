# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

admissions = pd.read_csv(f"{OUTPUT_DIR}/intermediate/icustays.csv")

icustay_ids = admissions["icustay_id"].unique()

seq_internal = pd.read_csv(f"{DATA_DIR}/seq.csv")

seq_internal = seq_internal[seq_internal["icustay_id"].isin(icustay_ids)]

map_var = pd.read_csv("rsc/extended_features.csv")
feat_internal = map_var["internal"].tolist()

seq_internal = seq_internal[seq_internal["variable"].isin(feat_internal)]


def remove_outliers(group):

    lower_threshold = group["value"].quantile(0.01)
    upper_threshold = group["value"].quantile(0.99)

    group_filtered = group[
        (group["value"] >= lower_threshold) & (group["value"] <= upper_threshold)
    ]

    return group_filtered


seq_internal = seq_internal.groupby("variable").apply(remove_outliers).reset_index(drop=True)

meds = [med for med in seq_internal["variable"].unique() if "med_" in med]

seq_internal.loc[seq_internal["variable"].isin(meds), "value"] = 1

seq_internal = seq_internal.sort_values(by=["icustay_id", "hours"])

seq_internal = seq_internal[(seq_internal["hours"] >= 0) & (seq_internal["hours"] < 24)]

print(f'Variables in internal: {len(seq_internal["variable"].unique())}')
print(f'ICU admissions in internal: {len(seq_internal["icustay_id"].unique())}')

seq_internal.drop("variable_code", axis=1, inplace=True)

seq_internal = seq_internal.reset_index(drop=True)

seq_internal.to_csv(f"{OUTPUT_DIR}/intermediate/seq.csv", index=False)
