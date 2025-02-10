#%%
# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR


#%%

# Build sequence data

seq_uf = pd.read_csv(f"{DATA_DIR}/seq.csv")

map_var = pd.read_csv("rsc/extended_features.csv")
feat_uf = map_var["uf"].tolist()

seq_uf = seq_uf[seq_uf["variable"].isin(feat_uf)]


def remove_outliers(group):

    lower_threshold = group["value"].quantile(0.01)
    upper_threshold = group["value"].quantile(0.99)

    group_filtered = group[
        (group["value"] >= lower_threshold) & (group["value"] <= upper_threshold)
    ]

    return group_filtered


seq_uf = seq_uf.groupby("variable").apply(remove_outliers).reset_index(drop=True)

meds = [med for med in seq_uf["variable"].unique() if "med_" in med]

seq_uf.loc[seq_uf["variable"].isin(meds), "value"] = 1

seq_uf = seq_uf.sort_values(by=["icustay_id", "hours"])

seq_uf["interval"] = ((seq_uf["hours"] // 24) + 1).astype(int)

print(f'Variables in UF: {len(seq_uf["variable"].unique())}')
print(f'ICU admissions in UF: {len(seq_uf["icustay_id"].unique())}')

seq_uf.drop("variable_code", axis=1, inplace=True)

seq_uf = seq_uf.reset_index(drop=True)

seq_uf.to_csv(f"{OUTPUT_DIR}/intermediate/seq_pretrain.csv", index=False)


#%%

# Build static data

static_uf = pd.read_csv(f"{DATA_DIR}/static.csv")

static_uf["sex"] = static_uf["sex"].map({0: "Female", 1: "Male"})
static_uf["race"] = static_uf["race"].map({0: "black", 1: "white", 2: "other"})

cols_names = (
    ["icustay_id", "sex", "age", "race", "bmi"]
    + [comob for comob in static_uf.columns.tolist() if "_poa" in comob]
    + ["charlson_comorbidity_total_score"]
)

static_uf = static_uf[cols_names]

print(static_uf.head())
print(static_uf.shape)

static_uf.to_csv(f"{OUTPUT_DIR}/intermediate/static_pretrain.csv", index=False)

# %%
