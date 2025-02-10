# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

admissions = pd.read_csv(f"{OUTPUT_DIR}/intermediate/icustays.csv")

icustay_ids = admissions["stay_id"].unique()

ditems = pd.read_csv("%s/d_items.csv.gz" % DATA_DIR, compression="gzip")

labs_vitals = pd.read_csv(
    "%s/intermediate/all_events.csv" % OUTPUT_DIR,
    usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
)
labs_vitals = pd.concat(
    [
        labs_vitals,
        pd.read_csv(
            "%s/intermediate/temp_conv.csv" % OUTPUT_DIR,
            usecols=["subject_id", "stay_id", "charttime", "itemid", "valuenum"],
        ),
    ]
)

labs_vitals = labs_vitals.merge(admissions, on=["stay_id", "subject_id"])

labs_vitals["hours"] = (
    pd.to_datetime(labs_vitals["charttime"]) - pd.to_datetime(labs_vitals["intime"])
) / np.timedelta64(1, "h")


labitems = ditems[ditems["itemid"].isin(labs_vitals["itemid"].unique())]
labitems_ids = labitems["itemid"].unique()

map_labs = dict()

for i in range(len(labitems_ids)):
    map_labs[labitems_ids[i]] = labitems[labitems["itemid"] == labitems_ids[i]][
        "label"
    ].values[0]

labs_vitals["label"] = labs_vitals["itemid"].map(map_labs)


def remove_outliers(group):

    lower_threshold = group["valuenum"].quantile(0.01)
    upper_threshold = group["valuenum"].quantile(0.99)

    group_filtered = group[
        (group["valuenum"] >= lower_threshold) & (group["valuenum"] <= upper_threshold)
    ]

    return group_filtered


labs_vitals = labs_vitals.groupby("label").apply(remove_outliers)

labs_vitals = labs_vitals.reset_index(drop=True)

lab_variables = list(labs_vitals["label"].unique())

meds = pd.read_csv("%s/intermediate/med_events.csv" % OUTPUT_DIR)

meds = meds.merge(admissions, on=["stay_id", "subject_id"])


meds["hours"] = (
    pd.to_datetime(meds["starttime"]) - pd.to_datetime(meds["intime"])
) / np.timedelta64(1, "h")

meditems = ditems[ditems["itemid"].isin(meds["itemid"].unique())]
meditems_ids = meditems["itemid"].unique()

map_meds = dict()

for i in range(len(meditems_ids)):
    map_meds[meditems_ids[i]] = meditems[meditems["itemid"] == meditems_ids[i]][
        "label"
    ].values[0]

meds["label"] = meds["itemid"].map(map_meds)


def remove_outliers(group):

    lower_threshold = group["amount"].quantile(0.01)
    upper_threshold = group["amount"].quantile(0.99)

    group_filtered = group[
        (group["amount"] >= lower_threshold) & (group["amount"] <= upper_threshold)
    ]

    return group_filtered


meds = meds.groupby("label").apply(remove_outliers)

meds = meds.rename({"amount": "valuenum"}, axis=1)

meds["valuenum"] = 1

labs_vitals = labs_vitals.loc[:, ["stay_id", "hours", "label", "valuenum"]]
meds = meds.loc[:, ["stay_id", "hours", "label", "valuenum"]]

labs_vitals = labs_vitals.rename({"label": "variable"}, axis=1)
labs_vitals = labs_vitals.rename({"valuenum": "value"}, axis=1)

meds = meds.rename({"label": "variable"}, axis=1)
meds = meds.rename({"valuenum": "value"}, axis=1)

scores = pd.read_csv("%s/intermediate/scores_raw.csv" % OUTPUT_DIR)
scores = scores.merge(admissions, on=["stay_id"])

scores["hours"] = (
    pd.to_datetime(scores["charttime"]) - pd.to_datetime(scores["intime"])
) / np.timedelta64(1, "h")
scores = scores.loc[:, ["stay_id", "hours", "variable", "value"]]

seq_mimic = pd.concat([labs_vitals, meds, scores], axis=0).sort_values(
    by=["stay_id", "hours"]
)

del labs_vitals, meds, scores

seq_mimic = seq_mimic[seq_mimic["stay_id"].isin(icustay_ids)]

seq_mimic = seq_mimic[(seq_mimic["hours"] >= 0) & (seq_mimic["hours"] < 24)]

map_var = pd.read_csv("rsc/extended_features.csv")
feat_mimic = map_var["mimic"].tolist()

seq_mimic = seq_mimic[seq_mimic["variable"].isin(feat_mimic)]

print(f'Variables in MIMIC: {len(seq_mimic["variable"].unique())}')
print(f'ICU admissions in MIMIC: {len(seq_mimic["stay_id"].unique())}')

seq_mimic = seq_mimic.reset_index(drop=True)

seq_mimic.to_csv(f"{OUTPUT_DIR}/intermediate/seq.csv", index=False)
