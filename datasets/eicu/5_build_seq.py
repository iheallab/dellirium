# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

admissions = pd.read_csv(f"{OUTPUT_DIR}/intermediate/icustays.csv")

icustay_ids = admissions["stay_id"].unique()

vitals = pd.read_csv(
    f"{OUTPUT_DIR}/intermediate/vitals.csv",
    usecols=[
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ],
)

use_vitals = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 Saturation",
    "Non-Invasive BP Diastolic",
    "Non-Invasive BP Systolic",
    "Temperature (C)",
    "Temperature (F)",
    "End Tidal CO2",
]

vitals = vitals[vitals["nursingchartcelltypevalname"].isin(use_vitals)]

vitals["nursingchartvalue"] = vitals["nursingchartvalue"].astype(float)

def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    return celsius

vitals.loc[
    (vitals["nursingchartcelltypevalname"] == "Temperature (F)"), "nursingchartvalue"
] = fahrenheit_to_celsius(
    vitals.loc[
        (vitals["nursingchartcelltypevalname"] == "Temperature (F)"),
        "nursingchartvalue",
    ]
)

vitals["nursingchartcelltypevalname"] = vitals["nursingchartcelltypevalname"].replace(
    {"Temperature (F)": "Temperature (C)"}
)

vitals = vitals.rename(
    {
        "patientunitstayid": "icustay_id",
        "nursingchartoffset": "hours",
        "nursingchartcelltypevalname": "label",
        "nursingchartvalue": "value",
    },
    axis=1,
)

vitals = vitals[["icustay_id", "hours", "label", "value"]]

vitals["hours"] = vitals["hours"] / 60

vitals["value"] = vitals["value"].astype(float)

labs = pd.read_csv(
    f"{DATA_DIR}/lab.csv.gz", compression="gzip",
    usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
)

labs = labs.rename(
    {
        "patientunitstayid": "icustay_id",
        "labresultoffset": "hours",
        "labname": "label",
        "labresult": "value",
    },
    axis=1,
)

labs = labs.dropna(subset=["value"])

labs["hours"] = labs["hours"] / 60

labs["value"] = labs["value"].astype(float)

seq_eicu = pd.concat([vitals, labs], axis=0)

del labs
del vitals

resp = pd.read_csv(
    f"{DATA_DIR}/respiratoryCharting.csv.gz", compression="gzip",
    usecols=[
        "patientunitstayid",
        "respchartoffset",
        "respchartvaluelabel",
        "respchartvalue",
    ],
)

use_resp = [
    "FiO2",
    "PEEP",
    "Tidal Volume Observed (VT)",
    "Peak Insp. Pressure",
    "Oxygen Flow Rate",
]

resp = resp[resp["respchartvaluelabel"].isin(use_resp)]

resp = resp.rename(
    {
        "patientunitstayid": "icustay_id",
        "respchartoffset": "hours",
        "respchartvaluelabel": "label",
        "respchartvalue": "value",
    },
    axis=1,
)

resp = resp.dropna(subset=["value"])

resp["hours"] = resp["hours"] / 60

resp["value"] = resp["value"].str.replace("%", "")

resp["value"] = resp["value"].astype(float)

seq_eicu = pd.concat([seq_eicu, resp], axis=0)

del resp

meds = pd.read_csv(
    f"{DATA_DIR}/infusionDrug.csv.gz", compression="gzip",
    usecols=["patientunitstayid", "infusionoffset", "drugname", "drugamount"],
)

meds = meds.rename(
    {
        "patientunitstayid": "icustay_id",
        "infusionoffset": "hours",
        "drugname": "label",
        "drugamount": "value",
    },
    axis=1,
)

meds = meds.dropna(subset=["value"])

meds.loc[
    meds["label"].str.contains(
        r"\b{}\b".format("norepinephrine"), case=False, regex=True
    ),
    "label",
] = "Norepinephrine"
meds.loc[
    meds["label"].str.contains(r"\b{}\b".format("dopamine"), case=False, regex=True),
    "label",
] = "Dopamine"
meds.loc[
    meds["label"].str.contains(r"\b{}\b".format("epinephrine"), case=False, regex=True),
    "label",
] = "Epinephrine"
meds.loc[
    meds["label"].str.contains(
        r"\b{}\b".format("phenylephrine"), case=False, regex=True
    ),
    "label",
] = "Phenylephrine"
meds.loc[
    meds["label"].str.contains(r"\b{}\b".format("vasopressin"), case=False, regex=True),
    "label",
] = "Vasopressin"

meds["hours"] = meds["hours"] / 60

meds["value"] = meds["value"].astype(float)

meds["value"] = 1

seq_eicu = pd.concat([seq_eicu, meds], axis=0)

del meds

scores = pd.read_csv(
    f"{OUTPUT_DIR}/intermediate/scores.csv",
    usecols=[
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ],
)

cam = scores[scores["nursingchartcelltypevalname"] == "Delirium Score"]
cam = cam[(cam["nursingchartvalue"] == "Yes") | (cam["nursingchartvalue"] == "No")]
cam["nursingchartvalue"] = cam["nursingchartvalue"].map({"Yes": 1, "No": 0})
cam["nursingchartcelltypevalname"] = "CAM"

gcs = scores[scores["nursingchartcelltypevalname"] == "GCS Total"]
gcs = gcs[gcs["nursingchartvalue"] != "Unable to score due to medication"]
gcs = gcs.dropna(subset=["nursingchartvalue"])
gcs["nursingchartvalue"] = gcs["nursingchartvalue"].astype(int)
gcs = gcs[(gcs["nursingchartvalue"] <= 15) & (gcs["nursingchartvalue"] >= 0)]
gcs["nursingchartcelltypevalname"] = "GCS"

rass = scores[scores["nursingchartcelltypevalname"] == "Sedation Score"]
rass["nursingchartvalue"] = rass["nursingchartvalue"].astype(int)
rass = rass[(rass["nursingchartvalue"] <= 4) & (rass["nursingchartvalue"] >= -5)]
rass["nursingchartcelltypevalname"] = "RASS"

del scores

scores = pd.concat([cam, gcs, rass], axis=0)

scores = scores.rename(
    {
        "patientunitstayid": "icustay_id",
        "nursingchartoffset": "hours",
        "nursingchartcelltypevalname": "label",
        "nursingchartvalue": "value",
    },
    axis=1,
)

scores = scores[["icustay_id", "hours", "label", "value"]]

scores["hours"] = scores["hours"] / 60

scores["value"] = scores["value"].astype(float)

seq_eicu = pd.concat([seq_eicu, scores], axis=0)

del scores, cam, gcs, rass

seq_eicu = seq_eicu.sort_values(by=["icustay_id", "hours"])

seq_eicu = seq_eicu[(seq_eicu["hours"] >= 0) & (seq_eicu["hours"] < 24)]

seq_eicu = seq_eicu[seq_eicu["icustay_id"].isin(icustay_ids)]

def remove_outliers(group):

    lower_threshold = group["value"].quantile(0.01)
    upper_threshold = group["value"].quantile(0.99)

    group_filtered = group[
        (group["value"] >= lower_threshold) & (group["value"] <= upper_threshold)
    ]

    return group_filtered


seq_eicu = seq_eicu.groupby("label").apply(remove_outliers)

seq_eicu = seq_eicu.reset_index(drop=True)

seq_eicu = seq_eicu.sort_values(by=["icustay_id", "hours"])

map_var = pd.read_csv("rsc/extended_features.csv")
feat_eicu = map_var["eicu"].tolist()

seq_eicu = seq_eicu[seq_eicu["label"].isin(feat_eicu)]

seq_eicu = seq_eicu.rename({"label": "variable"}, axis=1)

print(f'Variables in eICU: {len(seq_eicu["variable"].unique())}')
print(f'Variables in eICU: {list(seq_eicu["variable"].unique())}')
print(f'ICU admissions in eICU: {len(seq_eicu["icustay_id"].unique())}')

seq_eicu = seq_eicu.reset_index(drop=True)

seq_eicu = seq_eicu.rename({"icustay_id": "stay_id"}, axis=1)

seq_eicu.to_csv(f"{OUTPUT_DIR}/intermediate/seq.csv", index=False)
