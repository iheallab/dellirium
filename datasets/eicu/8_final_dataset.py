# Import libraries

import pandas as pd
import numpy as np
import torch

import os

from variables import OUTPUT_DIR, DATA_DIR

if not os.path.exists(f"{OUTPUT_DIR}/final"):
    os.makedirs(f"{OUTPUT_DIR}/final")

# Load clinical data

seq = pd.read_csv(f'{OUTPUT_DIR}/intermediate/seq.csv', usecols=["stay_id", "hours", "variable", "value"])

def remove_outliers(group):

    lower_threshold = group["value"].quantile(0.01)
    upper_threshold = group["value"].quantile(0.99)

    group_filtered = group[
        (group["value"] >= lower_threshold) & (group["value"] <= upper_threshold)
    ]

    return group_filtered

seq = seq.groupby("variable").apply(remove_outliers)

seq = seq.reset_index(drop=True)

seq = seq.sort_values(by=["stay_id", "hours"])

# Load static data

static = pd.read_csv(f'{OUTPUT_DIR}/intermediate/static.csv')

# Keep ICU admissions with clinical data in first 24h

static = static[static["stay_id"].isin(seq["stay_id"].unique())]

# Load and filter ICU admissions

icustays = pd.read_csv(f'{OUTPUT_DIR}/intermediate/icustays.csv')

orig = len(icustays['stay_id'].unique())

icustays = icustays[icustays["stay_id"].isin(seq["stay_id"].unique())]

dropped = orig - len(icustays['stay_id'].unique())


cohort_flow = pd.read_csv(f'{OUTPUT_DIR}/intermediate/cohort_flow.csv', index_col="Unnamed: 0")
cohort_flow.loc["Missing data", "eICU"] = dropped

hospitals = pd.read_csv(f'{DATA_DIR}/patient.csv.gz', compression='gzip',
                       usecols=['patientunitstayid', 'hospitalid'])

hospitals = hospitals[hospitals["patientunitstayid"].isin(icustays['stay_id'].unique())]
hospitals = len(hospitals["hospitalid"].unique())

icu_admissions = len(icustays['stay_id'].unique())

cohort_flow.loc["Final ICU admissions", "eICU"] = icu_admissions
cohort_flow.loc["Final hospitals", "eICU"] = hospitals

cohort_flow.to_csv(f'{OUTPUT_DIR}/final/cohort_flow.csv')

seq = seq[seq["stay_id"].isin(icustays["stay_id"].unique())]
static = static[static["stay_id"].isin(icustays["stay_id"].unique())]

# Label deceased patients

icustays = pd.read_csv(f'{OUTPUT_DIR}/intermediate/outcomes.csv')

icustays = icustays[icustays["stay_id"].isin(seq["stay_id"].unique())]

# icustays.loc[(icustays['stay_id'].isin(deceased['stay_id'].unique())), 'deceased'] = 1

print(seq.head())
print(static.head())
print(icustays.head())

print(f"ICU admissions with clinical data: {len(seq['stay_id'].unique())}")
print(f"ICU admissions with static data: {len(static['stay_id'].unique())}")
print(f"ICU admissions with outcome data: {len(icustays['stay_id'].unique())}")


print(f"Outcome count:\n {icustays['coma'].value_counts()}\n {icustays['delirium'].value_counts()}")


seq.to_csv(f"{OUTPUT_DIR}/final/clinical_data.csv", index=False)
static.to_csv(f"{OUTPUT_DIR}/final/static.csv", index=False)
icustays.to_csv(f"{OUTPUT_DIR}/final/outcomes.csv", index=False)
