# Import libraries

import pandas as pd
import numpy as np
import math
import os

from variables import DATA_DIR, OUTPUT_DIR

if not os.path.exists(f"{OUTPUT_DIR}/intermediate"):
    os.makedirs(f"{OUTPUT_DIR}/intermediate")

cohort_flow = pd.DataFrame(data=[], columns=["internal"])

# Load ICU stays

icustays = pd.read_csv(f'{DATA_DIR}/icustays.csv')

icustays['los'] = (pd.to_datetime(icustays['exit_datetime']) - pd.to_datetime(icustays['enter_datetime'])) / np.timedelta64(1, "h")
icustays['los'] = icustays['los'] / 24

print(f"Initial Patients: {len(icustays['patient_deiden_id'].unique())}")
print(f"Initial ICU admissions: {len(icustays['icustay_id'].unique())}")

orig = len(icustays['icustay_id'].unique())

# Keep only first ICU admission
icustays = icustays.sort_values(by=['patient_deiden_id', 'enter_datetime'])
icustays = icustays.drop_duplicates(subset=['patient_deiden_id'], keep='first')

dropped = orig - len(icustays['icustay_id'].unique())

cohort_flow.loc["First admission", "internal"] = dropped

orig = len(icustays['icustay_id'].unique())

# Keep ICU admissions lasting more than 1 day and less than 30 days
icustays = icustays[(icustays['los'] >= 1) & (icustays['los'] <= 30)]

dropped = orig - len(icustays['icustay_id'].unique())

cohort_flow.loc["los", "internal"] = dropped

orig = len(icustays['icustay_id'].unique())

# Load Patients file

patients = pd.read_csv(f'{DATA_DIR}/static.csv', usecols=['icustay_id', 'age'])

# Filter patients by age
patients = patients[patients['age'] >= 18]
icustays = icustays[icustays['icustay_id'].isin(patients['icustay_id'].unique())]

dropped = orig - len(icustays['icustay_id'].unique())

cohort_flow.loc["Age", "internal"] = dropped

orig = len(icustays['icustay_id'].unique())

# Drop patients that passed away within 48h

admissions = pd.read_csv(f'{DATA_DIR}/outcomes.csv', usecols=["icustay_id", "start_hours", "final_states_v2"])
admissions = admissions[admissions["final_states_v2"] == "Dead"]

admissions = admissions[admissions["start_hours"] <= 48]

icustays = icustays[~(icustays["icustay_id"].isin(admissions["icustay_id"].unique()))]

dropped = orig - len(icustays['icustay_id'].unique())

cohort_flow.loc["Time of death", "internal"] = dropped

orig = len(icustays['icustay_id'].unique())

# Drop patients with coma/delirium in first 24h

brain_status = pd.read_csv(f"{OUTPUT_DIR}/raw/brain_status.csv")

brain_status = brain_status[brain_status['icustay_id'].isin(
    icustays['icustay_id'].unique())]

icustays = icustays[icustays['icustay_id'].isin(brain_status['icustay_id'].unique())]

dropped = orig - len(icustays['icustay_id'].unique())

cohort_flow.loc["Brain status", "internal"] = dropped

orig = len(icustays['icustay_id'].unique())

print(f"Filtered Patients: {len(icustays['patient_deiden_id'].unique())}")
print(f"Filtered ICU admissions: {len(icustays['icustay_id'].unique())}")

icustays.to_csv(f'{OUTPUT_DIR}/intermediate/icustays.csv', index=None)

cohort_flow.to_csv(f'{OUTPUT_DIR}/intermediate/cohort_flow.csv')