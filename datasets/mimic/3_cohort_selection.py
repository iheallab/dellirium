# Import libraries

import pandas as pd
import numpy as np
import math
import os

from variables import DATA_DIR, OUTPUT_DIR

if not os.path.exists(f"{OUTPUT_DIR}/intermediate"):
    os.makedirs(f"{OUTPUT_DIR}/intermediate")

cohort_flow = pd.DataFrame(data=[], columns=["MIMIC"])

# Load ICU stays

icustays = pd.read_csv(f'{DATA_DIR}/icustays.csv.gz', compression='gzip',
                       usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])

print(f"Initial Patients: {len(icustays['subject_id'].unique())}")
print(f"Initial ICU admissions: {len(icustays['stay_id'].unique())}")

orig = len(icustays['stay_id'].unique())

# Keep only first ICU admission
icustays = icustays.sort_values(by=['subject_id', 'intime'])
icustays = icustays.drop_duplicates(subset=['subject_id'], keep='first')

dropped = orig - len(icustays['stay_id'].unique())

cohort_flow.loc["First admission", "MIMIC"] = dropped

orig = len(icustays['stay_id'].unique())

# Keep ICU admissions lasting more than 1 day
icustays = icustays[icustays['los'] >= 1]

dropped = orig - len(icustays['stay_id'].unique())

cohort_flow.loc["los", "MIMIC"] = dropped

orig = len(icustays['stay_id'].unique())

# Load Patients file

patients = pd.read_csv(f'{DATA_DIR}/patients.csv.gz',
                       compression='gzip', usecols=['subject_id', 'anchor_age'])
patients = patients.merge(icustays.loc[:, ['subject_id', 'stay_id']], on=[
                          'subject_id'], how='inner')
patients = patients.rename({"anchor_age": "age"}, axis=1)

# Filter patients by age
patients = patients[patients['age'] >= 18]
icustays = icustays[icustays['stay_id'].isin(patients['stay_id'].unique())]

dropped = orig - len(icustays['stay_id'].unique())

cohort_flow.loc["Age", "MIMIC"] = dropped

orig = len(icustays['stay_id'].unique())

# Load admissions file

admissions = pd.read_csv(f'{DATA_DIR}/admissions.csv.gz', compression='gzip')
admissions = admissions.dropna(subset=['deathtime'])

# Merge ICU stays and admissions

icustays = icustays.merge(
    admissions.loc[:, ['hadm_id', 'deathtime']], how='left', on='hadm_id')

del admissions

# Drop patients that passed away within 48h
icustays['hours'] = (pd.to_datetime(icustays['deathtime']) -
                     pd.to_datetime(icustays['intime'])) / np.timedelta64(1, 'h')
icustays = icustays[~(icustays['hours'] < 48)]

dropped = orig - len(icustays['stay_id'].unique())

cohort_flow.loc["Time of death", "MIMIC"] = dropped

orig = len(icustays['stay_id'].unique())

icustays.drop('deathtime', axis=1, inplace=True)
icustays.drop('hours', axis=1, inplace=True)

# Drop patients with coma/delirium in first 24h

brain_status = pd.read_csv(f"{OUTPUT_DIR}/raw/brain_status.csv")

brain_status = brain_status[brain_status['stay_id'].isin(
    icustays['stay_id'].unique())]

icustays = icustays[icustays['stay_id'].isin(brain_status['stay_id'].unique())]

dropped = orig - len(icustays['stay_id'].unique())

cohort_flow.loc["Brain status", "MIMIC"] = dropped

orig = len(icustays['stay_id'].unique())

print(f"Filtered Patients: {len(icustays['subject_id'].unique())}")
print(f"Filtered ICU admissions: {len(icustays['stay_id'].unique())}")

icustays.to_csv(f'{OUTPUT_DIR}/intermediate/icustays.csv', index=None)

cohort_flow.to_csv(f'{OUTPUT_DIR}/intermediate/cohort_flow.csv')