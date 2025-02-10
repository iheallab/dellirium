# Import libraries

import pandas as pd
import numpy as np
import math
import os

from variables import DATA_DIR, OUTPUT_DIR

if not os.path.exists(f"{OUTPUT_DIR}/intermediate"):
    os.makedirs(f"{OUTPUT_DIR}/intermediate")

cohort_flow = pd.DataFrame(data=[], columns=["eICU"])

# Load ICU stays

icustays = pd.read_csv(f'{DATA_DIR}/patient.csv.gz', compression='gzip',
                       usecols=['patientunitstayid', 'uniquepid', 'patienthealthsystemstayid', 'hospitaladmitoffset'])

print(f"Initial Patients: {len(icustays['uniquepid'].unique())}")
print(f"Initial Patients: {len(icustays['patienthealthsystemstayid'].unique())}")
print(f"Initial ICU admissions: {len(icustays['patientunitstayid'].unique())}")

orig = len(icustays['patientunitstayid'].unique())

# Keep only first ICU admission
icustays = icustays.sort_values(by=['patienthealthsystemstayid', 'uniquepid', 'hospitaladmitoffset'])
icustays = icustays.drop_duplicates(subset=['patienthealthsystemstayid'], keep='last')
icustays = icustays.drop_duplicates(subset=['uniquepid'], keep='last')

dropped = orig - len(icustays['patientunitstayid'].unique())

cohort_flow.loc["First admission", "eICU"] = dropped

orig = len(icustays['patientunitstayid'].unique())

# Keep ICU admissions lasting more than 1 day

los_eicu = pd.read_csv(
    f'{DATA_DIR}/patient.csv.gz', compression='gzip',
    usecols=["patientunitstayid", "unitdischargeoffset"],
)
los_eicu = los_eicu[
    los_eicu["patientunitstayid"].isin(icustays["patientunitstayid"].unique())
]

lower_lim = 1440

los_eicu = los_eicu[los_eicu["unitdischargeoffset"] <= 43200]
los_eicu = los_eicu[los_eicu["unitdischargeoffset"] >= lower_lim]

icustays = icustays[icustays['patientunitstayid'].isin(los_eicu['patientunitstayid'].unique())]

dropped = orig - len(icustays['patientunitstayid'].unique())

cohort_flow.loc["los", "eICU"] = dropped

orig = len(icustays['patientunitstayid'].unique())

# Filter by missing info and age

age_eicu = pd.read_csv(
    f'{DATA_DIR}/patient.csv.gz', compression='gzip',
    usecols=["patientunitstayid", "age", "gender", "ethnicity", "unitdischargestatus"],
)
age_eicu = age_eicu[
    age_eicu["patientunitstayid"].isin(icustays["patientunitstayid"].unique())
]
age_eicu = age_eicu.dropna(subset=["age", "gender", "ethnicity", "unitdischargestatus"])
age_eicu["age"] = age_eicu["age"].replace({"> 89": 89})
age_eicu["age"] = age_eicu["age"].astype(float)
age_eicu = age_eicu[age_eicu['age'] >= 18]

icustays = icustays[icustays['patientunitstayid'].isin(age_eicu['patientunitstayid'].unique())]

dropped = orig - len(icustays['patientunitstayid'].unique())

cohort_flow.loc["Age", "eICU"] = dropped

orig = len(icustays['patientunitstayid'].unique())

# Drop patients that passed away within 48h

admissions_eicu = pd.read_csv(
    f'{DATA_DIR}/patient.csv.gz', compression='gzip',
    usecols=["patientunitstayid", "unitdischargeoffset", "unitdischargelocation"],
)

admissions_eicu = admissions_eicu[admissions_eicu["unitdischargeoffset"] <= 2880]

dead_ids = admissions_eicu[admissions_eicu["unitdischargelocation"] == "Death"]

icustays = icustays[~(icustays["patientunitstayid"].isin(dead_ids["patientunitstayid"].unique()))]

dropped = orig - len(icustays['patientunitstayid'].unique())

cohort_flow.loc["Time of death", "eICU"] = dropped

orig = len(icustays['patientunitstayid'].unique())

# Drop patients with coma/delirium in first 24h

brain_status = pd.read_csv(f"{OUTPUT_DIR}/raw/brain_status.csv")

brain_status = brain_status[brain_status['stay_id'].isin(
    icustays['patientunitstayid'].unique())]

icustays = icustays[icustays['patientunitstayid'].isin(brain_status['stay_id'].unique())]

dropped = orig - len(icustays['patientunitstayid'].unique())

cohort_flow.loc["Brain status", "eICU"] = dropped

orig = len(icustays['patientunitstayid'].unique())

print(f"Filtered Patients: {len(icustays['uniquepid'].unique())}")
print(f"Filtered ICU admissions: {len(icustays['patientunitstayid'].unique())}")

icustays = icustays.rename({"patientunitstayid": "stay_id"}, axis=1)

icustays.to_csv(f'{OUTPUT_DIR}/intermediate/icustays.csv', index=None)

cohort_flow.to_csv(f'{OUTPUT_DIR}/intermediate/cohort_flow.csv')