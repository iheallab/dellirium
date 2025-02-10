# Import libraries

import pandas as pd
import numpy as np
import math
import os

from variables import DATA_DIR, OUTPUT_DIR


outcomes = pd.read_csv(f"{OUTPUT_DIR}/final/outcomes.csv")

icustays = pd.read_csv(f'{DATA_DIR}/icustays.csv.gz', compression='gzip',
                       usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])

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
icustays = icustays[(icustays['hours'] > 48)]

outcomes["dead"] = 0

outcomes.loc[(outcomes["stay_id"].isin(icustays["stay_id"].unique())), "dead"] = 1

print(outcomes["dead"].value_counts())

outcomes.to_csv(f"{OUTPUT_DIR}/final/outcomes_w_dead.csv", index=None)

