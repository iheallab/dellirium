# Import libraries

import h5py
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from variables import INTERNAL_DATA_DIR, EICU_DATA_DIR, MIMIC_DATA_DIR, MAIN_DIR

import os

if not os.path.exists(MAIN_DIR):
    os.makedirs(MAIN_DIR)

outcomes_internal = pd.read_csv(f"{INTERNAL_DATA_DIR}/outcomes.csv")

outcomes_eicu = pd.read_csv(f"{EICU_DATA_DIR}/outcomes.csv")
outcomes_eicu = outcomes_eicu.rename({"stay_id": "icustay_id"}, axis=1)

outcomes_mimic = pd.read_csv(f"{MIMIC_DATA_DIR}/outcomes.csv")
outcomes_mimic = outcomes_mimic.rename({"stay_id": "icustay_id"}, axis=1)

cols = ["icustay_id", "coma", "delirium"]

outcomes_internal = outcomes_internal.loc[:,cols]
outcomes_eicu = outcomes_eicu.loc[:,cols]
outcomes_mimic = outcomes_mimic.loc[:,cols]

outcomes_mimic["icustay_id"] = outcomes_mimic["icustay_id"].astype(int)
outcomes_eicu["icustay_id"] = outcomes_eicu["icustay_id"].astype(int)
outcomes_internal["icustay_id"] = outcomes_internal["icustay_id"].astype(int)

outcomes_mimic["icustay_id"] = "MIMIC_" + outcomes_mimic["icustay_id"].astype(str)
outcomes_eicu["icustay_id"] = "EICU_" + outcomes_eicu["icustay_id"].astype(str)
outcomes_internal["icustay_id"] = "internal_" + outcomes_internal["icustay_id"].astype(str)

outcomes = pd.concat([outcomes_eicu, outcomes_mimic, outcomes_internal])

del outcomes_eicu, outcomes_mimic, outcomes_internal

print(outcomes.head())

print(f'Total ICU admissions: {len(outcomes["icustay_id"].unique())}')

print(outcomes["delirium"].value_counts())

outcomes.to_csv(f"{MAIN_DIR}/outcomes.csv", index=False)

