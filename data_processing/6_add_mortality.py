# Import libraries

import h5py
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from variables import UF_DATA_DIR, EICU_DATA_DIR, MIMIC_DATA_DIR, MAIN_DIR

import os

if not os.path.exists(MAIN_DIR):
    os.makedirs(MAIN_DIR)

outcomes_uf = pd.read_csv(f"{UF_DATA_DIR}/outcomes_w_dead.csv")

outcomes_eicu = pd.read_csv(f"{EICU_DATA_DIR}/outcomes_w_dead.csv")
outcomes_eicu = outcomes_eicu.rename({"stay_id": "icustay_id"}, axis=1)

outcomes_mimic = pd.read_csv(f"{MIMIC_DATA_DIR}/outcomes_w_dead.csv")
outcomes_mimic = outcomes_mimic.rename({"stay_id": "icustay_id"}, axis=1)

cols = ["icustay_id", "coma", "delirium", "dead"]

outcomes_uf = outcomes_uf.loc[:,cols]
outcomes_eicu = outcomes_eicu.loc[:,cols]
outcomes_mimic = outcomes_mimic.loc[:,cols]

outcomes_mimic["icustay_id"] = outcomes_mimic["icustay_id"].astype(int)
outcomes_eicu["icustay_id"] = outcomes_eicu["icustay_id"].astype(int)
outcomes_uf["icustay_id"] = outcomes_uf["icustay_id"].astype(int)

outcomes_mimic["icustay_id"] = "MIMIC_" + outcomes_mimic["icustay_id"].astype(str)
outcomes_eicu["icustay_id"] = "EICU_" + outcomes_eicu["icustay_id"].astype(str)
outcomes_uf["icustay_id"] = "UF_" + outcomes_uf["icustay_id"].astype(str)

outcomes = pd.concat([outcomes_eicu, outcomes_mimic, outcomes_uf])

del outcomes_eicu, outcomes_mimic, outcomes_uf

print(outcomes.head())

print(f'Total ICU admissions: {len(outcomes["icustay_id"].unique())}')

print(outcomes["dead"].value_counts())

outcomes.to_csv(f"{MAIN_DIR}/outcomes_w_dead.csv", index=False)


