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

static_internal = pd.read_csv(f"{INTERNAL_DATA_DIR}/static.csv")
static_internal = static_internal.rename({"sex": "gender"}, axis=1)
static_internal = static_internal.rename({"charlson_comorbidity_total_score": "cci"}, axis=1)


static_eicu = pd.read_csv(f"{EICU_DATA_DIR}/static.csv")
static_eicu = static_eicu.rename({"stay_id": "icustay_id"}, axis=1)

static_mimic = pd.read_csv(f"{MIMIC_DATA_DIR}/static.csv")
static_mimic.drop("hadm_id", axis=1, inplace=True)
static_mimic = static_mimic.rename({"stay_id": "icustay_id"}, axis=1)
static_mimic = static_mimic.rename({"anchor_age": "age"}, axis=1)

features = static_eicu.columns.tolist()

static_mimic = static_mimic.loc[:, features]
static_internal.columns = features

static_mimic["icustay_id"] = static_mimic["icustay_id"].astype(int)
static_eicu["icustay_id"] = static_eicu["icustay_id"].astype(int)
static_internal["icustay_id"] = static_internal["icustay_id"].astype(int)

static_mimic["icustay_id"] = "MIMIC_" + static_mimic["icustay_id"].astype(str)
static_eicu["icustay_id"] = "EICU_" + static_eicu["icustay_id"].astype(str)
static_internal["icustay_id"] = "internal_" + static_internal["icustay_id"].astype(str)

static = pd.concat([static_eicu, static_mimic, static_internal])

del static_mimic, static_eicu, static_internal

print(static.head())

print(f'Total ICU admissions: {len(static["icustay_id"].unique())}')

static.to_csv(f"{MAIN_DIR}/static.csv", index=False)
