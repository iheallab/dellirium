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

seq_internal = pd.read_csv(f"{INTERNAL_DATA_DIR}/clinical_data.csv")
seq_eicu = pd.read_csv(f"{EICU_DATA_DIR}/clinical_data.csv")
seq_mimic = pd.read_csv(f"{MIMIC_DATA_DIR}/clinical_data.csv")

map_var = pd.read_csv("rsc/extended_features.csv")

map_internal_mimic = {}

for i in range(len(map_var)):
    map_internal_mimic[map_var.loc[i, "internal"]] = map_var.loc[i, "mimic"]

seq_internal["variable"] = seq_internal["variable"].map(map_internal_mimic)

map_eicu_mimic = {}

for i in range(len(map_var)):
    map_eicu_mimic[map_var.loc[i, "eicu"]] = map_var.loc[i, "mimic"]

seq_eicu["variable"] = seq_eicu["variable"].map(map_eicu_mimic)

print(f'MIMIC variables: {len(seq_mimic["variable"].unique())}')
print(f'eICU variables: {len(seq_eicu["variable"].unique())}')
print(f'internal variables: {len(seq_internal["variable"].unique())}')

var_mimic = list(seq_mimic["variable"].unique())
var_eicu = list(seq_eicu["variable"].unique())

missing_var = [var for var in var_mimic if var not in var_eicu]

print(f"Missing variables: {len(missing_var)}")


seq_mimic = seq_mimic.rename({"stay_id": "icustay_id"}, axis=1)
seq_eicu = seq_eicu.rename({"stay_id": "icustay_id"}, axis=1)

seq_mimic["icustay_id"] = seq_mimic["icustay_id"].astype(int)
seq_eicu["icustay_id"] = seq_eicu["icustay_id"].astype(int)
seq_internal["icustay_id"] = seq_internal["icustay_id"].astype(int)

seq_mimic["icustay_id"] = "MIMIC_" + seq_mimic["icustay_id"].astype(str)
seq_eicu["icustay_id"] = "EICU_" + seq_eicu["icustay_id"].astype(str)
seq_internal["icustay_id"] = "internal_" + seq_internal["icustay_id"].astype(str)

seq = pd.concat([seq_eicu, seq_mimic, seq_internal])

del seq_mimic, seq_eicu, seq_internal

seq = seq.dropna()
seq = seq.sort_values(by=["icustay_id", "hours"]).reset_index(drop=True)

print(seq.head())

print(f'Total variables: {len(seq["variable"].unique())}')
print(f'Total ICU admissions: {len(seq["icustay_id"].unique())}')

seq.to_csv(f"{MAIN_DIR}/clinical_data_unconverted.csv", index=False)
