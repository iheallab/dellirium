# Import libraries

import h5py
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from variables import DATA_DIR, MODEL_DIR

import os

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

seq = pd.read_csv(f"{DATA_DIR}/clinical_data.csv")

def remove_outliers(group):

    lower_threshold = group["value"].quantile(0.01)
    upper_threshold = group["value"].quantile(0.99)

    group_filtered = group[
        (group["value"] >= lower_threshold) & (
            group["value"] <= upper_threshold)
    ]

    return group_filtered


seq = seq.groupby("variable").apply(remove_outliers).reset_index(drop=True)

seq["hours"] = seq["hours"].astype(int)

# Pivot dataframe to have variables as columns

seq = seq.pivot_table(index=["icustay_id", "hours"], columns="variable", values="value", aggfunc="mean").reset_index()

# Expand dataframe so each icustay_id has a row for each hour
hours_range = pd.DataFrame({"hours": range(0, 24)})
icustay_ids = pd.DataFrame({"icustay_id": seq["icustay_id"].unique()})

hours_range = icustay_ids.merge(hours_range, how="cross")

seq = hours_range.merge(seq, on=["icustay_id", "hours"], how="left")

# Add columns to indicate missingness of each column
missingness_cols = seq.columns[2:]
for col in missingness_cols:
    seq[f"{col}_missing"] = seq[col].isnull().astype(int)

# Resample to hourly means and impute missing values
def resample_and_impute(group):
    group = group.ffill().bfill()
    return group

seq = seq.groupby("icustay_id").apply(resample_and_impute)
seq = seq.reset_index(drop=True)

# Impute remaining missing values with mean imputation

with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train = ids["train"]
    ids_val = ids["val"]
    ids_int = ids["internal"]
    ids_ext1 = ids["external_1"]
    ids_ext2 = ids["external_2"]

imputer = SimpleImputer(strategy="mean")
imputer = imputer.fit(seq[seq["icustay_id"].isin(ids_train)].iloc[:, 2:])
seq.iloc[:, 2:] = imputer.transform(seq.iloc[:, 2:])

scaler = MinMaxScaler()
scaler = scaler.fit(seq[seq["icustay_id"].isin(ids_train)].iloc[:, 2:])
seq.iloc[:, 2:] = scaler.transform(seq.iloc[:, 2:])

seq = seq.sort_values(by=["icustay_id", "hours"])

print(f"NaNs in seq after imputation: {seq.isnull().sum().sum()}")

# Process static data

static = pd.read_csv(f"{DATA_DIR}/static.csv")

static.replace([np.inf, -np.inf], np.nan, inplace=True)

num_cols = ["age", "bmi"]
cat_cols = ["gender", "race"]
exclude = num_cols + cat_cols + ["icustay_id"]
comob_cols = [col for col in static.columns if col not in exclude]

gender_scaler = LabelEncoder()
race_scaler = LabelEncoder()

static["gender"] = gender_scaler.fit_transform(static["gender"])
static["race"] = race_scaler.fit_transform(static["race"])

static.loc[:, comob_cols] = static.loc[:, comob_cols].fillna(0)

imputer = SimpleImputer(strategy="median")
imputer = imputer.fit(
    static[static["icustay_id"].isin(ids_train)].loc[:, num_cols])

static.loc[:, num_cols] = imputer.transform(static.loc[:, num_cols])

imputer = SimpleImputer(strategy="median")
imputer = imputer.fit(
    static[static["icustay_id"].isin(ids_train)].loc[:, cat_cols])

static.loc[:, cat_cols] = imputer.transform(static.loc[:, cat_cols])

scaler = MinMaxScaler()

scaler = scaler.fit(static[static["icustay_id"].isin(ids_train)].iloc[:, 1:])

static.iloc[:, 1:] = scaler.transform(static.iloc[:, 1:])

static = static.sort_values(by=["icustay_id"])

print(f"NaNs in static after imputation: {static.isnull().sum().sum()}")

# Load outcomes

icustay_ids = seq["icustay_id"].unique()

outcomes = pd.read_csv(
    f"{DATA_DIR}/outcomes.csv").loc[:, ["icustay_id", "delirium"]]

outcomes = outcomes[outcomes["icustay_id"].isin(icustay_ids)]
outcomes = outcomes.sort_values(by=["icustay_id"])

static.iloc[:, 1:] = static.iloc[:, 1:].astype(float)

static = static.iloc[:, 1:].values.astype(np.float64)
outcomes = outcomes["delirium"].values.reshape((-1, 1))

ids_train = np.isin(icustay_ids, ids_train)
ids_val = np.isin(icustay_ids, ids_val)
ids_int = np.isin(icustay_ids, ids_int)
ids_ext1 = np.isin(icustay_ids, ids_ext1)
ids_ext2 = np.isin(icustay_ids, ids_ext2)

qkv = seq.iloc[:, 2:].values.astype(np.float64)
qkv = qkv.reshape((len(icustay_ids), int(qkv.shape[0]/len(icustay_ids)), qkv.shape[1]))

X_train = qkv[ids_train]
static_train = static[ids_train]
y_train = outcomes[ids_train]
print(X_train.shape)
print(static_train.shape)

X_val = qkv[ids_val]
static_val = static[ids_val]
y_val = outcomes[ids_val]
print(X_val.shape)
print(static_val.shape)

X_int = qkv[ids_int]
static_int = static[ids_int]
y_int = outcomes[ids_int]
print(X_int.shape)
print(static_int.shape)

X_ext1 = qkv[ids_ext1]
static_ext1 = static[ids_ext1]
y_ext1 = outcomes[ids_ext1]
print(X_ext1.shape)
print(static_ext1.shape)

X_ext2 = qkv[ids_ext2]
static_ext2 = static[ids_ext2]
y_ext2 = outcomes[ids_ext2]
print(X_ext2.shape)
print(static_ext2.shape)

enc = OneHotEncoder()

y_train = enc.fit_transform(y_train).toarray()[:, 1:]
y_val = enc.fit_transform(y_val).toarray()[:, 1:]
y_int = enc.fit_transform(y_int).toarray()[:, 1:]
y_ext1 = enc.fit_transform(y_ext1).toarray()[:, 1:]
y_ext2 = enc.fit_transform(y_ext2).toarray()[:, 1:]

print(y_train.shape)
print(y_val.shape)
print(y_int.shape)
print(y_ext1.shape)
print(y_ext2.shape)

# Save hdf5 file


with h5py.File("%s/dataset.h5" % MODEL_DIR, "w", libver="latest") as hf:

    # Save train data
    training_group = hf.create_group("train")
    training_group.create_dataset("X", data=X_train)
    training_group.create_dataset("static", data=static_train)
    training_group.create_dataset("y", data=y_train)

    # Save validation data
    validation_group = hf.create_group("val")
    validation_group.create_dataset("X", data=X_val)
    validation_group.create_dataset("static", data=static_val)
    validation_group.create_dataset("y", data=y_val)

    # Save internal test data
    int_group = hf.create_group("internal")
    int_group.create_dataset("X", data=X_int)
    int_group.create_dataset("static", data=static_int)
    int_group.create_dataset("y", data=y_int)
    
    # Save external test 1 data
    ext1_group = hf.create_group("external_1")
    ext1_group.create_dataset("X", data=X_ext1)
    ext1_group.create_dataset("static", data=static_ext1)
    ext1_group.create_dataset("y", data=y_ext1)
    
    # Save external test 2 data
    ext2_group = hf.create_group("external_2")
    ext2_group.create_dataset("X", data=X_ext2)
    ext2_group.create_dataset("static", data=static_ext2)
    ext2_group.create_dataset("y", data=y_ext2)