# Import libraries

import pandas as pd
import numpy as np
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
        (group["value"] >= lower_threshold) & (group["value"] <= upper_threshold)
    ]

    return group_filtered

seq = seq.groupby("variable").apply(remove_outliers).reset_index(drop=True)

features = list(seq["variable"].unique())

# Calculate statistical features

# Define a function to calculate statistical features and tabularize sequential data
def calculate_and_tabularize(seq, stat_func, suffix):
    seq_stat = (
        seq.groupby(by=["icustay_id", "variable"])["value"].apply(stat_func).reset_index()
    )
    seq_stat["variable"] = seq_stat["variable"] + suffix
    seq_stat = seq_stat.set_index(["icustay_id", "variable"])
    multi_index = seq_stat.index
    data = seq_stat.values.flatten()
    seq_stat = pd.Series(data, index=multi_index)
    print("Unstacking")
    seq_stat = seq_stat.unstack("variable")
    return seq_stat

# Calculate statistical features
seq_mean = calculate_and_tabularize(seq, stat_func='mean', suffix='_mean')
seq_std = calculate_and_tabularize(seq, stat_func='std', suffix='_std')
seq_min = calculate_and_tabularize(seq, stat_func='min', suffix='_min')
seq_max = calculate_and_tabularize(seq, stat_func='max', suffix='_max')


# Calculate missing values
seq_missing = seq_mean.iloc[:, 1:].isnull().astype(int)
seq_missing.columns = [col.replace("_mean", "_missing") for col in seq_mean.columns[1:]]

# Concatenate all DataFrames
seq = pd.concat([seq_mean, seq_std.iloc[:, 1:], 
                 seq_min.iloc[:, 1:], 
                 seq_max.iloc[:, 1:],
                 seq_missing], axis=1).reset_index()

print(seq.head())

static = pd.read_csv(f"{DATA_DIR}/static.csv")

static.replace([np.inf, -np.inf], np.nan, inplace=True)

import pickle
with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train = ids["train"]
    ids_val = ids["val"]
    ids_int = ids["internal"]
    ids_ext1 = ids["external_1"]
    ids_ext2 = ids["external_2"]

# Impute input data

drop_cols = []

seq_train = seq[seq["icustay_id"].isin(ids_train)]

for column in seq.columns[1:]:
    
    if seq_train.loc[:, column].isna().all():
        
        drop_cols.append(column)

del seq_train
        
seq.drop(columns=drop_cols, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

imputer = imputer.fit(seq[seq["icustay_id"].isin(ids_train)].iloc[:, 1:])

seq.iloc[:, 1:] = imputer.transform(seq.iloc[:, 1:])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler = scaler.fit(seq[seq["icustay_id"].isin(ids_train)].iloc[:, 1:])

seq.iloc[:, 1:] = scaler.transform(seq.iloc[:, 1:])

num_cols = ["age", "bmi"]
cat_cols = ["gender", "race"]

exclude = num_cols + cat_cols + ["icustay_id"]

comob_cols = [col for col in static.columns if col not in exclude]

from sklearn.preprocessing import LabelEncoder

for col in cat_cols:

    enc = LabelEncoder()

    static.loc[:, col] = enc.fit_transform(static.loc[:, col])

static.loc[:, comob_cols] = static.loc[:, comob_cols].fillna(0)

imputer = SimpleImputer(strategy="median")
imputer = imputer.fit(static[static["icustay_id"].isin(ids_train)].loc[:, num_cols])

static.loc[:, num_cols] = imputer.transform(static.loc[:, num_cols])

imputer = SimpleImputer(strategy="median")
imputer = imputer.fit(static[static["icustay_id"].isin(ids_train)].loc[:, cat_cols])

static.loc[:, cat_cols] = imputer.transform(static.loc[:, cat_cols])

scaler = MinMaxScaler()

scaler = scaler.fit(static[static["icustay_id"].isin(ids_train)].iloc[:, 1:])

static.iloc[:, 1:] = scaler.transform(static.iloc[:, 1:])

seq = seq.merge(static, how="inner", on="icustay_id")

del static

feature_names = seq.columns.tolist()[1:]
print(feature_names)

with open(f"{MODEL_DIR}/features.pkl", "wb") as f:
    pickle.dump(
        {
            "features": feature_names
        },
        f,
        protocol=2,
    )

# Load outcomes

outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv").loc[:,["icustay_id", "delirium"]]

seq = seq.merge(outcomes, how="inner", on="icustay_id")

print(seq.head())

print(f"NaN values: {seq.isna().sum().sum()}")

X_train = seq[seq["icustay_id"].isin(ids_train)].iloc[:, 1:-1].values.astype(np.float64)
X_val = seq[seq["icustay_id"].isin(ids_val)].iloc[:, 1:-1].values.astype(np.float64)
X_int = seq[seq["icustay_id"].isin(ids_int)].iloc[:, 1:-1].values.astype(np.float64)
X_ext1 = seq[seq["icustay_id"].isin(ids_ext1)].iloc[:, 1:-1].values.astype(np.float64)
X_ext2 = seq[seq["icustay_id"].isin(ids_ext2)].iloc[:, 1:-1].values.astype(np.float64)

y_train = seq[seq["icustay_id"].isin(ids_train)].iloc[:, -1].values.reshape((-1,1))
y_val = seq[seq["icustay_id"].isin(ids_val)].iloc[:, -1].values.reshape((-1,1))
y_int = seq[seq["icustay_id"].isin(ids_int)].iloc[:, -1].values.reshape((-1,1))
y_ext1 = seq[seq["icustay_id"].isin(ids_ext1)].iloc[:, -1].values.reshape((-1,1))
y_ext2 = seq[seq["icustay_id"].isin(ids_ext2)].iloc[:, -1].values.reshape((-1,1))

enc = OneHotEncoder()

y_train = enc.fit_transform(y_train).toarray()[:,1:]
y_val = enc.fit_transform(y_val).toarray()[:,1:]
y_int = enc.fit_transform(y_int).toarray()[:,1:]
y_ext1 = enc.fit_transform(y_ext1).toarray()[:,1:]
y_ext2 = enc.fit_transform(y_ext2).toarray()[:,1:]

print(y_train.shape)
print(y_val.shape)
print(y_int.shape)
print(y_ext1.shape)
print(y_ext2.shape)


# Save hdf5 file

import h5py

with h5py.File("%s/dataset.h5" % MODEL_DIR, "w", libver="latest") as hf:

    # Save train data
    training_group = hf.create_group("train")
    training_group.create_dataset("X", data=X_train)
    training_group.create_dataset("y", data=y_train)

    # Save validation data
    validation_group = hf.create_group("val")
    validation_group.create_dataset("X", data=X_val)
    validation_group.create_dataset("y", data=y_val)

    # Save external test data
    internal_group = hf.create_group("internal")
    internal_group.create_dataset("X", data=X_int)
    internal_group.create_dataset("y", data=y_int)
    
    # Save external test data
    external1_group = hf.create_group("external_1")
    external1_group.create_dataset("X", data=X_ext1)
    external1_group.create_dataset("y", data=y_ext1)
    
    # Save external test data
    external2_group = hf.create_group("external_2")
    external2_group.create_dataset("X", data=X_ext2)
    external2_group.create_dataset("y", data=y_ext2)
