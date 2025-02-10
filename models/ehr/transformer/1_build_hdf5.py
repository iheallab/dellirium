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


def convert_variables_to_indices(seq):
    print("* Converting variables to indices...")
    var_idx, var_label = pd.factorize(seq["variable"])
    var_idx = var_idx + 1  # 0 will be for padding
    seq["variable_code"] = var_idx

    variable_code_mapping = {
        i + 1: label for (i, label) in enumerate(var_label)}
    variable_code_mapping[0] = "<PAD>"
    return seq, variable_code_mapping


seq, variable_mapping = convert_variables_to_indices(seq)

seq = seq.sort_values(by=["icustay_id", "hours"]).reset_index(drop=True)

with open("%s/variable_mapping.pkl" % MODEL_DIR, "wb") as f:
    pickle.dump(variable_mapping, f, protocol=2)

# Scale data

with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train = ids["train"]

SET_TRAIN = set(ids_train)
VALUE_COLS = ["value"]

scalers = {}


def _standardize_variable(group):
    ids_present = set(group["icustay_id"].unique())
    train_ids_present = ids_present.intersection(SET_TRAIN)
    variable_value = group["variable"].values[0]

    variable_code = group["variable_code"].values[0]

    if (
        len(train_ids_present) > 0
        and variable_value != "mob_level_of_assistance"
        and variable_value != "brain_status"
    ):
        scaler = MinMaxScaler()
        scaler.fit(group[group["icustay_id"].isin(train_ids_present)][VALUE_COLS])
        # scaler = scalers[f'scaler{variable_code}']
        group[VALUE_COLS] = scaler.transform(group[VALUE_COLS])
        scalers[f"scaler{variable_code}"] = scaler

    elif (
        len(train_ids_present) == 0
        and variable_value != "mob_level_of_assistance"
        and variable_value != "brain_status"
    ):
        print("Debug standardize")
        scaler = MinMaxScaler()
        group[VALUE_COLS] = scaler.fit_transform(group[VALUE_COLS])
        # scaler = scalers[f'scaler{variable_code}']
        # group[VALUE_COLS] = scaler.transform(group[VALUE_COLS])
        scalers[f"scaler{variable_code}"] = scaler

    return group


def standardize_variables(df):

    variables = df["variable"].unique()
    print(f"Variables after imputation: {len(variables)}")

    df = (
        df.groupby("variable")
        .apply(_standardize_variable)
        .sort_values(by=["icustay_id", "hours"])
    )
    variables = df["variable"].unique()
    print(f"Variables after standardization: {len(variables)}")

    return df


def get_hours_scaler(df):
    hours = df[["icustay_id", "hours"]]
    scaler = MinMaxScaler()
    scaler.fit(hours[hours["icustay_id"].isin(ids_train)][["hours"]])
    return scaler


print("* Scaling seq...")
hours_scaler = get_hours_scaler(seq)

scalers["scaler_hours"] = hours_scaler

seq = standardize_variables(seq)

print(f'Final ICU admissions: {len(seq["icustay_id"].unique())}')
seq = seq.reset_index(drop=True)

seq[["hours"]] = hours_scaler.transform(seq[["hours"]])

seq.drop("variable", axis=1, inplace=True)

pickle.dump(scalers, open("%s/scalers_seq.pkl" % MODEL_DIR, "wb"))

# Prepare sequential input data

seq_len = seq.groupby('icustay_id').size(
).reset_index().rename({0: 'seq_len'}, axis=1)

icustay_ids = seq_len["icustay_id"].tolist()

max_seq_len = 512

seq_len = seq_len['seq_len'].tolist()

qkv = np.zeros((len(icustay_ids), max_seq_len, 3))
X = seq.iloc[:, :].values

start = 0
for i in range(len(icustay_ids)):
    if seq_len[i] <= max_seq_len:
        end = start + seq_len[i]
        qkv[i, :seq_len[i], 0] = X[start:end, 1]
        qkv[i, :seq_len[i], 1] = X[start:end, 3]
        qkv[i, :seq_len[i], 2] = X[start:end, 2]
    else:
        end = start + seq_len[i]
        trunc = end - max_seq_len
        qkv[i, :, 0] = X[trunc:end, 1]
        qkv[i, :, 1] = X[trunc:end, 3]
        qkv[i, :, 2] = X[trunc:end, 2]

    start += seq_len[i]


    
print(qkv)
# Process static data

static = pd.read_csv(f"{DATA_DIR}/static.csv")

scalers = {}

static.replace([np.inf, -np.inf], np.nan, inplace=True)

with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train = ids["train"]
    ids_val = ids["val"]
    ids_int = ids["internal"]
    ids_ext1 = ids["external_1"]
    ids_ext2 = ids["external_2"]


num_cols = ["age", "bmi"]
cat_cols = ["gender", "race"]
exclude = num_cols + cat_cols + ["icustay_id"]
comob_cols = [col for col in static.columns if col not in exclude]

gender_scaler = LabelEncoder()
race_scaler = LabelEncoder()

static["gender"] = gender_scaler.fit_transform(static["gender"])
static["race"] = race_scaler.fit_transform(static["race"])

scalers["scaler_gender"] = gender_scaler
scalers["scaler_race"] = race_scaler

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

scalers["scaler_static"] = scaler

pickle.dump(scalers, open("%s/scalers_static.pkl" % MODEL_DIR, "wb"))

# Load outcomes

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