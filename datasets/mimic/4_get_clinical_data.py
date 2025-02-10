# Import libraries
import pandas as pd
import numpy as np
import pickle
import os

from variables import OUTPUT_DIR, DATA_DIR

# %%

# Get ICU admissions from cohort

admissions = pd.read_csv(f"{OUTPUT_DIR}/intermediate/icustays.csv")

icustay_ids = admissions["stay_id"].unique()


# %%

# Get variables to extract

variable_map = pd.read_csv("rsc/extended_features.csv")

variables = variable_map["mimic"].values

ditems = pd.read_csv(f"{DATA_DIR}/d_items.csv.gz", compression="gzip")

ditems = ditems[ditems["label"].isin(variables)]

itemids = ditems["itemid"].values


# # %%

# Extract all events

# Specify the chunk size (number of rows to read at a time)
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv(f"{DATA_DIR}/chartevents.csv.gz", compression="gzip", chunksize=chunk_size):
    chunk = chunk[chunk["stay_id"].isin(icustay_ids)]
    chunk = chunk[chunk["itemid"].isin(itemids)]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

chartevents = pd.concat(chunks)

# Convert temperature F to C
chartevents.loc[(chartevents["itemid"] == 223761), "valuenum"] = (
    (chartevents.loc[(chartevents["itemid"]
     == 223761), "valuenum"] - 32) * 5 / 9
)

chartevents.to_csv(f"{OUTPUT_DIR}/intermediate/all_events.csv", index=None)

del chunks

# %%

# Extract med events

# Specify the chunk size (number of rows to read at a time)
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv(f"{DATA_DIR}/inputevents.csv.gz", compression="gzip", chunksize=chunk_size):
    chunk = chunk[chunk["stay_id"].isin(icustay_ids)]
    chunk = chunk[chunk["itemid"].isin(itemids)]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

chartevents = pd.concat(chunks)


chartevents.to_csv(f"{OUTPUT_DIR}/intermediate/med_events.csv", index=None)

del chunks

# %%

# Extract height and weight

# Specify the chunk size (number of rows to read at a time)
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv(f"{DATA_DIR}/chartevents.csv.gz", compression="gzip", chunksize=chunk_size):
    chunk = chunk[chunk["stay_id"].isin(icustay_ids)]
    chunk = chunk[chunk["itemid"].isin([226512, 226730])]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

chartevents = pd.concat(chunks)


chartevents.to_csv(f"{OUTPUT_DIR}/intermediate/height_weight.csv", index=None)

del chunks

# %%

# Extract height and weight in US units and convert

# Specify the chunk size (number of rows to read at a time)
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv(f"{DATA_DIR}/chartevents.csv.gz", compression="gzip", chunksize=chunk_size):
    chunk = chunk[chunk["stay_id"].isin(icustay_ids)]
    chunk = chunk[chunk["itemid"].isin([226531, 226707])]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

chartevents = pd.concat(chunks)


chartevents.to_csv(
    f"{OUTPUT_DIR}/intermediate/height_weight_us.csv", index=None)

del chunks


height_weight = pd.read_csv(f"{OUTPUT_DIR}/intermediate/height_weight_us.csv")

height = height_weight[height_weight["itemid"] == 226707]
weight = height_weight[height_weight["itemid"] == 226531]


def inches_to_cm(inches):
    # Convert inches to centimeters
    return inches * 2.54


def lbs_to_kg(pounds):
    # Convert pounds to kilograms
    return pounds * 0.453592


height["valuenum"] = inches_to_cm(height["valuenum"])
height["itemid"] = 226730

weight["valuenum"] = lbs_to_kg(weight["valuenum"])
weight["itemid"] = 226512

height_weight = pd.concat([height, weight])

height_weight.to_csv(
    f"{OUTPUT_DIR}/intermediate/height_weight_conv.csv", index=None)

# %%

# Extract and convert temperature F to C

# Specify the chunk size (number of rows to read at a time)
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv(f"{DATA_DIR}/chartevents.csv.gz", compression="gzip", chunksize=chunk_size):
    chunk = chunk[chunk["stay_id"].isin(icustay_ids)]
    chunk = chunk[chunk["itemid"].isin([223761])]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

temp = pd.concat(chunks)

print(len(temp["stay_id"].unique()))


def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    return celsius


temp["valuenum"] = fahrenheit_to_celsius(temp["valuenum"])
temp["itemid"] = 223762

temp.to_csv(f"{OUTPUT_DIR}/intermediate/temp_conv.csv", index=None)


# %%

# Extract scores

ditems = pd.read_csv(f"{DATA_DIR}/d_items.csv.gz", compression="gzip")

cam = [
    228300,
    228301,
    228302,
    228303,
    228334,
    228335,
    228336,
    228337,
    229324,
    229325,
    229326,
]
rass = [228096, 228299]
gcs = [220739, 223900, 223901]

all_scores = cam + rass + gcs
chunk_size = 100000
chunks = []

count = 1
# Read the CSV file in chunks
for chunk in pd.read_csv(
    f"{DATA_DIR}/chartevents.csv.gz",
    compression="gzip",
    chunksize=chunk_size,
    usecols=["subject_id", "stay_id", "charttime",
             "itemid", "value", "valuenum"],
):
    chunk = chunk[chunk["stay_id"].isin(icustay_ids)]
    chunk = chunk[chunk["itemid"].isin(all_scores)]
    chunks.append(chunk)
    print(f"Processed chunk {count}")
    count += 1

brain_status = pd.concat(chunks)

map_cam = dict()

for i in range(len(all_scores)):
    map_cam[all_scores[i]] = (
        ditems[ditems["itemid"] == all_scores[i]]["label"].values[0].lower()
    )

brain_status["label"] = brain_status["itemid"].map(map_cam)

brain_status = brain_status.sort_values(by=["stay_id", "charttime"])

brain_status.loc[brain_status["value"] == "Unable to Assess", "valuenum"] = -1

brain_status = brain_status[["stay_id", "charttime", "label", "valuenum"]]

brain_status = brain_status.set_index(["stay_id", "charttime", "label"])

multi_index = brain_status.index
data = brain_status.values.flatten()
brain_status = pd.Series(data, index=multi_index)
print("Unstacking")
brain_status = brain_status.unstack("label")

brain_status = brain_status.reset_index()

brain_status.insert(0, "cam", np.nan)

brain_status.loc[
    (
        (brain_status["cam-icu ms change"] == 1)
        & (brain_status["cam-icu inattention"] == 1)
    )
    & (
        (brain_status["cam-icu disorganized thinking"] == 1)
        | (brain_status["cam-icu altered loc"] == 1)
        | (brain_status["cam-icu rass loc"] == 1)
    ),
    "cam",
] = 1
brain_status.loc[
    (brain_status["cam-icu ms change"] == 0)
    | (brain_status["cam-icu inattention"] == 0)
    | (brain_status["cam-icu disorganized thinking"] == 0)
    | (brain_status["cam-icu altered loc"] == 0)
    | (brain_status["cam-icu rass loc"] == 0),
    "cam",
] = 0
brain_status.loc[
    (brain_status["cam-icu ms change"] == -1)
    | (brain_status["cam-icu inattention"] == -1)
    | (brain_status["cam-icu disorganized thinking"] == -1)
    | (brain_status["cam-icu altered loc"] == -1)
    | (brain_status["cam-icu rass loc"] == -1),
    "cam",
] = -1

brain_status["gcs"] = (
    brain_status["gcs - eye opening"]
    + brain_status["gcs - verbal response"]
    + brain_status["gcs - motor response"]
)

brain_status.loc[
    brain_status[["gcs - eye opening",
                  "gcs - verbal response", "gcs - motor response"]]
    .isna()
    .any(axis=1),
    "gcs",
] = np.nan

brain_status = brain_status.rename({"richmond-ras scale": "rass"}, axis=1)

brain_status = brain_status[["stay_id", "charttime", "cam", "gcs", "rass"]]

icustay_ids = brain_status["stay_id"].unique()

admissions = pd.read_csv(f"{OUTPUT_DIR}/intermediate/icustays.csv", usecols=[
                         "stay_id", "intime", "outtime"])

admissions = admissions[admissions["stay_id"].isin(icustay_ids)]

brain_status = brain_status.merge(admissions, on=["stay_id"])

# Stack the three columns into one column based on the 'id' column using pd.melt()
stacked_column = pd.melt(
    brain_status, id_vars=["stay_id", "charttime"], value_vars=["cam", "gcs", "rass"]
)

# Rename the columns for clarity
stacked_column.columns = ["stay_id", "charttime", "variable", "value"]

stacked_column.sort_values(by=["stay_id", "charttime"], inplace=True)

stacked_column.dropna(subset=["value"], inplace=True)

stacked_column.to_csv(f"{OUTPUT_DIR}/intermediate/scores_raw.csv", index=None)
