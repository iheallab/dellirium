# Import libraries

import pandas as pd
import numpy as np
import pickle

from variables import MAIN_DIR

all_ids = pd.read_csv(f"{MAIN_DIR}/outcomes.csv")["icustay_id"].unique()

uf_ids = [icuid for icuid in all_ids if "UF_" in icuid]
eicu_ids = [icuid for icuid in all_ids if "EICU_" in icuid]
mimic_ids = [icuid for icuid in all_ids if "MIMIC_" in icuid]

develop_ids = uf_ids

print(f"UF admissions: {len(uf_ids)}")
print(f"eICU admissions: {len(eicu_ids)}")
print(f"MIMIC admissions: {len(mimic_ids)}")

# Define the proportions for train, validation, and test sets
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Calculate the number of samples for each set
num_samples = len(develop_ids)
num_train = int(train_ratio * num_samples)
num_validation = int(validation_ratio * num_samples)
num_test = num_samples - num_train - num_validation

# Shuffle the IDs
np.random.shuffle(develop_ids)

# Split the IDs into train, validation, and test sets
train_ids = develop_ids[:num_train]
validation_ids = develop_ids[num_train:num_train + num_validation]
test_ids = develop_ids[num_train + num_validation:]

# Print the IDs in each set
print("Train IDs:", len(train_ids))
print("Val IDs:", len(validation_ids))
print("Internal Test IDs:", len(test_ids))
print("External Test 1 IDs:", len(mimic_ids))
print("External Test 2 IDs:", len(eicu_ids))

with open(f"{MAIN_DIR}/ids.pkl", "wb") as f:
    pickle.dump(
        {
            "train": train_ids,
            "val": validation_ids,
            "internal": test_ids,
            "external_1": mimic_ids,
            "external_2": eicu_ids,
        },
        f,
        protocol=2,
    )
