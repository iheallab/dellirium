# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import math

from variables import DATA_DIR, OUTPUT_DIR

# Build MIMIC static data
admissions = pd.read_csv("%s/intermediate/icustays.csv" % OUTPUT_DIR)
icustay_ids = admissions["icustay_id"].tolist()

static_internal = pd.read_csv(f"{DATA_DIR}/static.csv")

static_internal = static_internal[static_internal["icustay_id"].isin(icustay_ids)]

static_internal["sex"] = static_internal["sex"].map({0: "Female", 1: "Male"})
static_internal["race"] = static_internal["race"].map({0: "black", 1: "white", 2: "other"})

cols_names = (
    ["icustay_id", "sex", "age", "race", "bmi"]
    + [comob for comob in static_internal.columns.tolist() if "_poa" in comob]
    + ["charlson_comorbidity_total_score"]
)

static_internal = static_internal[cols_names]

print(static_internal.head())
print(static_internal.shape)

static_internal.to_csv(f"{OUTPUT_DIR}/intermediate/static.csv", index=False)

