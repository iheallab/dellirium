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

static_uf = pd.read_csv(f"{DATA_DIR}/static.csv")

static_uf = static_uf[static_uf["icustay_id"].isin(icustay_ids)]

static_uf["sex"] = static_uf["sex"].map({0: "Female", 1: "Male"})
static_uf["race"] = static_uf["race"].map({0: "black", 1: "white", 2: "other"})

cols_names = (
    ["icustay_id", "sex", "age", "race", "bmi"]
    + [comob for comob in static_uf.columns.tolist() if "_poa" in comob]
    + ["charlson_comorbidity_total_score"]
)

static_uf = static_uf[cols_names]

print(static_uf.head())
print(static_uf.shape)

static_uf.to_csv(f"{OUTPUT_DIR}/intermediate/static.csv", index=False)

