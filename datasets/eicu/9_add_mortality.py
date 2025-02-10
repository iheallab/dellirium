# Import libraries

import pandas as pd
import numpy as np
import math
import os

from variables import DATA_DIR, OUTPUT_DIR


outcomes = pd.read_csv(f"{OUTPUT_DIR}/final/outcomes.csv")

admissions_eicu = pd.read_csv(
    f'{DATA_DIR}/patient.csv.gz', compression='gzip',
    usecols=["patientunitstayid", "unitdischargeoffset", "unitdischargelocation"],
)

admissions_eicu = admissions_eicu[admissions_eicu["unitdischargeoffset"] > 2880]

dead_ids = admissions_eicu[admissions_eicu["unitdischargelocation"] == "Death"]

outcomes["dead"] = 0

outcomes.loc[(outcomes["stay_id"].isin(dead_ids["patientunitstayid"].unique())), "dead"] = 1

print(outcomes["dead"].value_counts())

outcomes.to_csv(f"{OUTPUT_DIR}/final/outcomes_w_dead.csv", index=None)

