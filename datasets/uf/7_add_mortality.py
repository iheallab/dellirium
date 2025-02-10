# Import libraries

import pandas as pd
import numpy as np
import math
import os

from variables import DATA_DIR, OUTPUT_DIR


outcomes = pd.read_csv(f"{OUTPUT_DIR}/final/outcomes.csv")

admissions = pd.read_csv(f'{DATA_DIR}/outcomes.csv', usecols=["icustay_id", "start_hours", "final_states_v2"])
admissions = admissions[admissions["final_states_v2"] == "Dead"]

admissions = admissions[admissions["start_hours"] > 48]

outcomes["dead"] = 0

outcomes.loc[(outcomes["icustay_id"].isin(admissions["icustay_id"].unique())), "dead"] = 1

print(outcomes["dead"].value_counts())

outcomes.to_csv(f"{OUTPUT_DIR}/final/outcomes_w_dead.csv", index=None)

