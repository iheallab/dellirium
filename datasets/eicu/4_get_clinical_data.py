# Import libraries
import pandas as pd
import numpy as np
import pickle
import os

from variables import OUTPUT_DIR, DATA_DIR

# Get ICU admissions from cohort

admissions = pd.read_csv(f"{OUTPUT_DIR}/intermediate/icustays.csv")

icustay_ids = admissions["stay_id"].unique()
    
# Extract vitals

cols = [
    "patientunitstayid",
    "nursingchartoffset",
    "nursingchartcelltypecat",
    "nursingchartcelltypevalname",
    "nursingchartvalue",
]

use_vitals = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 Saturation",
    "Non-Invasive BP Diastolic",
    "Non-Invasive BP Systolic",
    "Temperature (C)",
    "Temperature (F)",
    "End Tidal CO2",
]

# Specify the chunk size (number of rows to read at a time)
chunk_size = 1000000
chunks = []
header_written = False


count = 1

for chunk in pd.read_csv(f"{DATA_DIR}/nurseCharting.csv.gz", compression="gzip", chunksize=chunk_size, usecols=cols):

    chunk = chunk[chunk["nursingchartcelltypevalname"].isin(use_vitals)]
    
    chunk = chunk[chunk["patientunitstayid"].isin(icustay_ids)]

    chunk.to_csv(f"{OUTPUT_DIR}/intermediate/vitals.csv", mode="a", index=False, header=not header_written)

    print(f"Processed chunk {count}")
    count += 1

    header_written = True
    

    
# Extract scores

cols = [
    "patientunitstayid",
    "nursingchartoffset",
    "nursingchartcelltypecat",
    "nursingchartcelltypevalname",
    "nursingchartvalue",
]

# Specify the chunk size (number of rows to read at a time)
chunk_size = 1000000
chunks = []
header_written = False


count = 1

for chunk in pd.read_csv(f"{DATA_DIR}/nurseCharting.csv.gz", compression="gzip", chunksize=chunk_size, usecols=cols):

    chunk = chunk[chunk["nursingchartcelltypecat"] == "Scores"]
    
    chunk = chunk[chunk["patientunitstayid"].isin(icustay_ids)]

    chunk.to_csv(f"{OUTPUT_DIR}/intermediate/scores.csv", mode="a", index=False, header=not header_written)

    print(f"Processed chunk {count}")
    count += 1

    header_written = True


