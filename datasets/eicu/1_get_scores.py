# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

if not os.path.exists(f"{OUTPUT_DIR}/raw"):
    os.makedirs(f"{OUTPUT_DIR}/raw")
    
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

for chunk in pd.read_csv(f"{DATA_DIR}/nurseCharting.csv.gz", chunksize=chunk_size, compression="gzip", usecols=cols):

    chunk = chunk[chunk["nursingchartcelltypecat"] == "Scores"]

    chunk.to_csv(f"{OUTPUT_DIR}/raw/scores.csv", mode="a", index=False, header=not header_written)

    print(f"Processed chunk {count}")
    count += 1

    header_written = True
    
