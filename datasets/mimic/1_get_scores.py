# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

if not os.path.exists(f"{OUTPUT_DIR}/raw"):
    os.makedirs(f"{OUTPUT_DIR}/raw")

ditems = pd.read_csv(f'{DATA_DIR}/d_items.csv.gz', compression='gzip')

cam = [228300, 228301, 228302, 228303, 228334, 228335, 228336, 228337, 229324, 229325, 229326]
rass = [228096, 228299]
gcs = [220739, 223900, 223901]

all_scores = cam + rass + gcs

ch = []

for chunk in tqdm(pd.read_csv(f'{DATA_DIR}/chartevents.csv.gz', compression='gzip', chunksize=1000000,
                usecols = ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'value', 'valuenum'])):
    chunk = chunk.loc[chunk.charttime.notna()]
    chunk = chunk[chunk['itemid'].isin(all_scores)]
    ch.append(chunk)
    

del chunk
print ('Done')
ch = pd.concat(ch)
print ('Done')

map_scores = dict()

for i in range(len(all_scores)):
    map_scores[all_scores[i]] = ditems[ditems['itemid'] == all_scores[i]]['label'].values[0].lower()

print(map_scores)

ch['label'] = ch['itemid'].map(map_scores)

print(ch.head())

ch.to_csv(f"{OUTPUT_DIR}/raw/scores.csv", index=False)