# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

if not os.path.exists(f"{OUTPUT_DIR}/raw"):
    os.makedirs(f"{OUTPUT_DIR}/raw")
    
brain_status = pd.read_csv(f'{DATA_DIR}/seq.csv')

brain_status = brain_status[(brain_status["variable"].str.contains("_gcs_")) | (brain_status["variable"].str.contains("_cam_")) | (brain_status["variable"].str.contains("_rass_"))]

brain_status["variable"] = brain_status["variable"].map({"score_gcs_total": "GCS", "score_cam_score": "CAM", "score_rass_score": "RASS"})

print(brain_status.head())
print(brain_status["variable"].value_counts())

brain_status = brain_status[brain_status["hours"] >= 0]

brain_status = brain_status.sort_values(by=["icustay_id", "hours"]).reset_index(drop=True)

brain_status["interval"] = (brain_status["hours"] // 12).astype(int)

brain_status.drop("hours", axis=1, inplace=True)

cam = brain_status[brain_status["variable"] == "CAM"]
rass = brain_status[brain_status["variable"] == "RASS"]
gcs = brain_status[brain_status["variable"] == "GCS"]

cam = cam.rename({"value": "cam"}, axis=1)
rass = rass.rename({"value": "rass"}, axis=1)
gcs = gcs.rename({"value": "gcs"}, axis=1)

cam = cam.groupby(by=["icustay_id", "interval"])["cam"].max().reset_index()
gcs = gcs.groupby(by=["icustay_id", "interval"])["gcs"].min().reset_index()
rass = rass.groupby(by=["icustay_id", "interval"])["rass"].min().reset_index()


brain_status = gcs.merge(cam, on=["icustay_id", "interval"], how="outer")
brain_status = brain_status.merge(
    rass, on=["icustay_id", "interval"], how="outer")


brain_status["gcs"] = brain_status.groupby(
    "icustay_id")["gcs"].fillna(method="ffill")
brain_status["cam"] = brain_status.groupby(
    "icustay_id")["cam"].fillna(method="ffill")
brain_status["rass"] = brain_status.groupby(
    "icustay_id")["rass"].fillna(method="ffill")

brain_status["gcs"] = brain_status['gcs'].fillna(15)
brain_status["cam"] = brain_status['cam'].fillna(0)
brain_status["rass"] = brain_status['rass'].fillna(999)

brain_status['brain_status'] = 'normal'

condition1 = brain_status['rass'] < -3
condition2 = ((brain_status['rass'] == -3) | (brain_status['rass'] == 999)) & (brain_status['gcs'] <= 8)
condition5 = (brain_status['rass'] >= -3) & (brain_status['cam'] == 1)


brain_status.loc[condition1, 'brain_status'] = 'coma'
brain_status.loc[condition2, 'brain_status'] = 'coma'
brain_status.loc[condition5, 'brain_status'] = 'delirium'

print(brain_status["brain_status"].value_counts())

keep_ids = brain_status.loc[(brain_status["interval"] <= 1), "icustay_id"].unique()

brain_status = brain_status[brain_status["icustay_id"].isin(keep_ids)]

drop_ids = brain_status.loc[((brain_status["interval"] <= 1) & (
    brain_status["brain_status"] != "normal")), "icustay_id"].unique()

brain_status = brain_status[~brain_status["icustay_id"].isin(drop_ids)]

brain_status["coma"] = 0
brain_status.loc[brain_status["brain_status"] == "coma", "coma"] = 1

brain_status["delirium"] = 0
brain_status.loc[brain_status["brain_status"] == "delirium", "delirium"] = 1

coma = brain_status.groupby("icustay_id")["coma"].max().reset_index()
delirium = brain_status.groupby("icustay_id")["delirium"].max().reset_index()

print(coma["coma"].value_counts())
print(delirium["delirium"].value_counts())

brain_status = coma.merge(delirium, on="icustay_id", how="inner")

print(brain_status["coma"].value_counts())
print(brain_status["delirium"].value_counts())

brain_status.to_csv(f'{OUTPUT_DIR}/raw/brain_status.csv', index=False)

