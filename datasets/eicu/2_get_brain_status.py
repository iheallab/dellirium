# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

scores = pd.read_csv(
    f"{OUTPUT_DIR}/raw/scores.csv",
    usecols=[
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ],
)

cam = scores[scores["nursingchartcelltypevalname"] == "Delirium Score"]
cam = cam[(cam["nursingchartvalue"] == "Yes") | (cam["nursingchartvalue"] == "No")]
cam["nursingchartvalue"] = cam["nursingchartvalue"].map({"Yes": 1, "No": 0})
cam["nursingchartcelltypevalname"] = "CAM"

gcs = scores[scores["nursingchartcelltypevalname"] == "GCS Total"]
gcs = gcs[gcs["nursingchartvalue"] != "Unable to score due to medication"]
gcs = gcs.dropna(subset=["nursingchartvalue"])
gcs["nursingchartvalue"] = gcs["nursingchartvalue"].astype(int)
gcs = gcs[(gcs["nursingchartvalue"] <= 15) & (gcs["nursingchartvalue"] >= 0)]
gcs["nursingchartcelltypevalname"] = "GCS"

rass = scores[scores["nursingchartcelltypevalname"] == "Sedation Score"]
rass["nursingchartvalue"] = rass["nursingchartvalue"].astype(int)
rass = rass[(rass["nursingchartvalue"] <= 4) & (rass["nursingchartvalue"] >= -5)]
rass["nursingchartcelltypevalname"] = "RASS"

del scores

brain_status = pd.concat([cam, gcs, rass], axis=0)

del cam, gcs, rass

brain_status = brain_status.rename(
    {
        "patientunitstayid": "stay_id",
        "nursingchartoffset": "hours",
        "nursingchartcelltypevalname": "label",
        "nursingchartvalue": "valuenum",
    },
    axis=1,
)

brain_status["hours"] = brain_status["hours"] / 60

brain_status = brain_status[brain_status["hours"] >= 0]

brain_status = brain_status.sort_values(by=["stay_id", "hours"]).reset_index(drop=True)

brain_status["interval"] = (brain_status["hours"] // 12).astype(int)

brain_status.drop("hours", axis=1, inplace=True)

cam = brain_status[brain_status["label"] == "CAM"]
rass = brain_status[brain_status["label"] == "RASS"]
gcs = brain_status[brain_status["label"] == "GCS"]

cam = cam.rename({"valuenum": "cam"}, axis=1)
rass = rass.rename({"valuenum": "rass"}, axis=1)
gcs = gcs.rename({"valuenum": "gcs"}, axis=1)

cam = cam.groupby(by=["stay_id", "interval"])["cam"].max().reset_index()
gcs = gcs.groupby(by=["stay_id", "interval"])["gcs"].min().reset_index()
rass = rass.groupby(by=["stay_id", "interval"])["rass"].min().reset_index()


brain_status = gcs.merge(cam, on=["stay_id", "interval"], how="outer")
brain_status = brain_status.merge(
    rass, on=["stay_id", "interval"], how="outer")


brain_status["gcs"] = brain_status.groupby(
    "stay_id")["gcs"].fillna(method="ffill")
brain_status["cam"] = brain_status.groupby(
    "stay_id")["cam"].fillna(method="ffill")
brain_status["rass"] = brain_status.groupby(
    "stay_id")["rass"].fillna(method="ffill")

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

keep_ids = brain_status.loc[(brain_status["interval"] <= 1), "stay_id"].unique()

brain_status = brain_status[brain_status["stay_id"].isin(keep_ids)]

drop_ids = brain_status.loc[((brain_status["interval"] <= 1) & (
    brain_status["brain_status"] != "normal")), "stay_id"].unique()

brain_status = brain_status[~brain_status["stay_id"].isin(drop_ids)]

brain_status["coma"] = 0
brain_status.loc[brain_status["brain_status"] == "coma", "coma"] = 1

brain_status["delirium"] = 0
brain_status.loc[brain_status["brain_status"] == "delirium", "delirium"] = 1

coma = brain_status.groupby("stay_id")["coma"].max().reset_index()
delirium = brain_status.groupby("stay_id")["delirium"].max().reset_index()

print(coma["coma"].value_counts())
print(delirium["delirium"].value_counts())

brain_status = coma.merge(delirium, on="stay_id", how="inner")

print(brain_status["coma"].value_counts())
print(brain_status["delirium"].value_counts())

brain_status.to_csv(f'{OUTPUT_DIR}/raw/brain_status.csv', index=False)

