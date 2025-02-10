# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from variables import DATA_DIR, OUTPUT_DIR

brain_status = pd.read_csv(f'{OUTPUT_DIR}/raw/scores.csv')

brain_status = brain_status.sort_values(by=["stay_id", "charttime"])

print(brain_status["label"].value_counts())

icustays = pd.read_csv(f'{DATA_DIR}/icustays.csv.gz',
                       compression='gzip', usecols=['stay_id', 'intime'])

brain_status = brain_status.merge(icustays, on="stay_id", how="inner")

brain_status["charttime"] = pd.to_datetime(brain_status["charttime"])
brain_status["intime"] = pd.to_datetime(brain_status["intime"])

brain_status["hours"] = (brain_status["charttime"] -
                         brain_status["intime"]) / np.timedelta64(1, "h")
brain_status["interval"] = (brain_status["hours"] // 12).astype(int)
brain_status["hours"] = brain_status["hours"].astype(int)

gcs = brain_status[brain_status["label"].str.contains(r'\bgcs\b', case=False)]
cam = brain_status[brain_status["label"].str.contains(
    r'\bcam-icu\b', case=False)]
rass = brain_status[brain_status["label"].str.contains(
    r'\brichmond\b', case=False)]

cam.loc[cam['value'].str.contains(
    "unable to assess", case=False), 'valuenum'] = -1
cam.loc[cam['value'].str.contains(r'\byes\b', case=False), 'valuenum'] = 1
cam.loc[cam['value'].str.contains(r'\bno\b', case=False), 'valuenum'] = 0

cam = cam[['stay_id', 'hours', 'label', 'valuenum']]

cam = cam.pivot_table(values="valuenum", index=[
                      "stay_id", "hours"], columns="label", aggfunc='max')

cam = cam.reset_index()

cam.insert(0, 'cam', np.nan)

cam.loc[((cam['cam-icu ms change'] == 1) & (cam['cam-icu inattention'] == 1)) & ((cam['cam-icu disorganized thinking']
                                                                                  == 1) | (cam['cam-icu altered loc'] == 1) | (cam['cam-icu rass loc'] == 1)), 'cam'] = 1
cam.loc[(cam['cam-icu ms change'] == 0) | (cam['cam-icu inattention'] == 0) | (cam['cam-icu disorganized thinking']
                                                                               == 0) | (cam['cam-icu altered loc'] == 0) | (cam['cam-icu rass loc'] == 0), 'cam'] = 0
cam.loc[(cam['cam-icu ms change'] == -1) | (cam['cam-icu inattention'] == -1) | (cam['cam-icu disorganized thinking']
                                                                                 == -1) | (cam['cam-icu altered loc'] == -1) | (cam['cam-icu rass loc'] == -1), 'cam'] = -1

cam["interval"] = (cam["hours"] // 12).astype(int)

cam = cam.groupby(by=["stay_id", "interval"])["cam"].max().reset_index()


gcs = gcs[['stay_id', 'hours', 'label', 'valuenum']]

gcs = gcs.pivot_table(values="valuenum", index=[
                      "stay_id", "hours"], columns="label", aggfunc='min')

gcs = gcs.reset_index()

gcs.dropna(inplace=True)

gcs['gcs'] = gcs['gcs - eye opening'] + gcs['gcs - verbal response'] + gcs['gcs - motor response']

gcs["interval"] = (gcs["hours"] // 12).astype(int)

gcs = gcs.groupby(by=["stay_id", "interval"])["gcs"].min().reset_index()


rass = rass[['stay_id', 'hours', 'label', 'valuenum']]

rass = rass.pivot_table(values="valuenum", index=[
                        "stay_id", "hours"], columns="label", aggfunc='min')

rass = rass.rename({'richmond-ras scale': 'rass'}, axis=1)

rass = rass.reset_index()

rass["interval"] = (rass["hours"] // 12).astype(int)

rass = rass.groupby(by=["stay_id", "interval"])["rass"].min().reset_index()


brain_status = gcs.merge(cam, on=["stay_id", "interval"], how="outer")
brain_status = brain_status.merge(
    rass, on=["stay_id", "interval"], how="outer")

# cam = cam.groupby("stay_id")["cam"].max().reset_index()

# print(cam["cam"].value_counts())
# print(len(cam))
# print(len(cam["stay_id"].unique()))


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

