# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

MAIN_DIR = '.../clinical_notes/main3'

ANALYSIS_DIR = f'{MAIN_DIR}/analyses/temporal'

if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR)

EICU_DIR = '.../clinical_notes/main3/datasets/eicu'
MIMIC_DIR = '.../clinical_notes/main3/datasets/mimic'
INTERNAL_DIR = '.../clinical_notes/main3/datasets/internal'

# EICU

scores = pd.read_csv(
    f"{EICU_DIR}/raw/scores.csv",
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

keep_ids = brain_status.loc[(brain_status["interval"] <= 1), "stay_id"].unique()

brain_status = brain_status[brain_status["stay_id"].isin(keep_ids)]

drop_ids = brain_status.loc[((brain_status["interval"] <= 1) & (
    brain_status["brain_status"] != "normal")), "stay_id"].unique()

brain_status = brain_status[~brain_status["stay_id"].isin(drop_ids)]

brain_status["coma"] = 0
brain_status.loc[brain_status["brain_status"] == "coma", "coma"] = 1

brain_status["delirium"] = 0
brain_status.loc[brain_status["brain_status"] == "delirium", "delirium"] = 1


delirium_eicu = brain_status[brain_status["brain_status"] == "delirium"].sort_values(by=["stay_id", "interval"])

delirium_eicu = delirium_eicu.drop_duplicates(subset=["stay_id"], keep="first").reset_index(drop=True)
delirium_eicu["days"] = ((delirium_eicu["interval"] // 2) + 1).astype(int)
print(delirium_eicu.iloc[:20])
print(delirium_eicu["days"].describe())


# MIMIC

brain_status = pd.read_csv(f'{MIMIC_DIR}/raw/scores.csv')

brain_status = brain_status.sort_values(by=["stay_id", "charttime"])

icustays = pd.read_csv(f'.../mimiciv/icustays.csv.gz',
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

delirium_mimic = brain_status[brain_status["brain_status"] == "delirium"].sort_values(by=["stay_id", "interval"])

delirium_mimic = delirium_mimic.drop_duplicates(subset=["stay_id"], keep="first").reset_index(drop=True)
delirium_mimic["days"] = ((delirium_mimic["interval"] // 2) + 1).astype(int)
print(delirium_mimic.iloc[:20])
print(delirium_mimic["days"].describe())


# internal

brain_status = pd.read_csv(f'.../internal_delirium/seq.csv')

brain_status = brain_status[(brain_status["variable"].str.contains("_gcs_")) | (brain_status["variable"].str.contains("_cam_")) | (brain_status["variable"].str.contains("_rass_"))]

brain_status["variable"] = brain_status["variable"].map({"score_gcs_total": "GCS", "score_cam_score": "CAM", "score_rass_score": "RASS"})

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

keep_ids = brain_status.loc[(brain_status["interval"] <= 1), "icustay_id"].unique()

brain_status = brain_status[brain_status["icustay_id"].isin(keep_ids)]

drop_ids = brain_status.loc[((brain_status["interval"] <= 1) & (
    brain_status["brain_status"] != "normal")), "icustay_id"].unique()

brain_status = brain_status[~brain_status["icustay_id"].isin(drop_ids)]

brain_status["coma"] = 0
brain_status.loc[brain_status["brain_status"] == "coma", "coma"] = 1

brain_status["delirium"] = 0
brain_status.loc[brain_status["brain_status"] == "delirium", "delirium"] = 1

delirium_internal = brain_status[brain_status["brain_status"] == "delirium"].sort_values(by=["icustay_id", "interval"])

delirium_internal = delirium_internal.drop_duplicates(subset=["icustay_id"], keep="first").reset_index(drop=True)
delirium_internal["days"] = ((delirium_internal["interval"] // 2) + 1).astype(int)
print(delirium_internal.iloc[:20])
print(delirium_internal["days"].describe())


# Merge all

delirium_eicu = delirium_eicu.rename({"stay_id": "icustay_id"}, axis=1)
delirium_mimic = delirium_mimic.rename({"stay_id": "icustay_id"}, axis=1)

cols = ["icustay_id", "delirium", "days"]

delirium_internal = delirium_internal.loc[:,cols]
delirium_eicu = delirium_eicu.loc[:,cols]
delirium_mimic = delirium_mimic.loc[:,cols]

delirium_mimic["icustay_id"] = delirium_mimic["icustay_id"].astype(int)
delirium_eicu["icustay_id"] = delirium_eicu["icustay_id"].astype(int)
delirium_internal["icustay_id"] = delirium_internal["icustay_id"].astype(int)

delirium_mimic["icustay_id"] = "MIMIC_" + delirium_mimic["icustay_id"].astype(str)
delirium_eicu["icustay_id"] = "EICU_" + delirium_eicu["icustay_id"].astype(str)
delirium_internal["icustay_id"] = "internal_" + delirium_internal["icustay_id"].astype(str)

delirium = pd.concat([delirium_eicu, delirium_mimic, delirium_internal])

print(delirium.iloc[:20])
print(delirium["days"].describe())
print(delirium.shape)

delirium.to_csv(f"{ANALYSIS_DIR}/delirium_times.csv", index=None)
