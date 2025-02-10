# Import libraries

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import math

from variables import DATA_DIR, OUTPUT_DIR

# Build MIMIC static data
admissions = pd.read_csv("%s/intermediate/icustays.csv" % OUTPUT_DIR)
icustay_ids = admissions["stay_id"].tolist()

diagnoses = pd.read_csv("%s/diagnoses_icd.csv.gz" % DATA_DIR, compression="gzip")

diagnoses = diagnoses[diagnoses["icd_version"] == 9]

diagnoses = diagnoses[diagnoses["subject_id"].isin(admissions["subject_id"].unique())]

diag_dict = pd.read_csv("%s/d_icd_diagnoses.csv.gz" % DATA_DIR, compression="gzip")

diag_dict = diag_dict[diag_dict["icd_version"] == 9]

diag_codes = diag_dict["icd_code"].unique()

aids = (
    [s for s in diag_codes if s.startswith("042")]
    + [s for s in diag_codes if s.startswith("043")]
    + [s for s in diag_codes if s.startswith("044")]
)
cancer = (
    list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(140, 173)
            ]
        )
    )
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(174, 196)
            ]
        )
    )
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(200, 209)
            ]
            + [s for s in diag_codes if s.startswith("2386")]
        )
    )
)
cervasc = list(
    itertools.chain(
        *[[s for s in diag_codes if s.startswith(str(sub))] for sub in range(430, 439)]
    )
)
chf = (
    [
        "39891",
        "40201",
        "40211",
        "40291",
        "40401",
        "40403",
        "40411",
        "40413",
        "40491",
        "40493",
    ]
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(4254, 4260)
            ]
        )
    )
    + [s for s in diag_codes if s.startswith("428")]
)
copd = ["4168", "4169", "5064", "5081", "5088"] + list(
    itertools.chain(
        *[[s for s in diag_codes if s.startswith(str(sub))] for sub in range(490, 506)]
    )
)
demen = [s for s in diag_codes if s.startswith("290")] + ["2941", "3312"]
diab_wo_comp = list(
    itertools.chain(
        *[
            [s for s in diag_codes if s.startswith(str(sub))]
            for sub in range(2500, 2504)
        ]
    )
) + ["2508", "2509"]
diab_w_comp = list(
    itertools.chain(
        *[
            [s for s in diag_codes if s.startswith(str(sub))]
            for sub in range(2504, 2508)
        ]
    )
)
mi = [s for s in diag_codes if s.startswith("410")] + [
    s for s in diag_codes if s.startswith("412")
]
met_car = list(
    itertools.chain(
        *[[s for s in diag_codes if s.startswith(str(sub))] for sub in range(196, 200)]
    )
)
mild_liv = (
    [
        "07022",
        "07023",
        "07032",
        "07033",
        "07044",
        "07054",
        "0706",
        "0709",
        "5733",
        "5734",
        "5738",
        "5739",
        "V427",
    ]
    + [s for s in diag_codes if s.startswith("570")]
    + [s for s in diag_codes if s.startswith("571")]
)
mod_sev_liv = list(
    itertools.chain(
        *[
            [s for s in diag_codes if s.startswith(str(sub))]
            for sub in range(4560, 4563)
        ]
    )
) + list(
    itertools.chain(
        *[
            [s for s in diag_codes if s.startswith(str(sub))]
            for sub in range(5722, 5729)
        ]
    )
)
par_hem = (
    ["3341", "3449"]
    + [s for s in diag_codes if s.startswith("342")]
    + [s for s in diag_codes if s.startswith("343")]
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(3440, 3447)
            ]
        )
    )
)
pep_ulc = list(
    itertools.chain(
        *[[s for s in diag_codes if s.startswith(str(sub))] for sub in range(531, 534)]
    )
)
peri_vasc = (
    ["0930", "4373", "4471", "5571", "5579", "V434"]
    + [s for s in diag_codes if s.startswith("440")]
    + [s for s in diag_codes if s.startswith("441")]
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(4431, 4440)
            ]
        )
    )
)
renal = (
    [
        "40301",
        "40311",
        "40391",
        "40402",
        "40403",
        "40412",
        "40413",
        "40492",
        "40493",
        "5880",
        "V420",
        "V451",
    ]
    + [s for s in diag_codes if s.startswith("582")]
    + [s for s in diag_codes if s.startswith("585")]
    + [s for s in diag_codes if s.startswith("586")]
    + [s for s in diag_codes if s.startswith("V56")]
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(5830, 5838)
            ]
        )
    )
)
rheu = (
    ["4465", "7148"]
    + [s for s in diag_codes if s.startswith("725")]
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(7100, 7105)
            ]
        )
    )
    + list(
        itertools.chain(
            *[
                [s for s in diag_codes if s.startswith(str(sub))]
                for sub in range(7140, 7143)
            ]
        )
    )
)

comob = (
    aids
    + cancer
    + cervasc
    + chf
    + copd
    + demen
    + diab_w_comp
    + diab_wo_comp
    + mi
    + met_car
    + mild_liv
    + mod_sev_liv
    + par_hem
    + pep_ulc
    + peri_vasc
    + renal
    + rheu
)

diagnoses = diagnoses[diagnoses["icd_code"].isin(comob)]

map_comob = {
    "aids": aids,
    "cancer": cancer,
    "cerebrovascular": cervasc,
    "chf": chf,
    "copd": copd,
    "dementia": demen,
    "diabetes_wo_comp": diab_wo_comp,
    "diabetes_w_comp": diab_w_comp,
    "m_i": mi,
    "metastatic_carc": met_car,
    "mild_liver_dis": mild_liv,
    "mod_sev_liver_dis": mod_sev_liv,
    "par_hem": par_hem,
    "pep_ulc": pep_ulc,
    "peri_vascular": peri_vasc,
    "renal_dis": renal,
    "rheu": rheu,
}

inverted_dict = {}
for key, values in map_comob.items():
    for value in values:
        inverted_dict[value] = key

diagnoses["comob"] = diagnoses["icd_code"].replace(inverted_dict)

diagnoses.drop_duplicates(subset=["hadm_id", "comob"], inplace=True)

diagnoses = diagnoses[["subject_id", "hadm_id", "comob"]]

cci_weights = {
    "aids": 6,
    "cancer": 2,
    "cerebrovascular": 1,
    "chf": 1,
    "copd": 1,
    "dementia": 1,
    "diabetes_w_comp": 2,
    "diabetes_wo_comp": 1,
    "metastatic_carc": 6,
    "m_i": 1,
    "mild_liver_dis": 1,
    "mod_sev_liver_dis": 3,
    "par_hem": 2,
    "pep_ulc": 1,
    "peri_vascular": 1,
    "renal_dis": 2,
    "rheu": 1,
}

diagnoses["poa"] = diagnoses["comob"].replace(cci_weights)

diagnoses = diagnoses.set_index(["subject_id", "hadm_id", "comob"])

multi_index = diagnoses.index
data = diagnoses.values.flatten()
diagnoses = pd.Series(data, index=multi_index)
diagnoses = diagnoses.unstack("comob")

diagnoses = diagnoses.reset_index()

diagnoses.fillna(0, inplace=True)

diagnoses = diagnoses.set_index(["subject_id", "hadm_id"])

diagnoses["cci"] = diagnoses.sum(axis=1)

diagnoses = diagnoses.reset_index()

bmi = pd.read_csv("%s/intermediate/height_weight.csv" % OUTPUT_DIR)
bmi = pd.concat([bmi, pd.read_csv("%s/intermediate/height_weight_conv.csv" % OUTPUT_DIR)])


def calc_bmi(group):
    weight = group.loc[group["itemid"] == 226512, "valuenum"].values
    height = group.loc[group["itemid"] == 226730, "valuenum"].values
    if len(weight) > 0 and len(height) > 0 and weight[0] != 0 and height[0] != 0:
        bmi = weight[0] / ((height[0] / 100) ** 2)
    else:
        bmi = np.nan

    new_group = pd.DataFrame(
        {
            "subject_id": group["subject_id"].values[0],
            "hadm_id": group["hadm_id"].values[0],
            "stay_id": group["stay_id"].values[0],
            "bmi": bmi,
        },
        index=[0],
    )
    return new_group


bmi = bmi.groupby("stay_id").apply(calc_bmi)

bmi.dropna(subset=["bmi"], inplace=True)
bmi.reset_index(drop=True, inplace=True)

patients = pd.read_csv("%s/patients.csv.gz" % DATA_DIR, compression="gzip")

patients = patients[["subject_id", "gender", "anchor_age"]]

admissions = pd.read_csv(
    "%s/admissions.csv.gz" % DATA_DIR, compression="gzip", usecols=["subject_id", "race"]
).drop_duplicates(subset=["subject_id"])

patients = patients.merge(admissions, on=["subject_id"], how="inner")

admissions = pd.read_csv(
    "%s/intermediate/icustays.csv" % OUTPUT_DIR, usecols=["subject_id", "stay_id"]
)

static_mimic = admissions.merge(patients, how="inner", on=["subject_id"])

bmi = bmi[bmi["stay_id"].isin(icustay_ids)]

bmi = bmi[["hadm_id", "stay_id", "bmi"]]

static_mimic = static_mimic.merge(bmi, how="outer", on=["stay_id"])

static_mimic = static_mimic.merge(diagnoses, how="outer", on=["subject_id", "hadm_id"])

static_mimic = static_mimic[static_mimic["stay_id"].isin(icustay_ids)]

static_mimic.drop("subject_id", axis=1, inplace=True)

all_races = list(static_mimic["race"].unique())
white = [race for race in all_races if "WHITE" in race]
black = [race for race in all_races if "BLACK" in race]
other = [race for race in all_races if race not in white and race not in black]

static_mimic["race"] = static_mimic["race"].replace(white, "white")
static_mimic["race"] = static_mimic["race"].replace(black, "black")
static_mimic["race"] = static_mimic["race"].replace(other, "other")

static_mimic["gender"] = static_mimic["gender"].map({"F": "Female", "M": "Male"})

print(static_mimic.head())
print(static_mimic.shape)


static_mimic.to_csv(f"{OUTPUT_DIR}/intermediate/static.csv", index=False)

