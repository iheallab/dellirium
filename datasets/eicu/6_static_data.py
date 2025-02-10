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

diag_dict = pd.read_csv(f"{DATA_DIR}/diagnosis.csv.gz", compression="gzip")

diag_dict = diag_dict[diag_dict["patientunitstayid"].isin(icustay_ids)]

diag_dict = diag_dict.dropna(subset=["icd9code"])

diag_codes = diag_dict["icd9code"].unique()

diag_dict = diag_dict[diag_dict["diagnosisoffset"] <= 60]


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

diag_dict = diag_dict[diag_dict["icd9code"].isin(comob)]

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

diag_dict["comob"] = diag_dict["icd9code"].replace(inverted_dict)

diag_dict.drop_duplicates(subset=["patientunitstayid", "comob"], inplace=True)

diag_dict = diag_dict[["patientunitstayid", "comob"]]

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

diag_dict["poa"] = diag_dict["comob"].replace(cci_weights)

diag_dict = diag_dict.set_index(["patientunitstayid", "comob"])

multi_index = diag_dict.index
data = diag_dict.values.flatten()
diag_dict = pd.Series(data, index=multi_index)
diag_dict = diag_dict.unstack("comob")

diag_dict = diag_dict.reset_index()

diag_dict.fillna(0, inplace=True)

diag_dict = diag_dict.set_index(["patientunitstayid"])

diag_dict["cci"] = diag_dict.sum(axis=1)

diag_dict = diag_dict.reset_index()

comorbid = list(map_comob.keys())

cols = diag_dict.columns[1:-1].tolist()

missing = [code for code in comorbid if code not in cols]

for code in missing:
    diag_dict[code] = 0


def convert_to_one(value):
    return 1 if value > 1 else value


diag_dict[comorbid] = diag_dict[comorbid].applymap(convert_to_one)

cols = ["patientunitstayid"] + comorbid + ["cci"]

diag_dict = diag_dict[cols]

admissions = pd.read_csv(
    f"{DATA_DIR}/patient.csv.gz", compression="gzip",
    usecols=[
        "patientunitstayid",
        "uniquepid",
        "gender",
        "age",
        "ethnicity",
        "admissionheight",
        "admissionweight",
    ],
)

admissions["bmi"] = admissions["admissionweight"] / (
    (admissions["admissionheight"] / 100) ** 2
)

admissions.drop("admissionweight", axis=1, inplace=True)
admissions.drop("admissionheight", axis=1, inplace=True)

static_eicu = admissions.merge(diag_dict, how="outer", on=["patientunitstayid"])

static_eicu = static_eicu[static_eicu["patientunitstayid"].isin(icustay_ids)]

all_races = list(static_eicu["ethnicity"].unique())
white = [race for race in all_races if race == "Caucasian"]
black = [race for race in all_races if race == "African American"]
other = [race for race in all_races if race not in white and race not in black]

static_eicu = static_eicu.rename({"ethnicity": "race"}, axis=1)

static_eicu["race"] = static_eicu["race"].replace(white, "white")
static_eicu["race"] = static_eicu["race"].replace(black, "black")
static_eicu["race"] = static_eicu["race"].replace(other, "other")

static_eicu["age"] = static_eicu["age"].replace({"> 89": 89})
static_eicu["age"] = static_eicu["age"].astype(float)

static_eicu = static_eicu.rename({"patientunitstayid": "stay_id"}, axis=1)
static_eicu.drop("uniquepid", axis=1, inplace=True)

print(static_eicu.head())
print(static_eicu.shape)
print(static_eicu.columns)


static_eicu.to_csv(f"{OUTPUT_DIR}/intermediate/static.csv", index=False)

