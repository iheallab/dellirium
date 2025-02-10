#%%

# Import libraries

import pandas as pd
import numpy as np
import pickle
import os

from scipy.stats import shapiro, f_oneway, kruskal, mannwhitneyu
from statsmodels.stats.proportion import proportions_chisquare
import re

INTERNAL_DIR = '.../internal_delirium'
MIMIC_DIR = '.../mimiciv'
EICU_DIR = '.../eicu'

MAIN_DIR = '.../clinical_notes/main3'

DATA_DIR = f'{MAIN_DIR}/final_data'
ANALYSIS_DIR = f'{MAIN_DIR}/analyses/patient_char'

if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR)


outcomes = pd.read_csv(f'{DATA_DIR}/outcomes_w_dead.csv')
all_admissions = pd.read_csv(f'{DATA_DIR}/static.csv')
seq_data = pd.read_csv(f'{DATA_DIR}/clinical_data.csv')

# Load admissions

admissions_internal = pd.read_csv(f'{INTERNAL_DIR}/icustays.csv')
admissions_mimic = pd.read_csv(f'{MIMIC_DIR}/icustays.csv.gz', compression='gzip',
                       usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])
admissions_eicu = pd.read_csv(f'{EICU_DIR}/patient.csv.gz', compression='gzip',
                       usecols=['patientunitstayid', 'uniquepid', 'patienthealthsystemstayid', 'unitdischargeoffset'])

# Convert LOS to days

admissions_eicu["unitdischargeoffset"] = admissions_eicu["unitdischargeoffset"] / 1440
admissions_internal.drop("icu_los", axis=1, inplace=True)
admissions_internal["icu_los"] = (
    pd.to_datetime(admissions_internal["exit_datetime"])
    - pd.to_datetime(admissions_internal["enter_datetime"])
) / np.timedelta64(1, "h")
admissions_internal["icu_los"] = admissions_internal["icu_los"] / 24

# Merge all admissions

admissions_eicu = admissions_eicu.rename(
    {
        "patientunitstayid": "icustay_id",
        "patienthealthsystemstayid": "merged_enc_id",
        "uniquepid": "patient_deiden_id",
        "unitdischargeoffset": "icu_los",
    },
    axis=1,
)

admissions_mimic = admissions_mimic.rename(
    {
        "stay_id": "icustay_id",
        "subject_id": "patient_deiden_id",
        "los": "icu_los",
        "hadm_id": "merged_enc_id",
    },
    axis=1,
)

cols = ["icustay_id", "icu_los"]

admissions_eicu = admissions_eicu.loc[:, cols]
admissions_mimic = admissions_mimic.loc[:, cols]
admissions_internal = admissions_internal.loc[:, cols]

admissions_mimic["icustay_id"] = admissions_mimic["icustay_id"].astype(int)
admissions_eicu["icustay_id"] = admissions_eicu["icustay_id"].astype(int)
admissions_internal["icustay_id"] = admissions_internal["icustay_id"].astype(int)

admissions_mimic["icustay_id"] = "MIMIC_" + admissions_mimic["icustay_id"].astype(str)
admissions_eicu["icustay_id"] = "EICU_" + admissions_eicu["icustay_id"].astype(str)
admissions_internal["icustay_id"] = "internal_" + admissions_internal["icustay_id"].astype(str)

all_admissions_los = pd.concat(
    [admissions_eicu, admissions_mimic, admissions_internal], axis=0
).reset_index(drop=True)

all_admissions = all_admissions.merge(all_admissions_los, on="icustay_id", how="inner")

print(all_admissions.shape)
print(all_admissions.head())
print(all_admissions["icu_los"].isna().sum())

coma = outcomes[outcomes["coma"] == 1]
dead = outcomes[outcomes["dead"] == 1]

coma = coma["icustay_id"].unique()
dead = dead["icustay_id"].unique()

all_admissions["coma"] = 0
all_admissions.loc[(all_admissions["icustay_id"].isin(coma)), "coma"] = 1

all_admissions["died"] = 0
all_admissions.loc[(all_admissions["icustay_id"].isin(dead)), "died"] = 1

delirium = list(outcomes[outcomes["delirium"] == 1]["icustay_id"].unique())
nodelirium = list(outcomes[outcomes["delirium"] == 0]["icustay_id"].unique())
all_ids = delirium + nodelirium

print(all_admissions.head())
print(all_admissions["coma"].value_counts())
print(all_admissions["died"].value_counts())

cohorts = ["EICU", "MIMIC", "internal"]

groups = [["Overall", all_ids], ["No Delirium", nodelirium], ["Delirium", delirium]]

for cohort in cohorts:
    
    cohort_df = pd.DataFrame()
    
    count = 0
    
    for group in groups:
    
        admissions_cohort = all_admissions[all_admissions["icustay_id"].str.contains(cohort)]
        
        admissions_cohort = admissions_cohort[admissions_cohort["icustay_id"].isin(group[1])]
        
        seq_cohort = seq_data[seq_data["icustay_id"].str.contains(cohort)]
        seq_cohort = seq_cohort[seq_cohort["icustay_id"].isin(group[1])]


       # Numeric characteristics

        numeric_skew = ["age", "bmi", "cci", "icu_los"]

        admissions_cohort_num_skew = (
            admissions_cohort.loc[:, numeric_skew].describe().loc[["50%", "25%", "75%"]].round(1)
        )
        admissions_cohort_num_skew = admissions_cohort_num_skew.apply(
            lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=0
        )

        # Binary characteristics

        binary = [col for col in admissions_cohort.columns[5:22]] + ["coma", "died"]

        admissions_cohort_bin = pd.DataFrame(
            data={
                "perc": ((admissions_cohort.loc[:, binary].sum() / len(admissions_cohort)) * 100).round(1),
                "count": admissions_cohort.loc[:, binary].sum().astype("int"),
            }
        )
        admissions_cohort_bin = admissions_cohort_bin.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

        # Gender and race

        admissions_cohort_gender = pd.DataFrame(
            data={
                "perc": ((len(admissions_cohort[admissions_cohort["gender"] == "Female"]) / len(admissions_cohort)) * 100),
                "count": len(admissions_cohort[admissions_cohort["gender"] == "Female"]),
            },
            index=["Female"],
        )
        admissions_cohort_race_b = pd.DataFrame(
            data={
                "perc": ((len(admissions_cohort[admissions_cohort["race"] == "black"]) / len(admissions_cohort)) * 100),
                "count": len(admissions_cohort[admissions_cohort["race"] == "black"]),
            },
            index=["Black"],
        )
        admissions_cohort_race_w = pd.DataFrame(
            data={
                "perc": ((len(admissions_cohort[admissions_cohort["race"] == "white"]) / len(admissions_cohort)) * 100),
                "count": len(admissions_cohort[admissions_cohort["race"] == "white"]),
            },
            index=["White"],
        )
        admissions_cohort_race_o = pd.DataFrame(
            data={
                "perc": ((len(admissions_cohort[admissions_cohort["race"] == "other"]) / len(admissions_cohort)) * 100),
                "count": len(admissions_cohort[admissions_cohort["race"] == "other"]),
            },
            index=["Other"],
        )

        admissions_cohort_socio = pd.concat(
            [admissions_cohort_gender, admissions_cohort_race_b, admissions_cohort_race_w, admissions_cohort_race_o], axis=0
        ).round(1)

        admissions_cohort_socio = admissions_cohort_socio.apply(
            lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1
        )

        # Number of patients and admissions

        admissions_cohort_pat_count = pd.DataFrame(
            data=[
                len(admissions_cohort["icustay_id"].unique()),
            ],
            index=[
                "Number of ICU admissions",
            ],
        )
        
        # Sequential data statistics
        
        meds = [
            "Amiodarone",
            "Folic Acid",
            "Heparin Sodium",
            "Dexmedetomidine (Precedex)",
            "Propofol",
            "Fentanyl",
            "Digoxin (Lanoxin)",
            "Phenylephrine",
            "Norepinephrine",
            "Epinephrine",
            "Dopamine",
            "Vasopressin",
        ]

        seq_meds = seq_cohort[seq_cohort["variable"].isin(meds)]
        seq_all = seq_cohort[~seq_cohort["variable"].isin(meds)]
        
        if count == 0:
            
            seq_numeric_vars = list(seq_all["variable"].unique())

        seq_meds = seq_meds.drop_duplicates(subset=["icustay_id", "variable"])

        seq_meds = pd.DataFrame(
            data={
                "perc": (
                    (seq_meds["variable"].value_counts() / len(admissions_cohort)) * 100
                ).round(1),
                "count": seq_meds["variable"].value_counts().astype("int"),
            }
        )
        seq_meds = seq_meds.apply(lambda x: f"{int(x['count'])} ({x['perc']}%)", axis=1)

        seq_all = seq_all.groupby(by=["variable"])["value"].describe()

        seq_all = seq_all.reset_index()

        seq_all = seq_all.loc[:, ["variable", "50%", "25%", "75%"]].round(4)

        seq_all = seq_all.set_index("variable")

        seq_all = seq_all.apply(lambda x: f"{x['50%']} ({x['25%']}-{x['75%']})", axis=1)

        admissions_cohort_chars = pd.concat(
            [admissions_cohort_pat_count, admissions_cohort_num_skew, admissions_cohort_socio, admissions_cohort_bin, seq_all, seq_meds],
            axis=0,
        )
        
        admissions_cohort_chars = admissions_cohort_chars.rename({0: group[0]}, axis=1)
        
        cohort_df = pd.concat([cohort_df, admissions_cohort_chars], axis=1)
        
        count += 1

    
    # print(cohort_df)
        
    # Perform statistical tests
    admissions_cohort = all_admissions[all_admissions["icustay_id"].str.contains(cohort)]
    
    cohort_df["p-value"] = 'N/A'

    numerical_vars_static = ["age", "bmi", "cci"]
    
    nodel_char = admissions_cohort[admissions_cohort["icustay_id"].isin(nodelirium)]
    del_char = admissions_cohort[admissions_cohort["icustay_id"].isin(delirium)]

    for var in numerical_vars_static:

        stat, p_value = mannwhitneyu(nodel_char.loc[:, var].dropna(), del_char.loc[:, var].dropna())
        
        if p_value < 0.005:
            
            cohort_df.loc[var, "p-value"] = '< 0.005'
            
        else:
            
            cohort_df.loc[var, "p-value"] = f'{p_value}'
            
    
    seq_cohort = seq_data[seq_data["icustay_id"].str.contains(cohort)]
    nodel_char = seq_cohort[seq_cohort["icustay_id"].isin(nodelirium)]
    del_char = seq_cohort[seq_cohort["icustay_id"].isin(delirium)]
    
    # print(nodel_char["variable"].unique())
    # print(del_char["variable"].unique())

    for var in seq_numeric_vars:
        
        if len(nodel_char.loc[nodel_char["variable"] == var, "value"].dropna()) > 0 and len(del_char.loc[del_char["variable"] == var, "value"].dropna()) > 0:

            stat, p_value = mannwhitneyu(nodel_char.loc[nodel_char["variable"] == var, "value"].dropna(), del_char.loc[del_char["variable"] == var, "value"].dropna())

            if p_value < 0.005:

                cohort_df.loc[var, "p-value"] = '< 0.005'

            else:

                cohort_df.loc[var, "p-value"] = f'{p_value}'
            
    all_numeric_vars = numerical_vars_static + seq_numeric_vars
    

    # Extract categorical variables
    categorical_vars = [var for var in cohort_df.index.tolist() if var not in all_numeric_vars]

    observed_table = cohort_df.loc[
        categorical_vars, ["Overall", "No Delirium", "Delirium"]
    ].applymap(lambda x: int(re.search(r"(\d+)", str(x)).group(1)))


    for var in categorical_vars:

        count1 = observed_table.loc[var, "No Delirium"]
        nobs1 = cohort_df.loc["Number of ICU admissions", "No Delirium"]

        count2 = observed_table.loc[var, "Delirium"]
        nobs2 = cohort_df.loc["Number of ICU admissions", "Delirium"]

        counts = np.array([count1, count2])
        nobs = np.array([nobs1, nobs2])

        chisq, pvalue, table = proportions_chisquare(counts, nobs)
        
        if p_value < 0.005:
            
            cohort_df.loc[var, "p-value"] = '< 0.005'
            
        else:
            
            cohort_df.loc[var, "p-value"] = f'{p_value}'
                
    print(cohort_df)
    cohort_df.to_csv(f"{ANALYSIS_DIR}/{cohort}_charac.csv")
