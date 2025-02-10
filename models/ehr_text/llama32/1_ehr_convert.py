# Import libraries

import pandas as pd
import numpy as np
import h5py
from nltk.tokenize import word_tokenize

from variables import DATA_DIR, MODEL_DIR

import os
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

seq = pd.read_csv(f"{DATA_DIR}/clinical_data.csv")

print(len(seq["icustay_id"].unique()))

vitals_feat = ['Respiratory Rate', 'Heart Rate', 'Non Invasive Blood Pressure diastolic',
          'Non Invasive Blood Pressure systolic', 'O2 saturation pulseoxymetry', 'Temperature Celsius',
          'Peak Insp. Pressure', 'Tidal Volume (observed)', 'Total PEEP Level', 'O2 Flow', 'EtCO2', 'Inspired O2 Fraction']

scores_feat = ['gcs', 'rass', 'cam']

medications_feat = ['Propofol', 'Fentanyl', 'Phenylephrine', 'Heparin Sodium', 'Dopamine', 'Norepinephrine',
               'Amiodarone', 'Dexmedetomidine (Precedex)', 'Epinephrine', 'Vasopressin', 'Digoxin (Lanoxin)', 'Folic Acid']

labs_feat = ['Arterial Base Excess', 'Arterial CO2 Pressure', 'Arterial O2 Saturation', 'Arterial O2 pressure',
        'Hematocrit (whole blood - calc)', 'Ionized Calcium', 'Lactic Acid', 'PH (Arterial)', 'Anion gap',
        'Calcium non-ionized', 'Chloride (serum)', 'Creatinine (serum)', 'Glucose (serum)', 'Hemoglobin',
        'INR', 'Platelet Count', 'Potassium (serum)', 'Sodium (serum)', 'Troponin-T', 'WBC', 'ALT', 'AST',
        'Total Bilirubin', 'Albumin', 'Brain Natiuretic Peptide (BNP)', 'Direct Bilirubin', 'C Reactive Protein (CRP)',
        'Absolute Count - Basos', 'Absolute Count - Eos', 'Absolute Count - Lymphs', 'Absolute Count - Monos', 'Specific Gravity (urine)']

static = pd.read_csv(f"{DATA_DIR}/static.csv")

print(len(static["icustay_id"].unique()))

char_dict = {
    "age": "Age",
    "gender": "Gender",
    "race": "Race",
    "bmi": "BMI",
    "aids": "AIDS",
    "cancer": "Cancer",
    "cerebrovascular": "Cerebrovascular Disease",
    "chf": "CHF",
    "copd": "COPD",
    "dementia": "Dementia",
    "diabetes_wo_comp": "Diabetes w/o Complications",
    "diabetes_w_comp": "Diabetes with Complications",
    "m_i": "Myocardial Infarction",
    "metastatic_carc": "Metastatic Carcinoma",
    "mild_liver_dis": "Mild Liver Disease",
    "mod_sev_liver_dis": "Moderate/Severe Liver Disease",
    "par_hem": "Paraplegia/Hemiplegia",
    "pep_ulc": "Peptic Ulcer Disease",
    "peri_vascular": "Peripheral Vascular Disease",
    "renal_dis": "Renal Disease",
    "rheu": "Rheumatologic Disease",
    "cci": "Charlson Comorbidity Index",
}

comob_dict = {
    "aids": "AIDS",
    "cancer": "Cancer",
    "cerebrovascular": "Cerebrovascular Disease",
    "chf": "CHF",
    "copd": "COPD",
    "dementia": "Dementia",
    "diabetes_wo_comp": "Diabetes w/o Complications",
    "diabetes_w_comp": "Diabetes with Complications",
    "m_i": "Myocardial Infarction",
    "metastatic_carc": "Metastatic Carcinoma",
    "mild_liver_dis": "Mild Liver Disease",
    "mod_sev_liver_dis": "Moderate/Severe Liver Disease",
    "par_hem": "Paraplegia/Hemiplegia",
    "pep_ulc": "Peptic Ulcer Disease",
    "peri_vascular": "Peripheral Vascular Disease",
    "renal_dis": "Renal Disease",
}

cci = ["Charlson Comorbidity Index"]

comob_list = list(comob_dict.values())

static = static.rename(char_dict, axis=1)

def seq_to_text(group):
    
    icu_id = group["icustay_id"].values[0]
    
    vitals = list(group.loc[group["variable"].isin(vitals_feat), "variable"].unique())

    vitals_report = ''

    for vital in vitals:
        
        vital_df = group[group["variable"] == vital]
        
        vitals_report = vitals_report + f'{vital}: {vital_df["value"].min():.2f}-{vital_df["value"].max():.2f}\n'

    vitals_report = "Vitals values:\n" + vitals_report

    labs_report = ''
    
    labs = list(group.loc[group["variable"].isin(labs_feat), "variable"].unique())
    
    if len(labs) > 0:

        for lab in labs:

            lab_df = group[group["variable"] == lab]

            labs_report = labs_report + f'{lab}: {lab_df["value"].min():.2f}-{lab_df["value"].max():.2f}\n'
            
    else:
        
        labs_report = labs_report + "None. "
        
    labs_report = "Laboratory values:\n" + labs_report
    
    med_report = ''
    
    meds = list(group.loc[group["variable"].isin(medications_feat), "variable"].unique())
        
    if len(meds) > 0:
    
        for med in meds:

            med_df = group[group["variable"] == med]

            med_report = med_report + f'{med}: {med_df["value"].sum():.2f}\n'


    else:
        
        med_report = med_report + "None. "
        
    med_report = "Medications:\n" + med_report
    
    scores_report = ''
    
    scores = list(group.loc[group["variable"].isin(scores_feat), "variable"].unique())
    
    if len(scores) > 0:

        for score in scores:

            score_df = group[group["variable"] == score]

            scores_report = scores_report + f'{score}: {score_df["value"].min():.2f}-{score_df["value"].max():.2f}\n'
            
    else:
        
        scores_report = scores_report + "None. "
        
    scores_report = "Score values:\n" + scores_report
        
    report = vitals_report + "\n" + labs_report + "\n" + med_report + "\n" + scores_report
            
    report = pd.DataFrame(data={"icustay_id": icu_id, "SEQ_TEXT": report}, index=[icu_id])
    
    return report


def static_to_text(row):
    
    row[comob_list+cci] = row[comob_list+cci].fillna(0)

    row[comob_list][row[comob_list] > 0] = 1

    row[comob_list] = row[comob_list].replace({0: "No", 1: "Yes"})
    
    row = row.fillna("Missing")
    
    comob_report = "Comorbidities at admission:\n "
    
    count = 0
    
    for comob in comob_list:

        if row[comob] == "Yes":

            comob_report = comob_report + f"{comob}. "
            
            count += 1
    
    if count == 0:
        
        comob_report = comob_report + "None. "


    admit_report = f"Age: {int(row['Age'])}\n" + f"Gender: {row['Gender']}\n" + f"Race: {row['Race']}\n"
        
    admit_report = admit_report + "\n" + comob_report + "\n"
                
    return admit_report


icustay_ids = seq["icustay_id"].unique()

text = seq.groupby("icustay_id").apply(seq_to_text).reset_index(drop=True)

print(text.head())
print(text.shape)

static["STATIC_TEXT"] = static.apply(static_to_text, axis=1)

text = text.merge(static.loc[:,["icustay_id", "STATIC_TEXT"]], how="inner", on="icustay_id")

text["SUMMARY"] = text["STATIC_TEXT"] + text["SEQ_TEXT"]


def get_num_tokens(row):
        
    tokens = word_tokenize(row["SUMMARY"])
    # Count the number of tokens
    num_tokens = len(tokens)
    
    return num_tokens


text["tokens"] = text.apply(get_num_tokens, axis=1)

text.drop("STATIC_TEXT", axis=1, inplace=True)
text.drop("SEQ_TEXT", axis=1, inplace=True)

print(text.head())
print(text.shape)

print(text["SUMMARY"].values[0])

print(text["tokens"].describe())

text.to_csv(f"{MODEL_DIR}/ehr_text.csv", index=False)

