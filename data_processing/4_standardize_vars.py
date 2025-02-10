# Import libraries

import h5py
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from variables import INTERNAL_DATA_DIR, EICU_DATA_DIR, MIMIC_DATA_DIR, MAIN_DIR

seq = pd.read_csv(f"{MAIN_DIR}/clinical_data_unconverted.csv")

# Convert absolute counts

var = "Absolute Count"
cohorts = ["EICU"]

for cohort in cohorts:
    
    seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] = seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] / 10

# Convert CRP

var = "C Reactive Protein"
cohorts = ["internal"]

for cohort in cohorts:
    
    seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] = seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] / 10
    
# Convert BNP

var = "Brain Natiuretic"
cohorts = ["internal"]

for cohort in cohorts:
    
    seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] = seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] / 0.289

    
# Convert direct bilirubin

var = "Direct Bilirubin"
cohorts = ["MIMIC"]

for cohort in cohorts:
    
    seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] = seq.loc[(seq["icustay_id"].str.contains(cohort)) & (seq["variable"].str.contains(var)), "value"] / 10


lab_vars = ['Arterial Base Excess', 'Arterial CO2 Pressure', 'Arterial O2 Saturation', 'Arterial O2 pressure',
        'Hematocrit (whole blood - calc)', 'Ionized Calcium', 'Lactic Acid', 'PH (Arterial)', 'Anion gap',
        'Calcium non-ionized', 'Chloride (serum)', 'Creatinine (serum)', 'Glucose (serum)', 'Hemoglobin',
        'INR', 'Platelet Count', 'Potassium (serum)', 'Sodium (serum)', 'Troponin-T', 'WBC', 'ALT', 'AST',
        'Total Bilirubin', 'Albumin', 'Brain Natiuretic Peptide (BNP)', 'Direct Bilirubin', 'C Reactive Protein (CRP)',
        'Absolute Count - Basos', 'Absolute Count - Eos', 'Absolute Count - Lymphs', 'Absolute Count - Monos', 'Specific Gravity (urine)']

cohorts = ["internal", "MIMIC", "EICU"]

lab_analysis = pd.DataFrame(data=[], columns=cohorts, index=lab_vars)

for cohort in cohorts:
    
    subset = seq[seq["icustay_id"].str.contains(cohort)]
    
    print(f"--------{cohort}--------")

    print("-"*40)

    for lab in lab_vars:

        lab_df = subset[subset["variable"] == lab]
        
        low = lab_df['value'].quantile(0.25)
        high = lab_df['value'].quantile(0.75)
        
        lab_analysis.loc[lab, cohort] = f"{low}-{high}"

        print(f"{lab}: {low}-{high}")

        print("-"*40)
        

print(lab_analysis)

lab_analysis.to_csv(f"{MAIN_DIR}/lab_analysis.csv")

print(seq.head())

print(f'Total variables: {len(seq["variable"].unique())}')
print(f'Total ICU admissions: {len(seq["icustay_id"].unique())}')

seq.to_csv(f"{MAIN_DIR}/clinical_data.csv", index=False)
