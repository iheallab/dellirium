#%%
import pandas as pd
import numpy as np
import torch
import pickle
from fuzzywuzzy import process
import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
pio.templates.default = "plotly_white"

#%%

MAIN_DIR = '/blue/parisa.rashidi/contreras.miguel/clinical_notes/main3'

model = 'gatortron_s_mlm'

DATA_DIR = f'{MAIN_DIR}/final_data'
MODEL_DIR = f'{MAIN_DIR}/models/ehr_text/{model}'
ANALYSIS_DIR = f'{MAIN_DIR}/analyses/{model}/shap'

outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv")

vitals_feat = ['Respiratory Rate', 'Heart Rate', 'Non Invasive Blood Pressure diastolic',
          'Non Invasive Blood Pressure systolic', 'O2 saturation pulseoxymetry', 'Temperature Celsius',
          'Peak Insp. Pressure', 'Tidal Volume (observed)', 'Total PEEP Level', 'O2 Flow', 'EtCO2', 'Inspired O2 Fraction']

scores_feat = ['GCS', 'RASS', 'CAM']

medications_feat = ['Propofol', 'Fentanyl', 'Phenylephrine', 'Heparin Sodium', 'Dopamine', 'Norepinephrine',
               'Amiodarone', 'Dexmedetomidine (Precedex)', 'Epinephrine', 'Vasopressin', 'Digoxin (Lanoxin)', 'Folic Acid']

labs_feat = ['Arterial Base Excess', 'Arterial CO2 Pressure', 'Arterial O2 Saturation', 'Arterial O2 pressure',
        'Hematocrit (whole blood - calc)', 'Ionized Calcium', 'Lactic Acid', 'PH (Arterial)', 'Anion gap',
        'Calcium non-ionized', 'Chloride (serum)', 'Creatinine (serum)', 'Glucose (serum)', 'Hemoglobin',
        'INR', 'Platelet Count', 'Potassium (serum)', 'Sodium (serum)', 'Troponin-T', 'WBC', 'ALT', 'AST',
        'Total Bilirubin', 'Albumin', 'Brain Natiuretic Peptide (BNP)', 'Direct Bilirubin', 'C Reactive Protein (CRP)',
        'Absolute Count - Basos', 'Absolute Count - Eos', 'Absolute Count - Lymphs', 'Absolute Count - Monos', 'Specific Gravity (urine)']


comorb_dict = {
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
}

comorb_feat = list(comorb_dict.values())


comorb_dict_summ = {
    "aids": "AIDS",
    "cancer": "Cancer",
    "cerebrovascular": "Cerebrovascular Disease",
    "chf": "CHF",
    "copd": "COPD",
    "dementia": "Dementia",
    "diabetes_wo_comp": "Diabetes",
    "m_i": "Myocardial Infarction",
    "metastatic_carc": "Metastatic Carcinoma",
    "mod_sev_liver_dis": "Liver Disease",
    "par_hem": "Paraplegia/Hemiplegia",
    "pep_ulc": "Peptic Ulcer Disease",
    "peri_vascular": "Peripheral Vascular Disease",
    "renal_dis": "Renal Disease",
    "rheu": "Rheumatologic Disease",
}

comorb_feat_summ = list(comorb_dict_summ.values())


demo_dict = {
    "age": "Age",
    "gender": "Gender",
    "race": "Race",
    "bmi": "BMI",
}

demo_feat = list(demo_dict.values())

cci_dict = {
    "cci": "Charlson Comorbidity Index",
}

cci_feat = list(cci_dict.values())

all_feats = vitals_feat + scores_feat + medications_feat + labs_feat + comorb_feat + demo_feat + cci_feat

feats_cat = [[vitals_feat, "Vital Signs"], [labs_feat, "Laboratory Tests"], [comorb_feat_summ, "Comorbidities"]]

#%%

test_sets = ["internal", "external_1", "external_2"]

for test_set in test_sets:
    df_exp = pd.read_csv(f'{ANALYSIS_DIR}/shap_df_{test_set}_balanced.csv')
    icu_ids = df_exp["icustay_id"].unique()
    outcomes_sample = outcomes[outcomes["icustay_id"].isin(icu_ids)]
    delirium = outcomes_sample[outcomes_sample["delirium"] == 1]["icustay_id"].unique()
    nodelirium = outcomes_sample[outcomes_sample["delirium"] == 0]["icustay_id"].unique()

    df_exp["shap_value"] = df_exp["shap_value"].abs()
    shap_values = df_exp.groupby(by=["feat_type", "icustay_id"], as_index=False).agg({'shap_value': 'sum', 'feat': ''.join})
    shap_values['base_feat'] = shap_values['feat'].apply(lambda x: x.split(':')[0])

    prop_del = shap_values[shap_values["icustay_id"].isin(delirium)].groupby('base_feat').size() / len(delirium)
    prop_del = pd.DataFrame({"prop_del": prop_del.values}, index=prop_del.index).reset_index().rename({"index": "base_feat"}, axis=1)

    prop_nodel = shap_values[shap_values["icustay_id"].isin(nodelirium)].groupby('base_feat').size() / len(nodelirium)
    prop_nodel = pd.DataFrame({"prop_nodel": prop_nodel.values}, index=prop_nodel.index).reset_index().rename({"index": "base_feat"}, axis=1)

    mean_shap_values_base = shap_values.groupby('base_feat')['shap_value'].mean().reset_index()
    counts_feat = shap_values.groupby('base_feat').size() / len(df_exp["icustay_id"].unique())
    counts_feat = pd.DataFrame({"prop": counts_feat.values}, index=counts_feat.index).reset_index().rename({"index": "base_feat"}, axis=1)

    mean_shap_values_base = mean_shap_values_base.merge(counts_feat, on="base_feat", how="inner")
    mean_shap_values_base = mean_shap_values_base.merge(prop_nodel, on="base_feat", how="inner")
    mean_shap_values_base = mean_shap_values_base.merge(prop_del, on="base_feat", how="inner")

    mean_shap_values_base.sort_values(by=["shap_value"], ascending=False, inplace=True)

    # Mapping feature names
    def map_to_most_similar(string, target_list):
        return process.extractOne(string, target_list)[0]

    mean_shap_values_base['mapped_feat'] = mean_shap_values_base['base_feat'].apply(lambda x: map_to_most_similar(x, all_feats))

    # Prepare a list to collect data for all categories
    plot_data_frames = []

    # Loop over feat_cat, collect top features & store them along with the category name
    for feat_cat in feats_cat:
        # Filter and select top 3 rows
        shap_cat = mean_shap_values_base[
            mean_shap_values_base["mapped_feat"].isin(feat_cat[0])
        ].reset_index(drop=True).iloc[:3]

        # Add a column to indicate the category name
        shap_cat["Feature Type"] = feat_cat[1]
        
        # Append to the list
        plot_data_frames.append(shap_cat)

    # Combine all the partial data into one DataFrame
    shap_plot_data = pd.concat(plot_data_frames, ignore_index=True)

    # Now create a single bar plot
    fig = px.bar(
        shap_plot_data,
        x="shap_value",
        y="mapped_feat",
        color="Feature Type",            # Use color to distinguish categories
        orientation="h",
        labels={"shap_value": "Mean Absolute SHAP Value", "mapped_feat": ""},
        color_discrete_sequence=["#AA2EE6", "#FF79CD", "#FFDF6B"]
    )
    
    fontsize = 24
    
    fig.update_layout(
        xaxis=dict(titlefont=dict(size=fontsize), tickfont=dict(size=fontsize), tickformat=".3f"),
        yaxis=dict(titlefont=dict(size=fontsize), tickfont=dict(size=fontsize), autorange="reversed"),
        legend=dict(
            font=dict(size=fontsize),
        ),
        template="plotly_white",
        barmode='group',
        width=1400,
        height=600,
    )
    
    fig.write_image(f'{ANALYSIS_DIR}/shap_plot_{test_set}_categorized.png', scale=2)

    
    fig.show()

    mean_shap_values_base = mean_shap_values_base.reset_index(drop=True).iloc[:15]

    fig = px.bar(
        mean_shap_values_base,
        x="shap_value",
        y="mapped_feat",
        orientation="h",
        labels={"shap_value": "Mean Absolute SHAP Value", "mapped_feat": ""},
        color_discrete_sequence=["#23049D"],
    )
    
    fig.update_layout(
        xaxis=dict(titlefont=dict(size=fontsize), tickfont=dict(size=fontsize), tickformat=".3f"),
        yaxis=dict(titlefont=dict(size=fontsize), tickfont=dict(size=fontsize), autorange="reversed"),
        template="plotly_white",
        width=1400,
        height=600,
    )
    
    fig.write_image(f'{ANALYSIS_DIR}/shap_plot_{test_set}_all.png', scale=2)
    
    fig.show()    
        
