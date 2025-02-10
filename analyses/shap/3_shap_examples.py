#%%
# Import libraries

import shap
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    auc,
)

#%%
# Define directories

MAIN_DIR = '/blue/parisa.rashidi/contreras.miguel/clinical_notes/main3'

model = 'gatortron_s_mlm'

DATA_DIR = f'{MAIN_DIR}/final_data'
MODEL_DIR = f'{MAIN_DIR}/models/ehr_text/{model}'
ANALYSIS_DIR = f'{MAIN_DIR}/analyses/{model}/shap'

test_set = "external_2"
outcome = "delirium_1"

#%%

# Calculate threshold and metrics

results = pd.read_csv(f"{MODEL_DIR}/results/{test_set}_{outcome}_results.csv")

model_probs = results["pred"].values
model_true = results["true"].values


fpr, tpr, thresholds = roc_curve(
    model_true, model_probs
)
roc_auc_class = auc(fpr, tpr)
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

precision, recall, thresholds = precision_recall_curve(
    model_true, model_probs
)
auprc = auc(recall, precision)

model_pred_class = (model_probs >= 0.1).astype("int")
cf_class = confusion_matrix(model_true, model_pred_class)
score_spec = cf_class[0, 0] / (cf_class[0, 0] + cf_class[0, 1])
score_npv = cf_class[0, 0] / (cf_class[0, 0] + cf_class[1, 0])
score_ppv = cf_class[1, 1] / (cf_class[1, 1] + cf_class[0, 1])
score_sens = cf_class[1, 1] / (cf_class[1, 1] + cf_class[1, 0])

print(f"Specificity: {score_spec}")
print(f"Sensitivity: {score_sens}")
print(f"PPV: {score_ppv}")
print(f"NPV: {score_npv}")

    
#%%


notes = pd.read_csv(f'{MODEL_DIR}/ehr_text.csv')
outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv").loc[:,["icustay_id", "delirium"]]

notes = notes.merge(outcomes, how="inner", on="icustay_id")
with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_test = ids["external_2"]

notes_test = notes[notes["icustay_id"].isin(ids_test)]
icu_ids = notes_test["icustay_id"].tolist()

#%%

# Get random samples

results_new = pd.DataFrame({"true": model_true, "pred": model_pred_class, "prob": model_probs, "icustay_id": icu_ids})

print(results_new.head())

fp_df = results_new[(results_new["true"] == 0) & (results_new["pred"] == 1)]
tp_df = results_new[(results_new["true"] == 1) & (results_new["pred"] == 1)]
fn_df = results_new[(results_new["true"] == 1) & (results_new["pred"] == 0)]
tn_df = results_new[(results_new["true"] == 0) & (results_new["pred"] == 0)]

print(fp_df.shape)
print(tp_df.shape)
print(fn_df.shape)
print(tn_df.shape)


random_sample = np.random.choice(fp_df.shape[1], size=1, replace=False)
fp_sample = fp_df.iloc[random_sample]["icustay_id"].values[0]

random_sample = np.random.choice(tp_df.shape[1], size=1, replace=False)
tp_sample = tp_df.iloc[random_sample]["icustay_id"].values[0]

random_sample = np.random.choice(fn_df.shape[1], size=1, replace=False)
fn_sample = fn_df.iloc[random_sample]["icustay_id"].values[0]

random_sample = np.random.choice(tn_df.shape[1], size=1, replace=False)
tn_sample = tn_df.iloc[random_sample]["icustay_id"].values[0]

print(fp_sample)
print(tp_sample)
print(fn_sample)
print(tn_sample)

samples = [fp_sample, tp_sample, fn_sample, tn_sample]

results_new = results_new[results_new["icustay_id"].isin(samples)]
print(results_new.head())

#%%

# Get text reports for samples

notes = pd.read_csv(f'{MODEL_DIR}/ehr_text.csv')
notes = notes[notes["icustay_id"].isin(samples)]

notes = notes.merge(results_new, on="icustay_id", how="inner")

texts = notes["SUMMARY"].tolist()

print(notes.head())

#%%

# Load fine-tuned model

fine_tuned_dir = "fine_tuned_model"

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(f'{MODEL_DIR}/{fine_tuned_dir}')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gatortron = AutoModelForSequenceClassification.from_pretrained(f'{MODEL_DIR}/{fine_tuned_dir}')

from transformers import TextClassificationPipeline


class DeliriumLLM(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        best_class = torch.nn.Sigmoid()(best_class[:,1])
        return best_class

pipe = TextClassificationPipeline(model=gatortron, tokenizer=tokenizer, device=0, batch_size=32, padding="max_length", truncation=True, max_length=512, return_all_scores=True)

# Define a wrapper function for SHAP
def predict_proba(texts):
    results = pipe(texts)
    return results

#%%

# Initialize SHAP explainer

explainer = shap.Explainer(pipe)

shap_values = explainer(texts)

#%%

# Plot first example

shap.plots.text(shap_values[0,:,1])

#%%

# Plot second example

shap.plots.text(shap_values[1,:,1])

#%%

# Plot third example

shap.plots.text(shap_values[2,:,1])

#%%

# Plot fourth example

shap.plots.text(shap_values[3,:,1])

