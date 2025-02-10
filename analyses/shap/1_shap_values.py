import shap
import torch
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import pickle

MAIN_DIR = '/blue/parisa.rashidi/contreras.miguel/clinical_notes/main3'

model = 'gatortron_s_mlm'

DATA_DIR = f'{MAIN_DIR}/final_data'
MODEL_DIR = f'{MAIN_DIR}/models/ehr_text/{model}'
ANALYSIS_DIR = f'{MAIN_DIR}/analyses/{model}/shap'

import os
if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR)
    
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


notes = pd.read_csv(f'{MODEL_DIR}/ehr_text.csv')

notes_sec = pd.read_csv(f'{MODEL_DIR}/ehr_text_sec_ver2.csv')


outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv").loc[:,["icustay_id", "delirium"]]

notes = notes.merge(outcomes, how="inner", on="icustay_id")

notes["SUMMARY"] = "[CLS] " + notes["SUMMARY"] + " [SEP]"
notes_sec["SUMMARY"] = "[CLS] " + notes_sec["SUMMARY"] + " [SEP]"

test_sets = ["internal", "external_1", "external_2"]

samples = 1000

samples_class = int(samples / 2)

for test_set in test_sets:


    with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
        ids = pickle.load(f)
        ids_test = ids[test_set]

    notes_test = notes[notes["icustay_id"].isin(ids_test)]
    notes_test_sec = notes_sec[notes_sec["icustay_id"].isin(ids_test)]
    
    pos_ids = list(notes_test.loc[notes_test["delirium"] == 1, "icustay_id"].unique())
    neg_ids = list(notes_test.loc[notes_test["delirium"] == 0, "icustay_id"].unique())
    
    if len(pos_ids) < samples_class:
        
        random_pos = pos_ids
        
        random_neg = np.random.choice(len(neg_ids), size=len(pos_ids), replace=False)

        random_neg = list(np.array(neg_ids)[random_neg])
    
    else:
    
        random_pos = np.random.choice(len(pos_ids), size=samples_class, replace=False)

        random_pos = list(np.array(pos_ids)[random_pos])

        random_neg = np.random.choice(len(neg_ids), size=samples_class, replace=False)

        random_neg = list(np.array(neg_ids)[random_neg])

    samples_ids = random_pos+random_neg

    np.random.shuffle(samples_ids)

    notes_test = notes_test[notes_test["icustay_id"].isin(samples_ids)]
    notes_test_sec = notes_test_sec[notes_test_sec["icustay_id"].isin(samples_ids)]

    icu_ids = notes_test["icustay_id"].tolist()

    print(notes_test.shape)
    print(notes_test["delirium"].value_counts())

    texts = notes_test["SUMMARY"].tolist()
    texts_sec = notes_test_sec["SUMMARY"].tolist()
    
    outputs = pipe(texts)
        
    tokenized_texts = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt", max_length=512)
    tokenized_texts_sec = tokenizer(texts_sec, padding="max_length", truncation=True, return_tensors="pt", max_length=512)

    explainer = shap.Explainer(pipe)

    shap_values = explainer(texts)

    
    feature_names = [tokenizer.convert_ids_to_tokens(tokenized_texts['input_ids'][i].tolist()) for i in range(len(texts))]
    feature_names_sec = [tokenizer.convert_ids_to_tokens(tokenized_texts_sec['input_ids'][i].tolist()) for i in range(len(texts_sec))]
    
            
    df_exp = pd.DataFrame()
        
    for i in range(len(texts)):
        
    
        shap_values_for_class = shap_values[i, :, :]
        feats = feature_names[i][:len(shap_values_for_class)]
        
        feats_sec = feature_names_sec[i]
        
        if len(feats) == len(list(shap_values_for_class.values[:,1])):
            
            print("Processing sample")

            df_exp_ind = pd.DataFrame(data={"feat": feats, "shap_value": list(shap_values_for_class.values[:,1])})

            df_exp_ind["icustay_id"] = icu_ids[i]

            df_sec =  pd.DataFrame(data={"feat": feats_sec})

            # Initialize feat_type column with zeros
            df_sec['feat_type'] = 0

            # Initialize counter
            counter = 0

            # Iterate through the DataFrame rows
            for i, row in df_sec.iterrows():
                if row['feat'] == '[SEP]':
                    counter += 1
                df_sec.at[i, 'feat_type'] = counter
 
            df_sec = df_sec[~(df_sec['feat'].str.contains("[SEP]")) | ~(df_sec['feat'].str.contains("[CLS]"))]
            df_sec = df_sec[~(df_sec['feat'].str.contains("[PAD]"))]
                
            df_exp_ind = df_exp_ind[~(df_exp_ind['feat'].str.contains("[SEP]")) | ~(df_exp_ind['feat'].str.contains("[CLS]"))]
             
            df_sec = df_sec.reset_index(drop=True)    
            df_exp_ind = df_exp_ind.reset_index(drop=True)             
            
            df_exp_ind = pd.concat([df_exp_ind, df_sec], axis=1)
            
            
            df_exp_ind.to_csv(
                f"{ANALYSIS_DIR}/shap_df_{test_set}_balanced.csv",
                mode="w" if (not os.path.isfile(f"{ANALYSIS_DIR}/shap_df_{test_set}_balanced.csv")) else "a",
                header=None if (os.path.isfile(f"{ANALYSIS_DIR}/shap_df_{test_set}_balanced.csv")) else "infer",
            )
            
        else:
            
            print("Debug")
    
    