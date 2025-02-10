import pandas as pd
import numpy as np
import torch
import pickle

from variables import DATA_DIR, MODEL_DIR, LLM_DIR

import os

if not os.path.exists(f"{MODEL_DIR}/results"):
    os.makedirs(f"{MODEL_DIR}/results")

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(f'{MODEL_DIR}/fine_tuned_model')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForSequenceClassification.from_pretrained(f'{MODEL_DIR}/fine_tuned_model')

from transformers import TextClassificationPipeline

class DeliriumLLM(TextClassificationPipeline):
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"]
        best_class = torch.nn.Sigmoid()(best_class[:,1])
        return best_class

pipe = DeliriumLLM(model=model, tokenizer=tokenizer, device=0, batch_size=32, padding="max_length", truncation=True, max_length=512)

notes = pd.read_csv(f'{MODEL_DIR}/ehr_text.csv')
outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv").loc[:,["icustay_id", "delirium"]]

notes = notes.merge(outcomes, how="inner", on="icustay_id")

notes["SUMMARY"] = "[CLS] " + notes["SUMMARY"] + " [SEP]"

test_sets = ["internal", "external_1", "external_2"]

for test_set in test_sets:

    with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
        ids = pickle.load(f)
        ids_test = ids[test_set]

    notes_test = notes[notes["icustay_id"].isin(ids_test)]

    print(notes_test.shape)

    test_samples = notes_test["SUMMARY"].tolist()
    test_labels = notes_test["delirium"].tolist()

    predictions = pipe(test_samples)

    predictions = np.array(predictions).flatten()

    test_labels = np.array(test_labels)

    from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

    print("-" * 40)
    print(f"Test performance: {test_set}")

    roc_auc = roc_auc_score(test_labels, predictions)

    print(f"val_roc_auc: {roc_auc}")

    precision, recall, _ = precision_recall_curve(
        test_labels, predictions
    )
    pr_auc = auc(recall, precision)

    print(f"val_pr_auc: {pr_auc}")
    
    d_output = 1
    
    outcomes = [f"delirium_{i+1}" for i in range(d_output)]

    for i in range(len(outcomes)):

        results = pd.DataFrame(np.concatenate([test_labels.reshape((-1,1)), predictions.reshape((-1,1))], axis=1), columns=["true", "pred"])

        results.to_csv(
            f"{MODEL_DIR}/results/{test_set}_{outcomes[i]}_results.csv", index=None
        )
