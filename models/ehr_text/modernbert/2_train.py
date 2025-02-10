import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from variables import DATA_DIR, MODEL_DIR, LLM_DIR

import os

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_DIR}/modernbert/")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(f"{MODEL_DIR}/modernbert/", num_labels=2).to('cuda:0')

print(model)

FREEZE_BASE = True

num_layers = 5

# Freeze the base model weights.
if FREEZE_BASE:
    for param in model.parameters():
        param.requires_grad = False
        
    for i in range(num_layers):

        layer_id = 21 - i
        
        for param in model.model.layers[layer_id].parameters():
            param.requires_grad = True
    
    for param in model.head.parameters():
        param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print('Parameters: {}'.format(params))


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print('Parameters: {}'.format(params))

notes = pd.read_csv(f'{MODEL_DIR}/ehr_text.csv')

outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv").loc[:,["icustay_id", "delirium"]]

notes = notes.merge(outcomes, how="inner", on="icustay_id")

notes["SUMMARY"] = "[CLS] " + notes["SUMMARY"] + " [SEP]"

print(notes.shape)
print(notes.head())
print(notes["SUMMARY"].values[0])
print(notes["delirium"].value_counts())

import pickle
with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train = ids["train"]
    ids_val = ids["val"]

    
notes_train = notes[notes["icustay_id"].isin(ids_train)]
notes_val = notes[notes["icustay_id"].isin(ids_val)]

print(notes_train["delirium"].value_counts())

train_texts = notes_train['SUMMARY'].tolist()
train_labels = notes_train['delirium'].tolist()

val_texts = notes_val['SUMMARY'].tolist()
val_labels = notes_val['delirium'].tolist()


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, labels):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Dataset and DataLoader
max_seq_length = 512
train_dataset = MyDataset(train_texts, tokenizer, max_seq_length, train_labels)
val_dataset = MyDataset(val_texts, tokenizer, max_seq_length, val_labels)

print(len(train_dataset))
print(len(val_dataset))

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from sklearn.metrics import roc_auc_score

from datasets import load_metric

metric = load_metric("roc_auc")

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    predictions = sigmoid_v(logits[:, 1])
    
    return metric.compute(prediction_scores=predictions, references=labels)

early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)

training_args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/trainer/fine_tune", 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    num_train_epochs=10, 
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    lr_scheduler_type="reduce_lr_on_plateau",
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model='eval_roc_auc',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)

trainer.train()

eval_results = trainer.evaluate()

print(eval_results)

model.save_pretrained(f'{MODEL_DIR}/fine_tuned_model')
tokenizer.save_pretrained(f'{MODEL_DIR}/fine_tuned_model')
