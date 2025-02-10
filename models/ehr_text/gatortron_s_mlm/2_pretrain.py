import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math

from variables import DATA_DIR, MODEL_DIR, LLM_DIR

import os

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained(f"{LLM_DIR}/gatortron_s/")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForMaskedLM.from_pretrained(f"{LLM_DIR}/gatortron_s/").to('cuda:0')

print(model)

FREEZE_BASE = True

num_layers = 5

# Freeze the base model weights.
if FREEZE_BASE:
    for param in model.parameters():
        param.requires_grad = False
        
    for i in range(num_layers):

        layer_id = 23 - i
        
        for param in model.bert.encoder.layer[layer_id].parameters():
            param.requires_grad = True
    
    for param in model.cls.parameters():
        param.requires_grad = True

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print('Parameters: {}'.format(params))

notes = pd.read_csv(f'{MODEL_DIR}/ehr_text.csv')

outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv").loc[:,["icustay_id", "delirium"]]

notes = notes.merge(outcomes, how="inner", on="icustay_id")

notes["delirium"] = notes["delirium"].map({0: "Negative", 1: "Positive"})

notes["SUMMARY"] = "[CLS] " + notes["SUMMARY"] + " [SEP]"

print(notes.shape)
print(notes.head())
print(notes["delirium"].value_counts())

import pickle
with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
    ids = pickle.load(f)
    ids_train = ids["train"]
    ids_val = ids["val"]

    
notes_train = notes[notes["icustay_id"].isin(ids_train)]
notes_val = notes[notes["icustay_id"].isin(ids_val)]

train_texts = notes_train['SUMMARY'].tolist()
val_texts = notes_val['SUMMARY'].tolist()

print(len(train_texts))
print(len(val_texts))

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        return item

# Dataset and DataLoader
max_seq_length = 512
train_dataset = MyDataset(train_texts, tokenizer, max_seq_length)
val_dataset = MyDataset(val_texts, tokenizer, max_seq_length)


from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=0.15)

from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir=f"{MODEL_DIR}/trainer/pretrain", 
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    num_train_epochs=10, 
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    lr_scheduler_type="reduce_lr_on_plateau",
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()

eval_results = trainer.evaluate()

print(eval_results)

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


model.save_pretrained(f'{MODEL_DIR}/pretrained_model')
tokenizer.save_pretrained(f'{MODEL_DIR}/pretrained_model')

