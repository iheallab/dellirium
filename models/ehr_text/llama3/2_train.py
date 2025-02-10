import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from variables import DATA_DIR, MODEL_DIR, LLM_DIR

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

import os

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

notes = pd.read_csv(f'{MODEL_DIR}/ehr_text.csv')

outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv").loc[:,["icustay_id", "delirium"]]

notes = notes.merge(outcomes, how="inner", on="icustay_id")

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

print(notes_train["delirium"].value_counts())
print(notes_val["delirium"].value_counts())

train_texts = notes_train['SUMMARY'].tolist()
train_labels = notes_train['delirium'].tolist()

val_texts = notes_val['SUMMARY'].tolist()
val_labels = notes_val['delirium'].tolist()


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from sklearn.metrics import roc_auc_score

from datasets import load_metric

metric = load_metric("roc_auc")

early_stop = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    predictions = sigmoid_v(logits[:, 1])
    
    return metric.compute(prediction_scores=predictions, references=labels)


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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"{LLM_DIR}/llama3/", trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': "<pad>"})


# Dataset and DataLoader
max_seq_length = 1024
train_dataset = MyDataset(train_texts, tokenizer, max_seq_length, train_labels)
val_dataset = MyDataset(val_texts, tokenizer, max_seq_length, val_labels)


# Quantization configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    f"{LLM_DIR}/llama3/",
    quantization_config=quant_config,
    device_map={"": 0}
)

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False
model.config.pretraining_tp = 1

print(model)

# Training arguments
training_params = TrainingArguments(
    output_dir=f"{MODEL_DIR}/trainer",
    evaluation_strategy="epoch",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=500,
    learning_rate=2e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=False,
    load_best_model_at_end=True,
    save_total_limit=2,
    lr_scheduler_type="constant",
    metric_for_best_model='eval_roc_auc',
)

# LoRA configuration
peft_params = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

model = get_peft_model(model, peft_params)

for param in model.score.parameters():
    param.requires_grad = True

model.print_trainable_parameters()
# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_params,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=None, 
    args=training_params,
    compute_metrics=compute_metrics,
    callbacks=[early_stop],
)

trainer.train()

eval_results = trainer.evaluate()

print(eval_results)

# Save the model and tokenizer
trainer.model.save_pretrained(f"{MODEL_DIR}/fine_tuned")
trainer.tokenizer.save_pretrained(f"{MODEL_DIR}/fine_tuned")


