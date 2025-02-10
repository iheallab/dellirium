# Import libraries

import pandas as pd
import numpy as np
import h5py
import os
import tqdm
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

#%%

from variables import MODEL_DIR

#%%

# Create directory for saving model

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#%%

# Load data

with h5py.File("%s/dataset.h5" % MODEL_DIR, "r") as f:
    data = f["train"]
    X_train = data["X"][:]
    static_train = data["static"][:]
    y_train = data["y"][:]

    data = f["val"]
    X_val = data["X"][:]
    static_val = data["static"][:]
    y_val = data["y"][:]


# %%

# Shuffle training data

from sklearn.utils import shuffle

X_train, static_train, y_train = shuffle(
    X_train, static_train, y_train
)

#%%

# Define hyper parameters and model parameters

import torch
import math
import torch.nn as nn
from apricotm import ApricotM

EPOCH = 100
BATCH_SIZE = 256
LR = 5e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"use DEVICE: {DEVICE}")

# Convert data to tensors

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
static_train = torch.FloatTensor(static_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)
static_val = torch.FloatTensor(static_val)


class_weights = []

d_input = X_train.shape[1]
d_static = static_train.shape[1]
d_output = y_train.shape[1]

for i in range(d_output):

    negative = np.unique(y_train[:, i], return_counts=True)[1][0]
    positive = np.unique(y_train[:, i], return_counts=True)[1][1]

    class_weight = negative / positive

    # class_weight = torch.tensor(negative/positive)
    class_weights.append(class_weight)

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch import optim
from torch.autograd import Variable

loss_bin = BCELoss()
loss_bin_mort = BCEWithLogitsLoss()

import pickle

with open("%s/best_params.pkl" % MODEL_DIR, "rb") as f:
    best_params = pickle.load(f)


# Define hyperparameters to be optimized
d_model = best_params["d_model"]
d_hidden = best_params["d_hidden"]
n_layer = best_params["n_layer"]
LR = best_params["learning_rate"]
BATCH_SIZE = best_params["batch_size"]
dropout = best_params["dropout"]
seq_len = best_params["seq_len"]
max_code = 60

nb_batches = int(len(X_train) / BATCH_SIZE) + 1
nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

# Initialize model, loss, and optimizer
net = ApricotM(
    d_model=d_model,
    d_hidden=d_hidden,
    d_input=d_input,
    max_code=max_code,
    d_static=d_static,
    d_output=d_output,
    n_layer=n_layer,
    dropout=dropout,
    device=DEVICE,
).to(DEVICE)

optimizer_name = "Adam"

if optimizer_name == "Adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
elif optimizer_name == "RMS":
    optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-5)


from functions import EarlyStopping

early_stopping = EarlyStopping(
    patience=3, verbose=True, path=f"{MODEL_DIR}/apricotm_weights.pth"
)

best_score = 0
best_auroc = 0
best_auprc = 0

# Function to apply Kaiming initialization to all parameters
def kaiming_init(m):
    for param in m.parameters():
        if param.dim() > 1:  # Apply only to weights, not biases
            nn.init.kaiming_normal_(param, nonlinearity='relu')
        else:
            nn.init.constant_(param, 0)
            
net.apply(kaiming_init)

for epoch in range(EPOCH):
    train_loss = 0
    # total_accuracy = 0
    count = 0
    y_true = np.zeros((len(X_train)))
    y_true_class = np.zeros((len(X_train), d_output))
    y_pred = np.zeros((len(X_train)))
    y_pred_prob = np.zeros((len(X_train), d_output))
    for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):
        # for patient in range(0, len(X_train), BATCH_SIZE):
        # Adjust sequence size to 256 for each sample
        inputs = []
        for sample in X_train[patient : patient + BATCH_SIZE]:

            # Find the last non-zero index in the first dimension (512)
            last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()

            # Take the last 256 values if the last non-zero index is greater than or equal to 256
            if last_non_zero_index >= seq_len:
                adjusted_sample = sample[
                    last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :
                ]
            # Pad the sequence with zeros if the last non-zero index is less than 256
            else:
                padding = torch.zeros(
                    (seq_len - last_non_zero_index - 1, sample.shape[1]),
                    dtype=sample.dtype,
                )
                adjusted_sample = torch.cat(
                    (sample[: last_non_zero_index + 1], padding), dim=0
                )

            inputs.append(adjusted_sample)

        inputs = torch.stack(inputs)
        inputs = Variable(inputs).to(DEVICE)

        # inputs = Variable(X_train[patient:patient+BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        optimizer.zero_grad()
        pred_y = net(inputs, static_input)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        loss.backward()
        train_loss += loss.data
        optimizer.step()

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

        count += 1

    print("-" * 40)
    print(f"Epoch {epoch+1}")

    print("Train performance")
    print("loss: {}".format(train_loss / nb_batches))

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
        aucs.append(ind_auc)

    print(f"train_roc_auc: {aucs}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"train_pr_auc: {aucs}")

    y_true = np.zeros((len(X_val)))
    y_true_class = np.zeros((len(X_val), d_output))
    y_pred = np.zeros((len(X_val)))
    y_pred_prob = np.zeros((len(X_val), d_output))
    val_loss = 0
    for patient in range(0, len(X_val), BATCH_SIZE):

        inputs = []
        for sample in X_val[patient : patient + BATCH_SIZE]:

            # Find the last non-zero index in the first dimension (512)
            last_non_zero_index = torch.where(sample[:, 1] != 0)[0][-1].item()

            # Take the last 256 values if the last non-zero index is greater than or equal to 256
            if last_non_zero_index >= seq_len:
                adjusted_sample = sample[
                    last_non_zero_index - seq_len + 1 : last_non_zero_index + 1, :
                ]
            # Pad the sequence with zeros if the last non-zero index is less than 256
            else:
                padding = torch.zeros(
                    (seq_len - last_non_zero_index - 1, sample.shape[1]),
                    dtype=sample.dtype,
                )
                adjusted_sample = torch.cat(
                    (sample[: last_non_zero_index + 1], padding), dim=0
                )

            inputs.append(adjusted_sample)

        inputs = torch.stack(inputs)
        inputs = Variable(inputs).to(DEVICE)

        # inputs = Variable(X_val[patient:patient+BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_val[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)

        pred_y = net(inputs, static_input)

        loss = 0

        for i in range(d_output):
            loss_class = loss_bin(pred_y[:, i], labels[:, i]) * class_weights[i]

            loss = loss + loss_class

        val_loss += loss.data

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

    print("-" * 40)
    print("Val performance")

    print("val_loss: {}".format(val_loss / nb_batches_val))

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        ind_auc = roc_auc_score(y_true_class[:, i], y_pred_prob[:, i])
        aucs.append(ind_auc)

    print(f"val_roc_auc: {aucs}")

    aucs = np.array(aucs)

    roc_auc_mean = np.mean(aucs)

    print(f"Overall AUROC: {roc_auc_mean}")

    aucs = []
    for i in range(y_pred_prob.shape[1]):
        precision, recall, _ = precision_recall_curve(
            y_true_class[:, i], y_pred_prob[:, i]
        )
        val_pr_auc = auc(recall, precision)
        aucs.append(val_pr_auc)

    print(f"val_pr_auc: {aucs}")

    aucs = np.array(aucs)

    pr_auc_mean = np.mean(aucs)

    print(f"Overall AUPRC: {pr_auc_mean}")

    roc_pr_auc = roc_auc_mean + pr_auc_mean

    if roc_pr_auc > best_score:
        best_score = roc_pr_auc
        best_auroc = roc_auc_mean
        best_auprc = pr_auc_mean

    early_stopping(roc_pr_auc, net)

    if early_stopping.early_stop:
        print("Early stopping")
        break


model_architecture = {
    "d_model": d_model,
    "d_hidden": d_hidden,
    "d_input": d_input,
    "max_code": max_code,
    "d_static": d_static,
    "d_output": d_output,
    "n_layer": n_layer,
    "dropout": dropout,
}

torch.save(model_architecture, f"{MODEL_DIR}/apricotm_architecture.pth")