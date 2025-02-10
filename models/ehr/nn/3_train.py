# Import libraries

import pandas as pd
import numpy as np
import h5py
import os
import optuna
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
    y_train = data["y"][:]

    data = f["val"]
    X_val = data["X"][:]
    y_val = data["y"][:]


# %%

# Shuffle training data

from sklearn.utils import shuffle

X_train, y_train = shuffle(
    X_train, y_train
)

#%%

# Define hyper parameters and model parameters

import torch
import math
import torch.nn as nn
from model import NeuralNetwork

EPOCH = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"use DEVICE: {DEVICE}")

# Convert data to tensors

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

d_input = X_train.shape[1]
d_output = y_train.shape[1]

class_weights = []

for i in range(d_output):

    negative = np.unique(y_train[:,i], return_counts=True)[1][0]
    positive = np.unique(y_train[:,i], return_counts=True)[1][1]

    class_weight = negative / positive
    
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

nb_batches = int(len(X_train) / BATCH_SIZE) + 1
nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1

# Initialize model, loss, and optimizer
net = NeuralNetwork(
    d_model=d_model,
    d_hidden=d_hidden,
    d_input=d_input,
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
    patience=3, verbose=True, path=f"{MODEL_DIR}/nn_weights.pth"
)

best_score = 0
best_auroc = 0
best_auprc = 0

for epoch in range(EPOCH):
    train_loss = 0
    # total_accuracy = 0
    count = 0
    y_true = np.zeros((len(X_train)))
    y_true_class = np.zeros((len(X_train), d_output))
    y_pred = np.zeros((len(X_train)))
    y_pred_prob = np.zeros((len(X_train), d_output))
    for patient in tqdm.trange(0, len(X_train), BATCH_SIZE):

        inputs = Variable(X_train[patient:patient+BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_train[patient : patient + BATCH_SIZE]).to(DEVICE)
        optimizer.zero_grad()
        pred_y = net(inputs)

        loss = 0

        for i in range(d_output):

            loss_class = loss_bin(pred_y[:,i], labels[:,i]) * class_weights[i]
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

    roc_aucs = []

    for i in range(d_output):

        roc_auc = roc_auc_score(y_true_class[:,i], y_pred_prob[:,i])

        roc_aucs.append(roc_auc)

    print(f"train_roc_auc: {roc_aucs}")

    pr_aucs = []

    for i in range(d_output):

        precision, recall, _ = precision_recall_curve(
            y_true_class[:,i], y_pred_prob[:,i]
        )
        pr_auc = auc(recall, precision)

        pr_aucs.append(pr_auc)

    print(f"train_pr_auc: {pr_aucs}")

    y_true = np.zeros((len(X_val)))
    y_true_class = np.zeros((len(X_val), d_output))
    y_pred = np.zeros((len(X_val)))
    y_pred_prob = np.zeros((len(X_val), d_output))
    val_loss = 0
    for patient in range(0, len(X_val), BATCH_SIZE):

        inputs = Variable(X_val[patient:patient+BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_val[patient : patient + BATCH_SIZE]).to(DEVICE)

        pred_y = net(inputs)

        loss = 0

        for i in range(d_output):

            loss_class = loss_bin(pred_y[:,i], labels[:,i]) * class_weights[i]
            loss = loss + loss_class

        val_loss += loss.data

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

    print("-" * 40)
    print("Val performance")

    print("val_loss: {}".format(val_loss / nb_batches_val))

    roc_aucs = []

    for i in range(d_output):

        roc_auc = roc_auc_score(y_true_class[:,i], y_pred_prob[:,i])

        roc_aucs.append(roc_auc)

    print(f"val_roc_auc: {roc_aucs}")

    pr_aucs = []

    for i in range(d_output):

        precision, recall, _ = precision_recall_curve(
            y_true_class[:,i], y_pred_prob[:,i]
        )
        pr_auc = auc(recall, precision)

        pr_aucs.append(pr_auc)

    print(f"val_pr_auc: {pr_aucs}")

    roc_pr_auc = np.mean(roc_aucs) + np.mean(pr_aucs)

    if roc_pr_auc > best_score:
        best_score = roc_pr_auc

    early_stopping(roc_pr_auc, net)

    if early_stopping.early_stop:
        print("Early stopping")
        break


model_architecture = {
    "d_model": d_model,
    "d_hidden": d_hidden,
    "d_input": d_input,
    "d_output": d_output,
    "n_layer": n_layer,
    "dropout": dropout,
}

torch.save(model_architecture, f"{MODEL_DIR}/nn_architecture.pth")