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

if not os.path.exists(f"{MODEL_DIR}/results"):
    os.makedirs(f"{MODEL_DIR}/results")

import torch
import math
import torch.nn as nn
from concare import ConCare

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"use DEVICE: {DEVICE}")

model_architecture = torch.load(
    f"{MODEL_DIR}/apricott_architecture.pth"
)

input_dim = model_architecture["input_dim"]
static_dim = model_architecture["static_dim"]
hidden_dim = model_architecture["hidden_dim"]
d_ff = model_architecture["d_ff"]
d_output = model_architecture["output_dim"]
MHD_num_head = model_architecture["MHD_num_head"]

model = ConCare(input_dim = input_dim, static_dim = static_dim, hidden_dim = hidden_dim, d_model = hidden_dim,  MHD_num_head = MHD_num_head , d_ff = d_ff, output_dim = d_output).to(DEVICE)


# Load model weights

model.load_state_dict(torch.load(f"{MODEL_DIR}/apricott_weights.pth"))


# Load best parameters

import pickle

with open("%s/best_params.pkl" % MODEL_DIR, "rb") as f:
    best_params = pickle.load(f)

print(f"Best Parameters: {best_params}")

BATCH_SIZE = best_params["batch_size"]
# Validation

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable


test_sets = ["internal", "external_1", "external_2"]

for test_set in test_sets:

    # Load data

    with h5py.File("%s/dataset.h5" % MODEL_DIR, "r") as f:
        data = f[test_set]
        X_test = data["X"][:]
        static_test = data["static"][:]
        y_test = data["y"][:]

    # Convert data to tensors

    X_test = torch.FloatTensor(X_test)
    static_test = torch.FloatTensor(static_test)
    y_test = torch.FloatTensor(y_test)

    y_true_class = np.zeros((len(X_test), d_output))
    y_pred_prob = np.zeros((len(X_test), d_output))
    for patient in range(0, len(X_test), BATCH_SIZE):

        inputs = Variable(X_test[patient : patient + BATCH_SIZE]).to(DEVICE)
        static_input = Variable(static_test[patient : patient + BATCH_SIZE]).to(DEVICE)
        labels = Variable(y_test[patient : patient + BATCH_SIZE]).to(DEVICE)

        pred_y = model(inputs, static_input)

        y_true_class[patient : patient + BATCH_SIZE, :] = labels.to("cpu").numpy()
        y_pred_prob[patient : patient + BATCH_SIZE, :] = (
            pred_y.to("cpu").detach().numpy()
        )

    print("-" * 40)
    print(f"Test performance: {test_set}")

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

    outcomes = [f"delirium_{i+1}" for i in range(d_output)]

    for i in range(len(outcomes)):

        results = pd.DataFrame(np.concatenate([y_true_class[:,i:i+1], y_pred_prob[:,i:i+1]], axis=1), columns=["true", "pred"])

        results.to_csv(
            f"{MODEL_DIR}/results/{test_set}_{outcomes[i]}_results.csv", index=None
        )

