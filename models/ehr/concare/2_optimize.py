#%%

# Import libraries

import pandas as pd
import numpy as np
import h5py
import os

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
from concare import ConCare

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

d_input = X_train.shape[2]
d_static = static_train.shape[1]
d_output = y_train.shape[1]

for i in range(d_output):

    negative = np.unique(y_train[:, i], return_counts=True)[1][0]
    positive = np.unique(y_train[:, i], return_counts=True)[1][1]

    class_weight = negative / positive

    # class_weight = torch.tensor(negative/positive)
    class_weights.append(class_weight)

from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

loss_bin = BCELoss()
loss_bin_mort = BCEWithLogitsLoss()


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        # self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, roc_pr_auc, model):

        if self.best_score is None:
            self.best_score = roc_pr_auc
        elif roc_pr_auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = roc_pr_auc
            self.counter = 0



from torch import optim
from torch.autograd import Variable
import tqdm
import optuna

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

# Function to apply Kaiming initialization to all parameters
def kaiming_init(m):
    for param in m.parameters():
        if param.dim() > 1:  # Apply only to weights, not biases
            nn.init.kaiming_normal_(param, nonlinearity='relu')
        else:
            nn.init.constant_(param, 0)
            

# Define objective function
def objective(trial):
    # Define hyperparameters to be optimized
    input_dim = d_input
    static_dim = d_static
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    MHD_num_head = trial.suggest_categorical("num_head", [2, 4, 8])
    # d_ff = int(MHD_num_head * hidden_dim)
    d_ff = trial.suggest_categorical("d_ff", [64, 128, 256])
    LR = trial.suggest_loguniform("learning_rate", 5e-4, 1e-3)
    BATCH_SIZE = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    nb_batches = int(len(X_train) / BATCH_SIZE) + 1
    nb_batches_val = int(len(X_val) / BATCH_SIZE) + 1
    
    net = ConCare(input_dim = input_dim, static_dim = static_dim, hidden_dim = hidden_dim, d_model = hidden_dim,  MHD_num_head = MHD_num_head , d_ff = d_ff, output_dim = d_output).to(DEVICE)

    optimizer_name = "Adam"

    if optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=LR, weight_decay=1e-5)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    elif optimizer_name == "RMS":
        optimizer = optim.RMSprop(net.parameters(), lr=LR, weight_decay=1e-5)

    early_stopping = EarlyStopping(patience=2, verbose=True)

    best_score = 0
    best_auroc = 0
    best_auprc = 0

    # net.apply(kaiming_init)
    
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
            static_input = Variable(static_train[patient : patient + BATCH_SIZE]).to(
                DEVICE
            )
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

            inputs = Variable(X_val[patient:patient+BATCH_SIZE]).to(DEVICE)
            static_input = Variable(static_val[patient : patient + BATCH_SIZE]).to(
                DEVICE
            )
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

    return best_score


# Create study
study = optuna.create_study(direction="maximize")

# Optimize hyperparameters
study.optimize(objective, n_trials=10)

# Get best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)


import pickle

with open("%s/best_params.pkl" % MODEL_DIR, "wb") as f:
    pickle.dump(best_params, f, protocol=2)

