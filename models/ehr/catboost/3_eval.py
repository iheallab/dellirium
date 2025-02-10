# Import libraries

import pandas as pd
import numpy as np
import h5py
import os
import optuna
import tqdm
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pickle

#%%

from variables import MODEL_DIR

#%%

# Create directory for saving model

if not os.path.exists(f"{MODEL_DIR}/results"):
    os.makedirs(f"{MODEL_DIR}/results")

with open("%s/catboost.pkl" % MODEL_DIR, "rb") as f:
    binary_classifiers = pickle.load(f)

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch.autograd import Variable

test_sets = ["internal", "external_1", "external_2"]

for test_set in test_sets:

    # Load data

    with h5py.File("%s/dataset.h5" % MODEL_DIR, "r") as f:
        data = f[test_set]
        X_test = data["X"][:]
        y_test = data["y"][:]
    
    d_output = y_test.shape[1]
    
    class_probabilities = []
    
    y_true_class = np.zeros((len(X_test), d_output))
    y_pred_prob = np.zeros((len(X_test), d_output))
    count = 0
    for model in binary_classifiers:
        # Predict the probability of positive class (1) for each sample in the test set
        y_pred = model.predict_proba(X_test)[:, 1]
        y_pred_prob[:, count] = y_pred
        y_true_class[:, count] = y_test[:, count]

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

