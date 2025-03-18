#%%
# Import libraries

import pandas as pd
import numpy as np
import h5py
import pickle
import os

from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

from variables import MODEL_DIR


if not os.path.exists(f"{MODEL_DIR}/results"):
    os.makedirs(f"{MODEL_DIR}/results")

#%%

# Load data

with h5py.File("%s/dataset.h5" % MODEL_DIR, "r") as f:
    data = f["train"]
    X_train = data["X"][:]
    y_train = data["y"][:]

    data = f["val"]
    X_val = data["X"][:]
    y_val = data["y"][:]

# Shuffle training data

from sklearn.utils import shuffle

X_train, y_train = shuffle(
    X_train, y_train
)

#%%

# Tune hyperparameters

num_classes = y_train.shape[1]

# Define the parameter grid
param_grid = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7, 9]
}

# Define early stopping parameters
early_stopping_rounds = 50

# Initialize a CatBoostClassifier with early stopping
model = CatBoostClassifier(task_type="GPU", silent=True, early_stopping_rounds=early_stopping_rounds)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc')

# Fit GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train[:, 0], eval_set=(X_val, y_val[:, 0]))  # Use the first class for hyperparameter tuning

# Get the best parameters
best_params = grid_search.best_params_

print(best_params)

#%%

# Create a list to store binary classifiers
binary_classifiers = []
num_classes = y_train.shape[1]

# Update the model with the best parameters
model = CatBoostClassifier(task_type="GPU", **best_params)

# Train one-vs-rest binary classifiers for each class
for class_idx in range(num_classes):
    # Initialize CatBoost Classifier with the best parameters
    binary_model = CatBoostClassifier(task_type="GPU", **best_params, silent=True, early_stopping_rounds=early_stopping_rounds)

    # Train the binary classifier for the current class
    binary_model.fit(X_train, y_train[:, class_idx])

    # Add the trained binary classifier to the list
    binary_classifiers.append(binary_model)


#%%

# Save model

import pickle

with open("%s/catboost.pkl" % MODEL_DIR, "wb") as f:
    pickle.dump(binary_classifiers, f, protocol=2)

# %%
