#%%
# Import libraries

import pandas as pd
import numpy as np
import h5py
import pickle
import os

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


# %%

# Shuffle training data

from sklearn.utils import shuffle

X_train, y_train = shuffle(
    X_train, y_train
)

#%%

# Train catboost model

from catboost import CatBoostClassifier

# Create a list to store binary classifiers
binary_classifiers = []

num_classes = y_train.shape[1]

# Train one-vs-rest binary classifiers for each class
for class_idx in range(num_classes):
    # Create a binary target variable for the current class
    # Initialize CatBoost Classifier for binary classification
    model = CatBoostClassifier(task_type="GPU")

    # Train the binary classifier for the current class
    model.fit(X_train, y_train[:, class_idx])

    # Add the trained binary classifier to the list
    binary_classifiers.append(model)

#%%

# Save model

import pickle

with open("%s/catboost.pkl" % MODEL_DIR, "wb") as f:
    pickle.dump(binary_classifiers, f, protocol=2)
