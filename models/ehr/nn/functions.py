import torch
import numpy as np
import tqdm
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path=""):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, roc_pr_auc, model):

        if self.best_score is None:
            self.best_score = roc_pr_auc
            self.save_checkpoint(roc_pr_auc, model)
        elif roc_pr_auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(roc_pr_auc, model)
            self.best_score = roc_pr_auc
            self.counter = 0

    def save_checkpoint(self, roc_pr_auc, model):
        if self.verbose:
            print(
                f"Validation AUROC-AUPRC increased ({self.best_score:.6f} --> {roc_pr_auc:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        
        
# class Trainer:
#     def __init__(self, epochs=5, batch_size=32, early_stop=early_stop, optimizer=optimizer, loss=loss_func):
#         self.batch_size = batch_size
#         self.early_stop = early_stop
#         self.optimizer = optimizer
#         self.loss_func = loss_func
        
#     def __call__(self, X_train, model):
    
