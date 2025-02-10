import plotly.graph_objects as go
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

MAIN_DIR = "/blue/parisa.rashidi/contreras.miguel/clinical_notes/main3"
best_models = ["transformer", "llama31", "gatortron_s_mlm"]
modalities = ["ehr", "ehr_text"]

DATA_DIR = f"{MAIN_DIR}/final_data"
MODEL_DIR = f"{MAIN_DIR}/models"
ANALYSIS_DIR = f"{MAIN_DIR}/analyses/roc_curves"

if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR)

n_iterations = 200  # Bootstrap iterations

def calculate_roc_with_ci(predicted_scores, true_labels, n_bootstraps=n_iterations):
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)

    # Bootstrapping for confidence intervals
    aucs = []
    fprs_tprs = []
    for _ in range(n_bootstraps):
        bootstrap_indices = resample(range(len(predicted_scores)))
        bootstrapped_scores = predicted_scores[bootstrap_indices]
        bootstrapped_labels = true_labels[bootstrap_indices]
        fpr_bootstrap, tpr_bootstrap, _ = roc_curve(
            bootstrapped_labels, bootstrapped_scores
        )
        aucs.append(auc(fpr_bootstrap, tpr_bootstrap))
        fprs_tprs.append([fpr_bootstrap, tpr_bootstrap])

    # Confidence intervals
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)

    return fpr, tpr, roc_auc, fprs_tprs, ci_lower, ci_upper


# **Plot ROC Curves Using Plotly**
def plot_roc_with_ci_plotly(models, true_labels, labels, colors, cohort):
    fig = go.Figure()

    for model, label in zip(models, labels):
        fpr, tpr, roc_auc, fprs_tprs, ci_lower, ci_upper = calculate_roc_with_ci(
            model, true_labels
        )

        fprs_tprs_ind = fprs_tprs[0][1]

        fpr_mean = np.linspace(0, 1, len(fprs_tprs_ind))
        interp_tprs = []
        for i in range(len(fprs_tprs)):
            fpr_ind = fprs_tprs[i][0]
            tpr_ind = fprs_tprs[i][1]
            interp_tpr = np.interp(fpr_mean, fpr_ind, tpr_ind)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_mean[-1] = 1.0
        tpr_std = 2 * np.std(interp_tprs, axis=0)
        upper_bound = np.clip(tpr_mean + tpr_std, 0, 1)
        lower_bound = tpr_mean - tpr_std

        # **Add Mean ROC Curve**
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{label} (AUC = {roc_auc*100:.1f} [{ci_lower*100:.1f}-{ci_upper*100:.1f}])",
                line=dict(color=colors[label], width=2),
            )
        )

        # **Add Confidence Interval as Shaded Region**
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([fpr_mean, fpr_mean[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill="toself",
                fillcolor=f"rgba{tuple(int(colors[label][1:][i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"{label} CI",
                showlegend=False,
            )
        )

    # **Diagonal Baseline (Random Classifier)**
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="No Skill Model (AUC = 50.0)",
        )
    )

    # **Customize Layout**
    fig.update_layout(
        xaxis=dict(title="1 - Specificity (False Positive Rate)", titlefont=dict(size=16), tickfont=dict(size=16)),
        yaxis=dict(title="Sensitivity (True Positive Rate)", titlefont=dict(size=16), tickfont=dict(size=16)),
        legend=dict(
            x=1,  # Position legend inside lower right
            y=0,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",  # Add semi-transparent white background
            font=dict(size=16),
        ),
        template="plotly_white",
        width=700,
        height=600,
    )


    fig.write_image(f"{ANALYSIS_DIR}/roc_curve_{cohort}.png")
    
    # **Show Plot**
    fig.show()


outcomes = ["delirium_1"]
test_sets = ["total_external"]

labels = ["Transformer", "LLaMa-3.1", "DeLLiriuM"]
colors = {
    "Transformer": "#50C4ED",
    "LLaMa-3.1": "#387ADF",
    "DeLLiriuM": "#FBA834",
}

for test_set in test_sets:
    
    model_outputs = []
    true_labels_external_1 = pd.read_csv(f"{MODEL_DIR}/ehr_text/gatortron_s_mlm/results/external_1_delirium_1_results.csv")["true"].values
    true_labels_external_2 = pd.read_csv(f"{MODEL_DIR}/ehr_text/gatortron_s_mlm/results/external_2_delirium_1_results.csv")["true"].values
    
    true_labels = np.concatenate([true_labels_external_1, true_labels_external_2], axis=0)
    
    for outcome in outcomes:

        for modality in modalities:
            
            models = os.listdir(f"{MODEL_DIR}/{modality}")

            for model in best_models:

                if model in models:
                    
                    results_external1 = pd.read_csv(f"{MODEL_DIR}/{modality}/{model}/results/external_1_{outcome}_results.csv")
                    results_external2 = pd.read_csv(f"{MODEL_DIR}/{modality}/{model}/results/external_2_{outcome}_results.csv")
                    
                    results = pd.concat([results_external1, results_external2], axis=0)

                    model_probs = results["pred"].values
                    model_outputs.append(model_probs)
    
    plot_roc_with_ci_plotly(model_outputs, true_labels, labels, colors, test_set)
