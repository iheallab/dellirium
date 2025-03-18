
# Import libraries
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import t
from scipy.stats import norm
import math
from plotly.subplots import make_subplots


MAIN_DIR = '.../clinical_notes/main3'

ANALYSIS_DIR = f'{MAIN_DIR}/analyses/temporal'
DATA_DIR = f"{MAIN_DIR}/final_data"
MODEL_DIR = f'{MAIN_DIR}/models'

best_models = ['transformer', 'llama31', 'gatortron_s_mlm']

modalities = ['ehr', 'ehr_text']

if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR)

n_iterations = 200

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

    # Calculate confidence intervals
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)

    return roc_auc, ci_lower, ci_upper    

times_delirium = pd.read_csv(f"{ANALYSIS_DIR}/delirium_times.csv")

outcomes = pd.read_csv(f"{DATA_DIR}/outcomes.csv")

times_delirium = times_delirium[times_delirium["icustay_id"].isin(outcomes["icustay_id"].unique())]

times_delirium = outcomes.merge(times_delirium[["icustay_id", "days"]], on="icustay_id", how="left")

times_delirium["days"] = times_delirium["days"].fillna(0)

print(times_delirium[times_delirium["delirium"] == 1]["days"].value_counts())
print(times_delirium[times_delirium["delirium"] == 0]["days"].value_counts())

outcomes_list = ["delirium_1"]

test_sets = ["total_external"]

labels = ["Transformer", "LLaMa-3.1", "DeLLiriuM"]

for test_set in test_sets:
    
    days = [2, 3, 4, 5, 6, 7]
    data = {label: {} for label in labels}

    with open("%s/ids.pkl" % DATA_DIR, "rb") as f:
        ids = pickle.load(f)
        ids_test_1 = ids["external_1"]
        ids_test_2 = ids["external_2"]
        ids_test = ids_test_1 + ids_test_2
    
    
    notes = pd.read_csv(f'{MODEL_DIR}/ehr_text/gatortron_s_mlm/ehr_text.csv')
    notes = notes.merge(outcomes, how="inner", on="icustay_id")
    
    notes_test_external1 = notes[notes["icustay_id"].isin(ids_test_1)]
    notes_test_external2 = notes[notes["icustay_id"].isin(ids_test_2)]
    
    notes_test = pd.concat([notes_test_external1, notes_test_external2], axis=0).reset_index(drop=True)
    
    
    # notes_test = notes_test.set_index("icustay_id")
    # notes_test = notes_test.loc[ids_test].reset_index()
    
    # times_delirium = times_delirium.set_index("icustay_id")
    # times_delirium = times_delirium.loc[ids_test].reset_index()

    icu_ids = notes_test["icustay_id"].tolist()

    model_outputs = []
    
    for outcome in outcomes_list:
        
        count_model = 0

        for modality in modalities:
            
            models = os.listdir(f"{MODEL_DIR}/{modality}")

            for model in best_models:

                if model in models:
                    
                    results_external1 = pd.read_csv(f"{MODEL_DIR}/{modality}/{model}/results/external_1_{outcome}_results.csv")
                    results_external2 = pd.read_csv(f"{MODEL_DIR}/{modality}/{model}/results/external_2_{outcome}_results.csv")
                    
                    results = pd.concat([results_external1, results_external2], axis=0).reset_index(drop=True)
                    
                    results["icustay_id"] = icu_ids
                                                            
                    results = results.merge(times_delirium[["icustay_id", "days"]], on="icustay_id", how="inner")
                    
                    print(f"{model}")
                    
                    count = 2
                    
                    delirium_day_incidence = {}

                    for day in days:
                        
                        print(f"day {count}")

                        results_week = results[((results["days"] == day)) | (results["days"] == 0)]
                        delirium_incidence = (results_week["true"].sum() / len(results_week)) * 100
                        print(f"Delirium incidence on day {day}: {delirium_incidence:.2f}")
                        
                        model_probs = results_week["pred"].values
                        model_true = results_week["true"].values
                        
                        roc_auc, ci_lower, ci_upper = calculate_roc_with_ci(model_probs, model_true)
                        
                        print(f"AUROC {roc_auc*100:.1f} ({ci_lower*100:.1f}-{ci_upper*100:.1f})")
                        
                        data[labels[count_model]][f"Day {count}"] = f"{roc_auc*100:.1f} ({ci_lower*100:.1f}-{ci_upper*100:.1f})"
                        delirium_day_incidence[f"Day {count}"] = delirium_incidence
                        
                        count += 1
                    
                    print("-"*40)
                    
                    count_model += 1

    # Extract days
    days = list(data['Transformer'].keys())

    # Initialize dictionaries to store means and errors
    means = {model: [] for model in data}
    errors = {model: [] for model in data}

    # Parse the data
    for model, results in data.items():
        for day, result in results.items():
            mean_str, interval_str = result.split(' ')
            mean = float(mean_str)
            lower, upper = map(float, interval_str[1:-1].split('-'))
            error = (upper - lower) / 2  # Calculate the error as half the confidence interval

            means[model].append(mean)
            errors[model].append(error)

    # Define colors for each model
    colors = {
        "Transformer": "#50C4ED",
        "LLaMa-3.1": "#387ADF",
        "DeLLiriuM": "#FBA834",
    }

    # Number of models
    n_models = len(data)
    x = np.arange(len(days))  # X-axis positions
    width = 0.25  # Width of bars

    # Initialize Plotly Figure
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot each model
    for i, (model, color) in enumerate(colors.items()):
        y_with_errors = [mean + error for mean, error in zip(means[model], errors[model])]

        fig.add_trace(
            go.Bar(
                name=model,
                x=days,
                y=means[model],
                error_y=dict(
                    type="data",
                    array=errors[model],
                    visible=True
                ),
                marker_color=color,
                width=width,
            ),
            secondary_y=False  # Bars use primary y-axis
        )
        
    fig.add_trace(
        go.Scatter(
            name="Delirium Incidence",
            x=days,
            y=list(delirium_day_incidence.values()),
            mode='lines+markers',
            line=dict(color='black', dash='dash'),  # Customize style as desired
            marker=dict(size=8),
            yaxis='y2'
        ),
        secondary_y=True
    )

    # Update layout to limit y-axis range
    fig.update_yaxes(range=[0, 5], secondary_y=True)

    # Function to perform t-tests and add significance annotations
    def add_significance_annotations(fig, x, means, errors, width):
        n_iterations = 100  # Adjust based on your dataset

        for i, day in enumerate(days):
            
            # Get model with highest performance
            
            all_means = [means[model][i] for model in means]
            best_model_index = len(all_means) - 1 - np.argmax(all_means[::-1])
            best_model_name = list(means.keys())[best_model_index]
            
            print(f"Best model for {day}: {best_model_name} ({best_model_index})")
            
            
            best_model_mean = means[best_model_name][i]
            best_model_error = errors[best_model_name][i]

            models_to_compare = [model for model in means.keys() if model != best_model_name]
            p_values = []

            for model in models_to_compare:
                model_mean = means[model][i]
                model_error = errors[model][i]

                std_1 = (math.sqrt(n_iterations) * best_model_error) / 3.92
                std_2 = (math.sqrt(n_iterations) * model_error) / 3.92

                dof = n_iterations + n_iterations - 2

                t_statistic = (best_model_mean - model_mean) / (
                    (std_1**2 / n_iterations) + (std_2**2 / n_iterations)
                ) ** 0.5

                p_value = 2 * (1 - t.cdf(abs(t_statistic), dof))
                p_values.append(p_value)

            if sum(p < 0.05 for p in p_values) == 3:
                sig_text = "***"
            elif sum(p < 0.05 for p in p_values) == 2:
                sig_text = "**"
            elif sum(p < 0.05 for p in p_values) == 1:
                sig_text = "*"
            else:
                sig_text = ""
                
            coeff = {
                "Transformer": 1.2,
                "LLaMa-3.1": 0.2,
                "DeLLiriuM": 1.05,
            }
                
            x_offset = +coeff[best_model_name]*(best_model_index-1) if best_model_index > 1 else -coeff[best_model_name]*(best_model_index+1)

            if sig_text:
                fig.add_annotation(
                    x=x[i] + width*x_offset,
                    y=best_model_mean + best_model_error + 8,
                    text=sig_text,
                    showarrow=False,
                    font=dict(size=20, color="black"),
                )
            
            for model in means.keys():
                mean = means[model]
                error = errors[model]
                fig.add_annotation(
                    x=x[i] + width*(list(means.keys()).index(model)-1),
                    y=mean[i] + error[i] + 3.5,
                    text=f"{mean[i]:.1f}",
                    showarrow=False,
                    font=dict(size=18, color="black"),
                )

    # Add significance annotations
    add_significance_annotations(fig, x, means, errors, width)

    fig.update_layout(
        yaxis=dict(
            title=dict(text="AUROC", font=dict(size=20)),
            range=[0, 100],
            tickfont=dict(size=20)
        ),
        yaxis2=dict(
            title=dict(text="Delirium Incidence (%)", font=dict(size=20)),
            overlaying='y',
            side='right',
            tickfont=dict(size=20)
        ),
        xaxis=dict(
            title=dict(text="Days", font=dict(size=20)),
            tickfont=dict(size=20),
            tickmode="array",
            tickvals=list(range(len(days))),
            ticktext=days
        ),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5,
            font=dict(size=20),
        ),
        template="plotly_white",
        width=1800,
        height=600,
    )

    
    fig.write_image(
        f'{ANALYSIS_DIR}/auroc_comparison_{test_set}.png'
    )

    # Show plot
    fig.show()
