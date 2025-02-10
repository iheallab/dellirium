
#%%

import plotly.io as pio
pio.templates.default = "plotly_white"

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import roc_curve, auc

# Main Directories
MAIN_DIR = "/blue/parisa.rashidi/contreras.miguel/clinical_notes/main3"
best_models = ["transformer", "llama31", "gatortron_s_mlm"]
modalities = ["ehr", "ehr_text"]

DATA_DIR = f"{MAIN_DIR}/final_data"
MODEL_DIR = f"{MAIN_DIR}/models"
ANALYSIS_DIR = f"{MAIN_DIR}/analyses/model_performance"

if not os.path.exists(ANALYSIS_DIR):
    os.makedirs(ANALYSIS_DIR)

# Define test sets and model details
outcomes = ["delirium_1"]
test_sets = ["total_external"]

labels = ["Transformer", "LLaMa-3.1", "DeLLiriuM"]
colors = {
    "Transformer": "#50C4ED",
    "LLaMa-3.1": "#387ADF",
    "DeLLiriuM": "#FBA834",
}
parameters = {
    "Transformer": 33,
    "LLaMa-3.1": 8000,
    "DeLLiriuM": 345,
}

# Collect AUROC data
data = []
cohort_names = {
    "external_1": "External 1 (MIMIC-IV)",
    "external_2": "External 2 (eICU)",
    "total_external": "External (MIMIC + eICU)",
}

for test_set in test_sets:
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
                    fpr, tpr, _ = roc_curve(true_labels, model_probs)
                    roc_auc = auc(fpr, tpr)
                    model_name = labels[best_models.index(model)]
                    data.append({
                        "Dataset": cohort_names[test_set],
                        "AUROC": roc_auc*100,
                        "Model": model_name,
                        "Parameters": parameters[model_name],
                        "Color": colors[model_name]
                    })

# Create DataFrame
df = pd.DataFrame(data)

scale_factor = 10  # Adjust this value to scale bubble sizes

# Scale bubble size for better visibility
df["BubbleSize"] = (df["Parameters"] / max(df["Parameters"]) * 100) * scale_factor

# Choose a max diameter in pixels for your largest bubble
max_bubble_diameter_px = 70

# Let’s say your bubble sizes are in df["BubbleSize"] 
max_value = df["BubbleSize"].max()

sizeref = max_value / max_bubble_diameter_px


print(sorted(list(df["BubbleSize"].unique())))

# Create Bubble Plot
fig = px.scatter(
    df,
    x="AUROC",
    y="Dataset",
    size="BubbleSize",
    color="Model",
    hover_data=["Model", "Parameters"],
    labels={"AUROC": "AUROC Score", "Dataset": "Dataset"},
    size_max=100,
    color_discrete_map=colors,  # Map colors to models
)

label_pos_x = 1.205
offset_x = 0.1

# Custom legend parameters
legend_sizes = sorted(list(df["BubbleSize"].unique()))[:3] / np.float64(1.4)
legend_labels = ["1M", "345M", "1B"]
legend_positions = [(1.034, 0.33), (1.033, 0.25), (1.03, 0.15)]  # (x, y) positions
legend_labels_positions = [(label_pos_x - offset_x, 0.355), (label_pos_x - offset_x, 0.255), (label_pos_x - offset_x, 0.1)]  # (x, y) positions


fig.add_annotation(
    x=label_pos_x,
    y=0.45,
    xref="paper",
    yref="paper",
    text="Parameters",
    showarrow=False,
    font=dict(size=14)
)

# Add custom legend elements
for size, label, pos, label_pos in zip(legend_sizes, legend_labels, legend_positions, legend_labels_positions):
    
    # Add text annotation
    fig.add_annotation(
        x=label_pos[0],
        y=label_pos[1],
        xref="paper",
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=12)
    )


# Adjust layout to position legends if needed
fig.update_layout(
    legend=dict(
        title=dict(text="Model"),
        font=dict(size=12),
        x=1.02,  # Position main legend outside to the right
        y=1
    ),
    xaxis=dict(range=[75, 85], title="AUROC"),
    yaxis=dict(title="", categoryorder="total ascending", showticklabels=False),
)


fig.write_image(f"{ANALYSIS_DIR}/model_performance_bubble_plot.png")


# Show Plot
fig.show()


# %%
