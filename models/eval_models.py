# Import libraries

import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
    auc,
)
import math
import re
from scipy.stats import t

import os

#%%

DATA_DIR = "/blue/parisa.rashidi/contreras.miguel/clinical_notes/main3/models"

modalities = ["ehr", "ehr_text"]
# modalities = ["ehr"]

exclude = ["OLD", "llama3_instruct", "meditron_instruct", "catboost", "meditron", "llama3", "gatortron_large_mlm", "dellirium"]

outcomes = ["delirium_1"]

test_sets = ["internal", "external_1", "external_2"]

n_iterations = 200

for outcome in outcomes:
    
    all_models = []

    all_results = pd.DataFrame()

    for modality in modalities:

        models = os.listdir(f"{DATA_DIR}/{modality}")

        for model in models:
            
            if model not in exclude:
                
                all_tests = pd.DataFrame()
                
                for test_set in test_sets:

                    results = pd.read_csv(f"{DATA_DIR}/{modality}/{model}/results/{test_set}_{outcome}_results.csv")


                    model_probs = results["pred"].values
                    model_true = results["true"].values

                    auroc = []
                    auprc = []

                    for i in range(n_iterations):

                        random = np.random.choice(len(model_true), len(model_true), replace=True)

                        sample_true = model_true[random]
                        sample_pred = model_probs[random]

                        while (sample_true == 0).all(axis=0).any():
                            random_sample = np.random.choice(
                                len(model_true), len(model_true), replace=True
                            )

                            sample_true = model_true[random_sample]
                            sample_pred = model_probs[random_sample]

                        ind_auroc = roc_auc_score(
                            sample_true, sample_pred
                        )

                        prec, rec, _ = precision_recall_curve(
                            sample_true, sample_pred
                        )
                        ind_auprc = auc(rec, prec)

                        auroc.append(ind_auroc)

                        auprc.append(ind_auprc)

                        print(f"Iteration {i+1}")

                    metrics_names = [f"AUROC_{test_set}", f"AUPRC_{test_set}"]

                    auroc = pd.DataFrame(data=auroc, columns=[f"AUROC_{test_set}"])
                    auprc = pd.DataFrame(data=auprc, columns=[f"AUPRC_{test_set}"])

                    auroc_sum = auroc.apply(
                        lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                        axis=0,
                    ).values[0]
                    auprc_sum = auprc.apply(
                        lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                        axis=0,
                    ).values[0]
                    
                    print(auroc_sum)
                    print(auprc_sum)

                    # metrics = pd.concat(
                    #     [
                    #         auroc_sum,
                    #         auprc_sum,
                    #     ],
                    #     axis=1,
                    # )
                    
                    metrics = pd.DataFrame(data=[[str(auroc_sum), str(auprc_sum)]], columns=metrics_names, index=[f"{model}_{modality}"])

                    # metrics.columns = metrics_names

                    print(metrics)
                    
                    all_tests = pd.concat([all_tests, metrics], axis=1)

                all_tests["model"] = model
                all_tests["modality"] = modality
                
                all_results = pd.concat([all_results, all_tests], axis=0)

                all_models.append(model)


    # sup = ["a", "b", "c", "d", "e"]
    sup = ["a", "b", "c", "d"]

    def get_super(x):
        normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
        super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
        res = x.maketrans("".join(normal), "".join(super_s))
        return x.translate(res)


    def extract_value(s):

        matches = re.findall(r"\d+\.\d+", s)

        mean = float(matches[0])
        ci = float(matches[2]) - float(matches[1])

        return mean, ci

    # comparisons = ["nn", "transformer", "bag_words", "bow_nn", "bow_transfo"]
    comparisons = ["nn", "transformer", "mamba"]
    # comparisons = ["catboost", "nn"]
    # comparisons = ["catboost"]

    test_sets = ["internal", "external_1", "external_2"]

    # comparisons = [
    #     ["nn", "bag_words", "bow_nn"],
    #     ["transformer", "bag_words", "bow_transfo"],
    #     ["nn", "clinicalbert", "clinicalbert_nn"],
    #     ["transformer", "clinicalbert", "clinicalbert_transfo"],
    #     ["nn", "gatortron", "gatortron_nn"],
    #     ["transformer", "gatortron", "gatortron_transfo"],
    #     ["nn", "meditron", "meditron_nn"],
    #     ["transformer", "meditron", "meditron_transfo"],
    # ]

    metrics = ["AUROC", "AUPRC"]
    
    print(all_results)
    
    count = 0

    for comp in comparisons:

        for model in all_models:

            for metric in metrics:
                
                for test_set in test_sets:

                    superscript = sup[count]

                    sample_1 = all_results.loc[(all_results["model"] == comp), f"{metric}_{test_set}"].values[0]
                    sample_2 = all_results.loc[(all_results["model"] == model), f"{metric}_{test_set}"].values[0]

                    mean_1, ci_1 = extract_value(sample_1)
                    mean_2, ci_2 = extract_value(sample_2)

                    std_1 = (math.sqrt(n_iterations) * ci_1) / 3.92
                    std_2 = (math.sqrt(n_iterations) * ci_2) / 3.92

                    dof = n_iterations + n_iterations - 2

                    if std_1 > 0 or std_2 > 0:

                        t_statistic = (mean_1 - mean_2) / (
                            (std_1**2 / n_iterations) + (std_2**2 / n_iterations)
                        ) ** 0.5
                        p_value = 2 * (1 - t.cdf(abs(t_statistic), dof))

                    elif mean_2 > mean_1:

                        p_value = 0

                    else:

                        p_value = 1.00

                    if p_value < 0.05:

                        all_results.loc[(all_results["model"] == model), f"{metric}_{test_set}"] = (
                            all_results.loc[(all_results["model"] == model), f"{metric}_{test_set}"].values[0] + "\u2E34" + get_super(superscript)
                        )
                        

        count += 1


    # Regular expression pattern to match text after closing parenthesis
    pattern = r"\)([^\s])"

    # Function to replace matched text with a space
    def replace_text(value):
        return re.sub(pattern, r") ", value)

    # Apply the replacement operation to each element in the dataframe
    
    all_aurocs = [f"AUROC_{test_set}" for test_set in test_sets]
    all_auprcs = [f"AUPRC_{test_set}" for test_set in test_sets]
    
    all_metrics = all_aurocs + all_auprcs

    all_results.loc[:,all_metrics] = all_results.loc[:,all_metrics].applymap(replace_text)

    all_results["modality"] = pd.Categorical(all_results["modality"], categories=modalities, ordered=True)
    
    all_results = all_results.sort_values(by=["modality", "model"])

    print(all_results)

    with open(f"{DATA_DIR}/{outcome}_results_table.html", 'w') as f:
        f.write(all_results.to_html())

    all_results.to_csv(f"{DATA_DIR}/{outcome}_results_table.csv", index=None)
    
