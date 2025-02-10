#%%
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

exclude = ["OLD"]

outcomes = ["delirium_1"]

test_sets = ["internal", "external_1", "external_2"]

n_iterations = 200

for outcome in outcomes:
    
    all_models = []

    all_results = pd.DataFrame()
    
    best_models = []
    

    for modality in modalities:

        models = os.listdir(f"{DATA_DIR}/{modality}")
        
        best_score = 0

        for model in models:
            
            if model not in exclude:
                
                all_tests = pd.DataFrame()
                
                overall_performance = []
                
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

                        auroc.append(ind_auroc)
                        
                        precision, recall, _ = precision_recall_curve(sample_true, sample_pred)
                        
                        
                        ind_auprc = auc(recall, precision)
                        
                        auprc.append(ind_auprc)

                        # print(f"Iteration {i+1}")

                    metrics_names = [f"AUROC_{test_set}", f"AUPRC_{test_set}"]

                    auroc = pd.DataFrame(data=auroc, columns=[f"AUROC_{test_set}"])
                    auprc = pd.DataFrame(data=auprc, columns=[f"AUPRC_{test_set}"])

                    auroc_ind_test = auroc.median().values[0]
                    auprc_ind_test = auprc.median().values[0]

                    overall_performance.append(auroc_ind_test)
                    overall_performance.append(auprc_ind_test)

                    auroc_sum = auroc.apply(
                        lambda x: f"{x.median()*100:.1f} ({max(0.0, x.quantile(0.025))*100:.1f}-{min(1, x.quantile(0.975))*100:.1f})",
                        axis=0,
                    ).values[0]
                    

                    auprc_sum = auprc.apply(
                        lambda x: f"{x.median()*100:.1f} ({max(0.0, x.quantile(0.025))*100:.1f}-{min(1, x.quantile(0.975))*100:.1f})",
                        axis=0,
                    ).values[0]
                    
                    metrics = pd.DataFrame(data=[[str(auroc_sum), str(auprc_sum)]], columns=metrics_names, index=[f"{model}_{modality}"])

                    # metrics.columns = metrics_names

                    # print(metrics)
                    
                    all_tests = pd.concat([all_tests, metrics], axis=1)
                    
                overall_performance = np.mean(overall_performance)
                
                print(f"{model}: {overall_performance}")
                
                if overall_performance > best_score:
                    
                    if model != "gatortron_s_mlm":
                    
                        best_model = model
                        best_score = overall_performance

                all_tests["model"] = model
                all_tests["modality"] = modality
                
                all_results = pd.concat([all_results, all_tests], axis=0)

                all_models.append(model)
                
                print(f"{model} done")
                
        best_models.append(best_model)
        
    print(best_models)


    # sup = ["a", "b", "c", "d", "e"]
    sup = ["a", "b"]

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
    comparisons = best_models
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
    
    models_to_compare = ["gatortron_s_mlm"]
    
    print(all_results)
    
    count = 0

    for comp in comparisons:

        for model in models_to_compare:

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

    all_results.loc[:,all_aurocs] = all_results.loc[:,all_aurocs].applymap(replace_text)

    all_results["modality"] = pd.Categorical(all_results["modality"], categories=modalities, ordered=True)
    
    all_results = all_results.sort_values(by=["modality", "model"])

    print(all_results)

    with open(f"{DATA_DIR}/{outcome}_results_table2.html", 'w') as f:
        f.write(all_results.to_html())

    all_results.to_csv(f"{DATA_DIR}/{outcome}_results_table2.csv", index=None)
    

# %%



DATA_DIR = "/blue/parisa.rashidi/contreras.miguel/clinical_notes/main3/models"

modalities = ["ehr", "ehr_text"]

main_model = "gatortron_s_mlm"

exclude = ["OLD", "dellirium"]

outcomes = ["delirium_1"]

# test_sets = ["internal", "external_1", "external_2", "total_external"]
test_sets = ["internal", "total_external"]


n_iterations = 200

for outcome in outcomes:
    
    all_models = []

    all_results = pd.DataFrame()
    
    best_models = []
    

    for modality in modalities:

        models = os.listdir(f"{DATA_DIR}/{modality}")
        
        best_score = 0

        for model in models:
            
            if model not in exclude:
                
                all_tests = pd.DataFrame()
                
                overall_performance = []
                
                for test_set in test_sets:
                    
                    if test_set != "total_external":

                        results = pd.read_csv(f"{DATA_DIR}/{modality}/{model}/results/{test_set}_{outcome}_results.csv")
                        
                    else:
                        
                        results_external1 = pd.read_csv(f"{DATA_DIR}/{modality}/{model}/results/external_1_{outcome}_results.csv")
                        results_external2 = pd.read_csv(f"{DATA_DIR}/{modality}/{model}/results/external_2_{outcome}_results.csv")
                        
                        results = pd.concat([results_external1, results_external2], axis=0)


                    model_probs = results["pred"].values
                    model_true = results["true"].values

                    auroc = []

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

                        auroc.append(ind_auroc)                        
                        
                    
                    metrics_names = [f"AUROC_{test_set}"]

                    auroc = pd.DataFrame(data=auroc, columns=[f"AUROC_{test_set}"])

                    auroc_ind_test = auroc.median().values[0]

                    overall_performance.append(auroc_ind_test)
                    
                    auroc_sum = auroc.apply(
                        lambda x: f"{x.median()*100:.1f} ({max(0.0, x.quantile(0.025))*100:.1f}-{min(1, x.quantile(0.975))*100:.1f})",
                        axis=0,
                    ).values[0]

                    # auroc_sum = auroc.apply(
                    #     lambda x: f"{x.median():.2f} ({max(0.0, x.quantile(0.025)):.2f}-{min(1, x.quantile(0.975)):.2f})",
                    #     axis=0,
                    # ).values[0]
                    
                    
                    metrics = pd.DataFrame(data=[[str(auroc_sum)]], columns=metrics_names, index=[f"{model}_{modality}"])

                    # metrics.columns = metrics_names

                    # print(metrics)
                    
                    all_tests = pd.concat([all_tests, metrics], axis=1)
                    
                overall_performance = overall_performance[-1]
                
                print(f"{model}: {overall_performance}")
                
                if overall_performance > best_score:
                    
                    if model != main_model:
                    
                        best_model = model
                        best_score = overall_performance

                all_tests["model"] = model
                all_tests["modality"] = modality
                
                all_results = pd.concat([all_results, all_tests], axis=0)

                all_models.append(model)
                
                print(f"{model} done")
                
        best_models.append(best_model)
        
    print(best_models)


    sup = ["a", "b"]

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

    comparisons = best_models


    test_sets = ["internal", "total_external"]

    metrics = ["AUROC"]
    
    models_to_compare = [main_model]
    
    print(all_results)
    
    count = 0

    for comp in comparisons:

        for model in models_to_compare:

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
                        # all_results.loc[(all_results["model"] == model), f"{metric}_{test_set}"] = (
                        #     all_results.loc[(all_results["model"] == model), f"{metric}_{test_set}"].values[0] + "*"
                        # )

        count += 1


    # Regular expression pattern to match text after closing parenthesis
    pattern = r"\)([^\s])"

    # Function to replace matched text with a space
    def replace_text(value):
        return re.sub(pattern, r") ", value)

    # Apply the replacement operation to each element in the dataframe
    
    all_aurocs = [f"AUROC_{test_set}" for test_set in test_sets]
    
    all_metrics = all_aurocs

    all_results.loc[:,all_aurocs] = all_results.loc[:,all_aurocs].applymap(replace_text)

    all_results["modality"] = pd.Categorical(all_results["modality"], categories=modalities, ordered=True)
    
    model_order = [
        "catboost",
        "nn",
        "transformer",
        "mamba",
        "modernbert",
        "clinicalbert",
        "gatortron_s",
        "gatortron_large",
        "llama3",
        "llama31",
        "llama32",
        main_model
    ]
    
    all_results["model"] = pd.Categorical(all_results["model"], categories=model_order, ordered=True)
    
    all_results = all_results.sort_values(by=["model"])

    print(all_results)

    # with open(f"{DATA_DIR}/{outcome}_results_table2.html", 'w') as f:
    #     f.write(all_results.to_html())

    # all_results.to_csv(f"{DATA_DIR}/{outcome}_results_table2.csv", index=None)

# %%
