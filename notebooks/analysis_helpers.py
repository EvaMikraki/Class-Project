# analysis_helpers.py

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scpred_py_final import _analysis_utils

def plot_results_comprehensive(adata_pred, true_labels, predicted_labels, classifier_classes, y_pred_probs_df, scenario_title, threshold_val, random_state, dataset_name):
    """
    Fully comprehensive and generic plotting function for scPred results.
    """
    print(f"\n--- Visualizing Results ({scenario_title}) ---")

    # UMAP computation
    if 'X_scpred_pca' in adata_pred.obsm:
        print("Computing neighbors and UMAP based on X_scpred_pca...")
        sc.pp.neighbors(adata_pred, n_neighbors=10, use_rep='X_scpred_pca', random_state=random_state)
        sc.tl.umap(adata_pred, random_state=random_state)
    else:
        print("X_scpred_pca not found. Skipping UMAP plots.")
        return

    # Plot 1: Confusion Matrix
    print(f"\n--- Confusion Matrix ({scenario_title}) ---")
    all_labels_for_cm = sorted(list(set(true_labels.astype(str)) | set(predicted_labels.astype(str))))
    cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels_for_cm)
    cm_df = pd.DataFrame(cm, index=all_labels_for_cm, columns=all_labels_for_cm)
    plt.figure(figsize=(12, 10)); sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({dataset_name} - {scenario_title})'); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.show()
    
    # ... (all other plotting code for ROC curves, boxplots, and UMAPs goes here, exactly as in the last script) ...
    # ... ensure every `plt.title()` uses the `dataset_name` variable, e.g.:
    # plt.title(f'ROC Curve ({dataset_name} - {scenario_title})')

def run_scenario(scpred_model, query_adata_raw, threshold, scenario_name, random_state, dataset_name):
    """
    Generic function to run a full analysis scenario.
    """
    print(f"\n\n======== {scenario_name} ========")
    query_adata_pred = scpred_model.predict(query_adata_raw.copy(), threshold=threshold)
    
    print("\nQuery Data with Predictions (first 5 rows):")
    print(query_adata_pred.obs[['cell_type', 'scpred_prediction']].head())
    print("\nPredicted label distribution:\n", query_adata_pred.obs['scpred_prediction'].value_counts(dropna=False))

    prob_cols = [f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_]
    query_adata_pred.obs['max_prob_value'] = query_adata_pred.obs[prob_cols].max(axis=1)

    valid_cells_mask = query_adata_pred.obs['scpred_prediction'].notna()
    true_labels = query_adata_pred.obs['cell_type'][valid_cells_mask]
    predicted_labels = query_adata_pred.obs['scpred_prediction'][valid_cells_mask]
    y_pred_probs_df = query_adata_pred.obs.loc[valid_cells_mask, prob_cols].copy()

    print(f"\n--- Evaluating Predictions ({scenario_name}) ---")
    _analysis_utils.evaluate_and_report_metrics(
        true_labels=true_labels, predicted_labels=predicted_labels,
        classifier_classes=scpred_model.classifier_.classes_, y_pred_probs=y_pred_probs_df
    )

    plot_results_comprehensive(
        adata_pred=query_adata_pred[valid_cells_mask].copy(), true_labels=true_labels,
        predicted_labels=predicted_labels, classifier_classes=scpred_model.classifier_.classes_,
        y_pred_probs_df=y_pred_probs_df, scenario_title=scenario_name,
        threshold_val=threshold, random_state=random_state, dataset_name=dataset_name
    )