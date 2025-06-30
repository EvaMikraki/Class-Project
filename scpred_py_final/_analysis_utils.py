import numpy as np
import pandas as pd
import scanpy as sc # Still needed for AnnData context within comments/docstrings if any remain
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelBinarizer # Useful for consistent binary labels for ROC AUC


def evaluate_and_report_metrics(true_labels, predicted_labels, classifier_classes=None, y_pred_probs=None):
    """
    Calculates and prints standard and advanced classification metrics.

    Args:
        true_labels (pd.Series or np.ndarray): True cell type labels.
        predicted_labels (pd.Series or np.ndarray): Predicted cell type labels.
        classifier_classes (list, optional): List of all unique class labels from the classifier.
                                             Required for consistent ROC AUC and classification report.
        y_pred_probs (pd.DataFrame, optional): DataFrame of prediction probabilities, where columns
                                               are class labels (e.g., 'scpred_prob_0', 'scpred_prob_1').
                                               Required for ROC AUC calculation.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    print("\n--- Evaluating Predictions ---")

    # Ensure labels are string type for consistency in scikit-learn metrics
    true_labels_str = true_labels.astype(str)
    predicted_labels_str = predicted_labels.astype(str)

    # Mask for cells that are NOT "unassigned"
    non_unassigned_mask = (predicted_labels_str != "unassigned")
    true_labels_filtered = true_labels_str[non_unassigned_mask]
    predicted_labels_filtered = predicted_labels_str[non_unassigned_mask]

    # Calculate overall Balanced Accuracy
    if len(true_labels_filtered) > 0:
        overall_balanced_accuracy = balanced_accuracy_score(true_labels_filtered, predicted_labels_filtered)
        print(f"Overall Balanced Accuracy (excluding 'unassigned'): {overall_balanced_accuracy:.4f}")
    else:
        overall_balanced_accuracy = np.nan
        print("No assigned predictions to calculate Balanced Accuracy.")


    # Calculate Matthews Correlation Coefficient (MCC)
    if len(true_labels_filtered) > 0:
        mcc = matthews_corrcoef(true_labels_filtered, predicted_labels_filtered)
        print(f"Matthews Correlation Coefficient (MCC) (excluding 'unassigned'): {mcc:.4f}")
    else:
        mcc = np.nan
        print("No assigned predictions to calculate MCC.")

    print("\n--- Full Classification Report ---")
    # Determine all unique labels for the report, ensuring sorted order
    if classifier_classes is not None:
        all_labels_for_report = sorted([str(c) for c in classifier_classes] + (['unassigned'] if 'unassigned' in predicted_labels_str.unique() else []))
    else:
        all_labels_for_report = sorted(list(set(true_labels_str.unique()) | set(predicted_labels_str.unique())))

    print(classification_report(true_labels_str, predicted_labels_str,
                                 labels=all_labels_for_report, zero_division=0))

    metrics_results = {
        'balanced_accuracy': overall_balanced_accuracy,
        'mcc': mcc,
    }

    print("\n--- Per-Class ROC AUC (One-vs-Rest) ---")
    if y_pred_probs is not None and classifier_classes is not None:
        roc_auc_scores = {}
        
        # FIX: Filter y_pred_probs using the same mask as for true_labels
        # This ensures y_scores_filtered and y_true_binary have consistent lengths
        y_scores_filtered = y_pred_probs.loc[non_unassigned_mask, :].copy()

        # Convert true_labels_filtered to binary for ROC calculation
        label_binarizer = LabelBinarizer()
        # Fit on classifier_classes as these are the "true" classes the classifier knows about
        label_binarizer.fit([str(c) for c in classifier_classes]) 
        y_true_binary = label_binarizer.transform(true_labels_filtered)

        # Check if y_true_binary is empty or has only one class after filtering
        if y_true_binary.size == 0 or np.all(y_true_binary == y_true_binary[0, :]):
            print("Not enough assigned cells with diverse true labels to compute ROC AUC for any class.")
            metrics_results['roc_auc_scores'] = {} # No ROC AUC scores
            return metrics_results

        for i, class_label in enumerate(classifier_classes):
            class_label_str = str(class_label)
            prob_col_name = f"scpred_prob_{class_label_str}"

            if prob_col_name in y_scores_filtered.columns:
                # Check if there's at least one positive and one negative sample for ROC AUC
                # within the filtered set of labels for the current class
                if y_true_binary.shape[1] > i and len(np.unique(y_true_binary[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores_filtered[prob_col_name])
                    roc_auc = auc(fpr, tpr)
                    roc_auc_scores[class_label_str] = roc_auc
                else:
                    roc_auc_scores[class_label_str] = np.nan # Class not found or only one label in binary target
            else:
                print(f"Warning: Probability column '{prob_col_name}' not found in provided y_pred_probs. Skipping ROC for class '{class_label_str}'.")
        
        if len(roc_auc_scores) > 0 and not all(pd.isna(list(roc_auc_scores.values()))):
            print("Per-class ROC AUC scores:")
            for label, score in roc_auc_scores.items():
                print(f"  Class {label}: {score:.4f}")
            metrics_results['roc_auc_scores'] = roc_auc_scores
        else:
            print("No valid probability columns or sufficient data to compute ROC AUC for any class.")
    else:
        print("Prediction probabilities (y_pred_probs) or classifier classes not provided, cannot compute ROC AUC.")

    return metrics_results
