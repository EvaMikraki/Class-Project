# scpred_py/_analysis_utils.py

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

    # Calculate overall Balanced Accuracy
    overall_balanced_accuracy = balanced_accuracy_score(true_labels_str, predicted_labels_str)
    print(f"Overall Balanced Accuracy: {overall_balanced_accuracy:.4f}")

    # Calculate Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(true_labels_str, predicted_labels_str)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    print("\n--- Full Classification Report ---")
    # Determine all unique labels for the report, ensuring sorted order
    # Use classifier_classes if provided, otherwise infer from true/predicted labels
    if classifier_classes is not None:
        all_labels_for_report = sorted([str(c) for c in classifier_classes])
    else:
        all_labels_for_report = sorted(list(set(true_labels_str.unique()) | set(predicted_labels_str.unique())))

    print(classification_report(true_labels_str, predicted_labels_str,
                                labels=all_labels_for_report, zero_division=0))

    metrics_results = {
        'balanced_accuracy': overall_balanced_accuracy,
        'mcc': mcc,
        # classification_report returns a string, parse it if specific metrics are needed as numbers
        # For simplicity, we'll just return overall metrics for now.
    }

    print("\n--- Per-Class ROC AUC (One-vs-Rest) ---")
    if y_pred_probs is not None and classifier_classes is not None:
        roc_auc_scores = {}
        # Ensure that y_pred_probs columns match classifier_classes order for consistent indexing
        # and handle potential type mismatches if class names are numeric (e.g., '0' vs 0)
        prob_col_names = [f"scpred_prob_{c}" for c in classifier_classes]
        
        # Filter to only columns that actually exist in y_pred_probs
        existing_prob_cols = [col for col in prob_col_names if col in y_pred_probs.columns]
        
        # Map actual class labels to column names used in y_pred_probs for robust lookup
        class_to_prob_col = {str(c): f"scpred_prob_{c}" for c in classifier_classes}

        y_scores_ordered = []
        target_classes_for_roc = []

        for class_label in classifier_classes:
            prob_col_name = class_to_prob_col[str(class_label)]
            if prob_col_name in y_pred_probs.columns:
                y_scores_ordered.append(y_pred_probs[prob_col_name])
                target_classes_for_roc.append(str(class_label))
            else:
                print(f"Warning: Probability column '{prob_col_name}' not found for class '{class_label}'. Skipping ROC for this class.")
        
        if len(y_scores_ordered) > 0:
            y_scores_ordered = np.array(y_scores_ordered).T # Transpose to (n_samples, n_classes)

            # Binarize true labels, ensuring consistent order of classes for ROC
            label_binarizer = LabelBinarizer()
            # Fit on the actual target_classes_for_roc, as these are the ones we have scores for
            label_binarizer.fit(target_classes_for_roc) 
            y_true_binary = label_binarizer.transform(true_labels_str)

            for i, class_label_str in enumerate(target_classes_for_roc):
                # Check if there's at least one positive and one negative sample for ROC AUC
                if y_true_binary.shape[1] > i and len(np.unique(y_true_binary[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores_ordered[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_auc_scores[class_label_str] = roc_auc
                else:
                    roc_auc_scores[class_label_str] = np.nan # Class not found or only one label in binary target
            
            print("Per-class ROC AUC scores:")
            for label, score in roc_auc_scores.items():
                print(f"  Class {label}: {score:.4f}")
            metrics_results['roc_auc_scores'] = roc_auc_scores
        else:
            print("No valid probability columns found to compute ROC AUC.")
    else:
        print("Prediction probabilities (y_pred_probs) or classifier classes not provided, cannot compute ROC AUC.")

    return metrics_results

