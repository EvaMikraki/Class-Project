import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelBinarizer


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

    # --- Robust Filtering for NaN values ---
    # Identify cells that have valid predictions (not NaN)
    # This handles cells that might have been filtered out during preprocessing steps
    # and thus have NaN in their prediction columns after reindexing.
    valid_prediction_mask = predicted_labels_str.notna()

    # Apply this mask to both true and predicted labels for all subsequent calculations
    filtered_true_labels_all = true_labels_str[valid_prediction_mask].copy()
    filtered_predicted_labels_all = predicted_labels_str[valid_prediction_mask].copy()

    # Filter out 'unassigned' cells for Balanced Accuracy, MCC, and Classification Report
    # This is done *after* filtering out NaNs.
    assigned_mask = (filtered_predicted_labels_all != 'unassigned')

    filtered_true_labels_assigned = filtered_true_labels_all[assigned_mask].copy()
    filtered_predicted_labels_assigned = filtered_predicted_labels_all[assigned_mask].copy()

    # Calculate overall Balanced Accuracy (only on assigned cells)
    overall_balanced_accuracy = np.nan
    if len(filtered_true_labels_assigned) > 0:
        overall_balanced_accuracy = balanced_accuracy_score(filtered_true_labels_assigned, filtered_predicted_labels_assigned)
    print(f"Overall Balanced Accuracy (excluding 'unassigned'): {overall_balanced_accuracy:.4f}")

    # Calculate Matthews Correlation Coefficient (MCC) (only on assigned cells)
    mcc = np.nan
    if len(filtered_true_labels_assigned) > 0:
        mcc = matthews_corrcoef(filtered_true_labels_assigned, filtered_predicted_labels_assigned)
    print(f"Matthews Correlation Coefficient (MCC) (excluding 'unassigned'): {mcc:.4f}")

    print("\n--- Full Classification Report ---")
    # Determine all unique labels for the report, ensuring sorted order
    if classifier_classes is not None:
        # Ensure 'unassigned' is included in the labels for the report if it's a possible prediction
        all_labels_for_report = sorted([str(c) for c in classifier_classes] + ['unassigned'])
    else:
        all_labels_for_report = sorted(list(set(filtered_true_labels_all.unique()) | set(filtered_predicted_labels_all.unique())))

    # Use the full set of valid predicted labels (including 'unassigned') for the report
    print(classification_report(filtered_true_labels_all, filtered_predicted_labels_all,
                                labels=all_labels_for_report, zero_division=0))

    metrics_results = {
        'balanced_accuracy': overall_balanced_accuracy,
        'mcc': mcc,
    }

    print("\n--- Per-Class ROC AUC (One-vs-Rest) ---")
    if y_pred_probs is not None and classifier_classes is not None:
        roc_auc_scores = {}
        
        # Filter y_pred_probs to only include rows that correspond to valid predictions (not NaN)
        # This is crucial for ROC AUC calculation.
        y_pred_probs_filtered = y_pred_probs.loc[valid_prediction_mask].copy()

        # Ensure that y_pred_probs columns match classifier_classes order for consistent indexing
        class_to_prob_col = {str(c): f"scpred_prob_{c}" for c in classifier_classes}

        y_scores_ordered = []
        target_classes_for_roc = []

        for class_label in classifier_classes:
            class_label_str = str(class_label)
            prob_col_name = class_to_prob_col[class_label_str]
            
            if prob_col_name in y_pred_probs_filtered.columns:
                # Check if the probability column itself contains NaNs after filtering
                # This should ideally not happen if valid_prediction_mask works correctly, but as a safeguard
                if y_pred_probs_filtered[prob_col_name].isna().any():
                    print(f"Skipping ROC plot for Class {class_label_str}: NaN probabilities detected after filtering. All values are NaN.")
                    roc_auc_scores[class_label_str] = np.nan
                    continue # Skip to next class

                y_scores_ordered.append(y_pred_probs_filtered[prob_col_name])
                target_classes_for_roc.append(class_label_str)
            else:
                print(f"Warning: Probability column '{prob_col_name}' not found for class '{class_label_str}'. Skipping ROC for this class.")
                roc_auc_scores[class_label_str] = np.nan
        
        if len(y_scores_ordered) > 0:
            y_scores_ordered_array = np.array(y_scores_ordered).T # Transpose to (n_samples, n_classes)

            # Binarize true labels, ensuring consistent order of classes for ROC
            label_binarizer = LabelBinarizer()
            # Fit on the actual target_classes_for_roc, as these are the ones we have scores for
            label_binarizer.fit(target_classes_for_roc) 
            
            # Use the filtered true labels that correspond to the valid probabilities
            y_true_binary = label_binarizer.transform(filtered_true_labels_all)

            for i, class_label_str in enumerate(target_classes_for_roc):
                # Check if there's at least one positive and one negative sample for ROC AUC
                # within the filtered set of labels for the current class
                if y_true_binary.shape[1] > i and len(np.unique(y_true_binary[:, i])) > 1:
                    # Ensure y_score for roc_curve is a 1D array
                    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores_ordered_array[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_auc_scores[class_label_str] = roc_auc
                else:
                    print(f"Skipping ROC plot for Class {class_label_str}: Not enough unique true labels after filtering for ROC.")
                    roc_auc_scores[class_label_str] = np.nan # Class not found or only one label in binary target
            
            print("Per-class ROC AUC scores:")
            for label, score in roc_auc_scores.items():
                print(f"  Class {label}: {score:.4f}")
            metrics_results['roc_auc_scores'] = roc_auc_scores
        else:
            print("No valid probability columns found to compute ROC AUC after filtering.")
    else:
        print("Prediction probabilities (y_pred_probs) or classifier classes not provided, cannot compute ROC AUC.")

    return metrics_results
