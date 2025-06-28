# scpred_py/_analysis_utils.py

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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
        # Re-order columns of y_pred_probs if necessary based on classifier_classes
        y_scores_ordered = y_pred_probs[[f"scpred_prob_{c}" for c in classifier_classes]].values

        # Binarize true labels, ensuring consistent order of classes
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(classifier_classes) # Fit on the full set of classifier classes
        y_true_binary = label_binarizer.transform(true_labels_str)

        for i, class_label in enumerate(classifier_classes):
            # Handle cases where a class might be missing in y_true_binary or y_scores_ordered
            if i < y_true_binary.shape[1] and i < y_scores_ordered.shape[1]:
                # Check if there's at least one positive and one negative sample for ROC AUC
                if len(np.unique(y_true_binary[:, i])) > 1:
                    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores_ordered[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_auc_scores[class_label] = roc_auc
                else:
                    roc_auc_scores[class_label] = np.nan # Or 0.5, or a specific indicator
            else:
                roc_auc_scores[class_label] = np.nan # Class not found in data or probabilities

        print("Per-class ROC AUC scores:")
        for label, score in roc_auc_scores.items():
            print(f"  Class {label}: {score:.4f}")
        metrics_results['roc_auc_scores'] = roc_auc_scores
    else:
        print("Prediction probabilities (y_pred_probs) or classifier classes not provided, cannot compute ROC AUC.")

    return metrics_results


def plot_confusion_matrix(true_labels, predicted_labels, class_labels=None, figsize=(8, 6)):
    """
    Generates and displays a confusion matrix heatmap.

    Args:
        true_labels (pd.Series or np.ndarray): True cell type labels.
        predicted_labels (pd.Series or np.ndarray): Predicted cell type labels.
        class_labels (list, optional): Ordered list of all unique class labels to ensure
                                      consistent axis ordering. If None, inferred from data.
        figsize (tuple): Figure size (width, height) in inches.
    """
    print("\n--- Plotting Confusion Matrix ---")
    # Ensure labels are string type consistently
    true_labels_str = true_labels.astype(str)
    predicted_labels_str = predicted_labels.astype(str)

    if class_labels is None:
        cm_labels = sorted(true_labels_str.unique().tolist())
    else:
        cm_labels = sorted([str(c) for c in class_labels]) # Ensure labels are strings

    cm = confusion_matrix(true_labels_str, predicted_labels_str, labels=cm_labels)
    cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)

    plt.figure(figsize=figsize)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Cells'})
    plt.title('Confusion Matrix (True vs. Predicted Labels)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_roc_curves(true_labels, y_pred_probs, classifier_classes, figsize=(10, 8)):
    """
    Generates and displays One-vs-Rest ROC curves for each class.

    Args:
        true_labels (pd.Series or np.ndarray): True cell type labels.
        y_pred_probs (pd.DataFrame): DataFrame of prediction probabilities.
        classifier_classes (list): List of all unique class labels from the classifier.
        figsize (tuple): Figure size (width, height) in inches.
    """
    print("\n--- Plotting Per-Class ROC AUC (One-vs-Rest) ---")
    true_labels_str = true_labels.astype(str)
    
    # Ensure that y_pred_probs columns match classifier_classes order for consistent indexing
    y_scores_ordered = y_pred_probs[[f"scpred_prob_{c}" for c in classifier_classes]].values

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(classifier_classes)
    y_true_binary = label_binarizer.transform(true_labels_str)

    plt.figure(figsize=figsize)
    for i, class_label in enumerate(classifier_classes):
        if i < y_true_binary.shape[1] and i < y_scores_ordered.shape[1] and len(np.unique(y_true_binary[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores_ordered[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
        else:
            print(f"Skipping ROC plot for Class {class_label}: Not enough unique true labels or probabilities.")

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - One-vs-Rest')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_probability_distributions(query_adata_pred, classifier_classes, figsize=(10, 5)):
    """
    Generates and displays box plots of prediction probabilities for each true class.

    Args:
        query_adata_pred (ad.AnnData): AnnData object with 'scpred_prob_X' columns and 'cell_type'.
        classifier_classes (list): List of all unique class labels from the classifier.
        figsize (tuple): Figure size (width, height) in inches for each plot.
    """
    print("\n--- Plotting Distribution of Prediction Probabilities by True Class ---")
    
    prob_df = query_adata_pred.obs[[f"scpred_prob_{c}" for c in classifier_classes]].copy()
    prob_df['true_cell_type'] = query_adata_pred.obs['cell_type'].astype(str)

    # Melt the DataFrame for easier plotting with seaborn
    prob_melted = prob_df.melt(
        id_vars='true_cell_type',
        value_vars=[f"scpred_prob_{c}" for c in classifier_classes],
        var_name='predicted_class_prob_of',
        value_name='probability'
    )
    prob_melted['predicted_class_prob_of'] = prob_melted['predicted_class_prob_of'].str.replace('scpred_prob_', '')

    # Plotting loop for each true cell type
    for true_type in sorted(prob_melted['true_cell_type'].unique()):
        plt.figure(figsize=figsize)
        subset_df = prob_melted[prob_melted['true_cell_type'] == true_type]
        sns.boxplot(
            data=subset_df,
            x='predicted_class_prob_of',
            y='probability',
            hue='predicted_class_prob_of', # Assign x to hue for deprecation warning fix
            palette='viridis',
            legend=False # Hide redundant legend
        )
        plt.title(f'Prediction Probabilities for True Cell Type: {true_type}')
        plt.xlabel('Probability of being Predicted as Class')
        plt.ylabel('Probability')
        plt.ylim(-0.05, 1.05) # Consistent y-axis limits
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()


def plot_umap_true_vs_predicted(adata, true_label_key='cell_type', predicted_label_key='scpred_prediction', figsize=(12, 6)):
    """
    Generates a UMAP plot comparing true and predicted labels side-by-side.

    Args:
        adata (ad.AnnData): AnnData object with UMAP embedding and true/predicted labels in .obs.
        true_label_key (str): Key in .obs for true labels.
        predicted_label_key (str): Key in .obs for predicted labels.
        figsize (tuple): Figure size (width, height) in inches.
    """
    print("\n--- Plotting UMAP: True vs Predicted Labels ---")
    plt.figure(figsize=figsize) # Use figsize here to control the overall figure size
    
    sc.pl.umap(
        adata,
        color=[true_label_key, predicted_label_key],
        title=[f'True Labels ({true_label_key})', 'scPred Predictions'],
        frameon=False,
        show=False, # Set show=False to control display manually
        ncols=2,
        ax_fontsize=12,
        wspace=0.3 # Adjust horizontal space between subplots
    )
    plt.suptitle('UMAP of Query Data: True vs Predicted Labels', y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


def plot_misclassified_umap(adata, true_label_key='cell_type', predicted_label_key='scpred_prediction', figsize=(8, 6)):
    """
    Generates a UMAP plot highlighting misclassified cells.

    Args:
        adata (ad.AnnData): AnnData object with UMAP embedding and true/predicted labels in .obs.
        true_label_key (str): Key in .obs for true labels.
        predicted_label_key (str): Key in .obs for predicted labels.
        figsize (tuple): Figure size (width, height) in inches.
    """
    print("\n--- Plotting Misclassified Cells UMAP ---")
    
    # Ensure true_labels_str and predicted_labels_str are available (as they are in notebook scope usually)
    true_labels_str = adata.obs[true_label_key].astype(str)
    predicted_labels_str = adata.obs[predicted_label_key].astype(str)

    # Create 'misclassified' column, ensuring categories are strings
    adata.obs['misclassified'] = (true_labels_str != predicted_labels_str).astype(str).astype('category')
    
    # Custom palette: red for misclassified ('True'), light gray for correctly classified ('False')
    misclassified_palette = {'True': 'red', 'False': 'lightgray'}
    
    plt.figure(figsize=figsize)
    sc.pl.umap(
        adata,
        color='misclassified',
        palette=misclassified_palette,
        size=50,
        alpha=0.7,
        title='UMAP of Query Data: Misclassified Cells',
        frameon=False,
        show=False,
        legend_loc='on data' # Show legend on data
    )
    # Adjust legend text for clarity if it's overlapping
    # This might require grabbing the legend object and modifying its labels
    # For now, let's just make sure it displays cleanly
    # You might manually adjust legend position if 'on data' is messy, e.g., 'right margin'
    plt.show()


def plot_prediction_confidence_umap(adata, predicted_label_key='scpred_prediction', classifier_classes=None, figsize=(8, 6)):
    """
    Generates a UMAP plot colored by the confidence (probability) of the predicted class.

    Args:
        adata (ad.AnnData): AnnData object with UMAP embedding and 'scpred_prob_X' columns in .obs.
        predicted_label_key (str): Key in .obs for predicted labels.
        classifier_classes (list): List of all unique class labels from the classifier. Required to find prob columns.
        figsize (tuple): Figure size (width, height) in inches.
    """
    print("\n--- Plotting UMAP by Prediction Confidence ---")
    
    if not hasattr(adata.obs, predicted_label_key) or classifier_classes is None:
        print("Skipping UMAP by prediction confidence: Predicted labels or classifier classes not available.")
        return

    # Get the probability for the *predicted* class for each cell
    def get_predicted_prob(row, classes):
        pred_class = row[predicted_label_key]
        prob_col_name = f'scpred_prob_{pred_class}'
        # Ensure the prob column exists for the predicted class
        if prob_col_name in row.index:
            return row[prob_col_name]
        return np.nan # Return NaN if probability column not found for some reason

    # Apply this function only to the relevant probability columns for efficiency
    prob_cols = [f"scpred_prob_{c}" for c in classifier_classes]
    
    # Ensure all required probability columns exist before applying
    if not all(col in adata.obs.columns for col in prob_cols):
        print("Skipping UMAP by prediction confidence: Required probability columns not found in .obs.")
        return

    # Create a temporary dataframe with just the predicted class and all probabilities
    temp_obs_for_apply = adata.obs[[predicted_label_key] + prob_cols]
    adata.obs['predicted_prob_score'] = temp_obs_for_apply.apply(
        lambda row: get_predicted_prob(row, classifier_classes), axis=1
    )

    plt.figure(figsize=figsize)
    sc.pl.umap(
        adata,
        color='predicted_prob_score',
        cmap='viridis', # Use a sequential colormap for probabilities
        title='UMAP of Query Data: Prediction Confidence',
        frameon=False,
        vmin=0.0, vmax=1.0, # Standardize color scale from 0 to 1
        # colorbar_title='Predicted Probability', # Removed this parameter due to compatibility issues
        show=False
    )
    plt.show()

