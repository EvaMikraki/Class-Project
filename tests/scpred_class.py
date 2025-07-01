import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import mannwhitneyu

class scPred:
    """
    A Python implementation of the scPred algorithm for single-cell classification.

    This class follows the methodology described in:
    Alquicira-Hernandez, J., et al. (2019). scPred: accurate supervised method for cell-type
    classification from single-cell RNA-seq data. Genome Biology, 20(264).

    The workflow includes:
    1.  Data preprocessing (CPM and log2 transformation).
    2.  Scaling and dimensionality reduction using Principal Component Analysis (PCA) on a training dataset.
    3.  Selection of informative principal components (PCs) based on the Wilcoxon rank-sum test.
    4.  Training a one-vs-rest Support Vector Machine (SVM) classifier on the informative PCs.
    5.  Projection of new data onto the trained PC space and prediction of cell types.
    """

    def __init__(self, probability_threshold=0.9):
        """
        Initializes the scPred model.

        Args:
            probability_threshold (float): The minimum probability required to assign a cell
                                           to a class. Cells with a max probability below
                                           this threshold will be labeled 'Unassigned'.
        """
        self.scaler = None
        self.pca = None
        self.classifier = None
        self.informative_pcs = None
        self.class_labels = None
        self.probability_threshold = probability_threshold

    def _cpm_transform(self, data):
        """Transforms count data to Counts Per Million (CPM)."""
        # Ensure data is float for division
        data = data.astype(float)
        total_counts = np.sum(data, axis=1, keepdims=True)
        # Avoid division by zero for cells with no counts
        total_counts[total_counts == 0] = 1
        return (data / total_counts) * 1_000_000

    def _log2_transform(self, data):
        """Applies log2(x + 1) transformation."""
        return np.log2(data + 1)

    def train(self, X_train, y_train, p_value_threshold=0.05, variance_threshold=0.0001, perform_hpt=False):
        """
        Trains the scPred model on a reference dataset.

        Args:
            X_train (np.ndarray): The training gene expression matrix (cells x genes).
                                  Assumed to be raw counts.
            y_train (np.ndarray or list): The cell type labels for the training data.
            p_value_threshold (float): The p-value cutoff for selecting informative PCs.
            variance_threshold (float): The minimum variance a PC must explain to be considered.
            perform_hpt (bool): If True, performs hyperparameter tuning for the SVM.
                                This can be time-consuming.
        """
        print("Starting training process...")

        # --- 1. Preprocessing ---
        print("Step 1: Preprocessing data (CPM, Log2)...")
        X_cpm = self._cpm_transform(X_train)
        X_log = self._log2_transform(X_cpm)
        
        # --- 2. Scaling and PCA ---
        print("Step 2: Scaling data and performing PCA...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_log)

        # The paper uses SVD, but PCA from scikit-learn is a convenient implementation
        # that centers the data and provides the principal components (scores).
        self.pca = PCA()
        pc_scores = self.pca.fit_transform(X_scaled)

        # --- 3. Feature Selection (Informative PCs) ---
        print("Step 3: Selecting informative PCs...")
        
        # Filter PCs by explained variance
        explained_variance = self.pca.explained_variance_ratio_
        significant_variance_indices = np.where(explained_variance > variance_threshold)[0]
        pc_scores_filtered = pc_scores[:, significant_variance_indices]
        
        self.class_labels = np.unique(y_train)
        informative_pc_union = set()

        # Perform Wilcoxon rank-sum test for each class in a one-vs-rest manner
        for cell_type in self.class_labels:
            print(f"  - Finding informative PCs for class: {cell_type}")
            in_class_mask = (y_train == cell_type)
            
            p_values = []
            for i in range(pc_scores_filtered.shape[1]):
                pc_col = pc_scores_filtered[:, i]
                group1 = pc_col[in_class_mask]
                group2 = pc_col[~in_class_mask]
                
                # The Wilcoxon test requires non-identical samples
                if len(np.unique(group1)) > 1 and len(np.unique(group2)) > 1:
                    stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    p_values.append(p_val)
                else:
                    p_values.append(1.0) # Assign a non-significant p-value

            # Note: The paper mentions Benjamini-Hochberg correction. For simplicity,
            # this is omitted here but could be added using statsmodels.
            significant_pcs_for_class = np.where(np.array(p_values) < p_value_threshold)[0]
            informative_pc_union.update(significant_variance_indices[significant_pcs_for_class])

        self.informative_pcs = sorted(list(informative_pc_union))
        
        if not self.informative_pcs:
            raise ValueError("No informative PCs found. Try relaxing the p-value or variance thresholds.")
            
        print(f"Found {len(self.informative_pcs)} unique informative PCs across all classes.")
        
        X_train_final = pc_scores[:, self.informative_pcs]
        
        # --- 4. Model Training ---
        print("Step 4: Training the SVM classifier...")
        
        # Using a one-vs-rest strategy with an SVM, as described in the paper and proposal
        svm = SVC(probability=True, kernel='rbf')
        
        if perform_hpt:
            print("  - Performing hyperparameter tuning (GridSearchCV)...")
            # This is a basic grid. For real data, a wider search may be needed.
            param_grid = {'estimator__C': [0.1, 1, 10], 'estimator__gamma': ['scale', 'auto']}
            self.classifier = GridSearchCV(OneVsRestClassifier(svm), param_grid, cv=3)
        else:
            # Use default parameters if not tuning
            self.classifier = OneVsRestClassifier(SVC(probability=True, kernel='rbf'))
            
        self.classifier.fit(X_train_final, y_train)
        
        print("Training complete.")

    def predict(self, X_test):
        """
        Predicts cell types for a new dataset.

        Args:
            X_test (np.ndarray): The query gene expression matrix (cells x genes).
                                 Assumed to be raw counts.

        Returns:
            pd.DataFrame: A DataFrame with predicted labels, max probabilities, and assigned class.
        """
        if self.scaler is None or self.pca is None or self.classifier is None:
            raise RuntimeError("The model has not been trained yet. Call train() first.")

        print("Starting prediction process...")
        
        # --- 1. Preprocessing (using stored training parameters) ---
        print("Step 1: Preprocessing query data...")
        X_cpm = self._cpm_transform(X_test)
        X_log = self._log2_transform(X_cpm)
        
        # --- 2. Scaling and Projection ---
        print("Step 2: Scaling and projecting data onto trained PC space...")
        X_scaled = self.scaler.transform(X_log)
        pc_scores_test = self.pca.transform(X_scaled)
        
        # --- 3. Prediction ---
        print("Step 3: Predicting class probabilities...")
        X_test_final = pc_scores_test[:, self.informative_pcs]
        probabilities = self.classifier.predict_proba(X_test_final)
        
        # --- 4. Assigning Labels ---
        print("Step 4: Assigning final labels...")
        max_probs = np.max(probabilities, axis=1)
        pred_indices = np.argmax(probabilities, axis=1)
        
        # Map indices to class labels
        predicted_labels = self.class_labels[pred_indices]
        
        # Apply rejection threshold
        final_labels = np.where(max_probs >= self.probability_threshold, predicted_labels, "Unassigned")
        
        # Create a results dataframe
        results_df = pd.DataFrame({
            'Predicted_Label': final_labels,
            'Max_Probability': max_probs,
            'Assigned_Class': predicted_labels # The class with the highest probability before thresholding
        })
        
        # Add probability columns for each class
        for i, label in enumerate(self.class_labels):
            results_df[f'Prob_{label}'] = probabilities[:, i]
            
        print("Prediction complete.")
        return results_df


if __name__ == '__main__':
    # --- Example Usage with Dummy Data ---
    # This example demonstrates the workflow. Replace this with your actual data loading.
    
    print("--- scPred Python Implementation Demo ---")
    
    # 1. Generate dummy training and testing data
    # In a real scenario, you would load your AnnData or CSV files here.
    n_train_cells, n_test_cells, n_genes = 500, 100, 1000
    
    # Training data
    X_train_dummy = np.random.randint(0, 100, size=(n_train_cells, n_genes))
    y_train_dummy = np.random.choice(['T-cell', 'B-cell', 'Macrophage'], size=n_train_cells)
    
    # Testing data
    X_test_dummy = np.random.randint(0, 100, size=(n_test_cells, n_genes))
    
    print(f"\nGenerated dummy data: {n_train_cells} training cells, {n_test_cells} test cells, {n_genes} genes.")
    print(f"Training classes: {np.unique(y_train_dummy)}")
    
    # 2. Initialize and train the model
    # Set perform_hpt=True to run hyperparameter tuning (slower).
    model = scPred()
    model.train(X_train_dummy, y_train_dummy, perform_hpt=False)
    
    # 3. Predict on new data
    predictions = model.predict(X_test_dummy)
    
    # 4. Display results
    print("\n--- Prediction Results ---")
    print(predictions.head())
    
    print("\nDistribution of predicted labels:")
    print(predictions['Predicted_Label'].value_counts())
    
    print("\nDemo finished.")










# --- 0. Imports and Setup ---
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import pickle # For saving/loading models
import matplotlib.pyplot as plt
import seaborn as sns
import random # For setting Python's random seed

# Make sure you have these installed: pip install scikit-learn scikit-misc
from sklearn.model_selection import train_test_split
from sklearn.metrics import ( # Re-import these for direct use in notebook for plots/confusion matrix
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Adjust the path to import your scpred_py_simple package
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Importing the main ScPredModel class and utilities
# Ensure this import matches your package name (e.g., scpred_py_simple or scpred_py_workhorse_v5)
from scpred_py_revised import ScPredModel # Adjusted to scpred_py_simple as per context
from scpred_py_revised import _analysis_utils
from scpred_py_revised import _preprocessing # Explicitly import for manual preprocessing steps


print("--- Setting up environment for reproducibility and file management ---")
# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE) # For Python's built-in random module
# Set Scanpy's random state for operations that support it
sc.settings.set_figure_params(dpi=80, facecolor='white', figsize=(8, 8)) # Set initial figure params

# Define paths for saving
MODELS_DIR = '../models'
PREPROCESSED_DATA_DIR = '../data/preprocessed'

# Create directories if they don't exist
print(f"Ensuring directories exist: {MODELS_DIR} and {PREPROCESSED_DATA_DIR}")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
print("Directories ensured.")


# --- 1. Data Loading and Initial Annotation (for Paul15 dataset) ---
print("\n--- Step 1: Loading and Annotating Data (Paul15) ---")
# Load the Paul15 dataset from Scanpy
# This dataset is typically already normalized and log-transformed by Scanpy's datasets,
# but it usually does NOT have the 'log1p' flag in adata.uns, which our safety net checks.
adata_raw_paul15 = sc.datasets.paul15()
adata_raw_paul15.var_names_make_unique() # Ensure gene names are unique

print(f"Initial raw AnnData object shape (Paul15): {adata_raw_paul15.shape}")
print(f"Initial raw AnnData .obs keys: {adata_raw_paul15.obs.keys().tolist()}")
print(f"Initial raw AnnData .var keys: {adata_raw_paul15.var.keys().tolist()}\n")
print(f"Initial raw AnnData .uns keys: {list(adata_raw_paul15.uns.keys())}\n") # Check uns for log1p flag

# For the Paul15 dataset, the cell type labels are already available in .obs['paul15_clusters']
# We will use this directly as our 'cell_type' key for scPred.
# Convert to string category for consistent handling by metrics and plotting functions
adata_raw_paul15.obs['cell_type'] = adata_raw_paul15.obs['paul15_clusters'].astype(str).astype('category')

print(f"Full dataset (raw counts + Paul15 clusters) shape: {adata_raw_paul15.shape}")
print(f"Full dataset .obs keys after labeling: {adata_raw_paul15.obs.keys().tolist()}")
print("Cell type distribution (all data):\n", adata_raw_paul15.obs['cell_type'].value_counts())


# --- 2. Train-Test Split (Stratified) ---
# We split the raw data (with labels) into reference and query sets.
print("\n--- Step 2: Splitting Data (Stratified) ---")
indices = range(adata_raw_paul15.n_obs)

ref_idx, query_idx = train_test_split(
    indices,
    test_size=0.3,
    random_state=RANDOM_STATE, # Ensuring split is reproducible
    stratify=adata_raw_paul15.obs['cell_type'] # CRITICAL for balanced classes
)

# These AnnData objects contain raw counts and cell_type labels.
# The ScPredModel will handle their internal preprocessing.
ref_adata_raw_split = adata_raw_paul15[ref_idx, :].copy()
query_adata_raw_split = adata_raw_paul15[query_idx, :].copy()

print(f"Reference data (raw + labels) shape: {ref_adata_raw_split.shape}")
print(f"Query data (raw + labels) shape: {query_adata_raw_split.shape}")
print("\nReference cell type distribution:\n", ref_adata_raw_split.obs['cell_type'].value_counts())
print("\nQuery cell type distribution:\n", query_adata_raw_split.obs['cell_type'].value_counts())

# Calculate and report class imbalance
print("\n--- Class Imbalance Analysis ---")
print("Reference data class imbalance (proportion):\n", ref_adata_raw_split.obs['cell_type'].value_counts(normalize=True).sort_index())
print("Query data class imbalance (proportion):\n", query_adata_raw_split.obs['cell_type'].value_counts(normalize=True).sort_index())


# ==============================================================================
# DEMONSTRATION 1: scPredModel with perform_preprocessing=True (Default)
# ==============================================================================
print("\n======== DEMONSTRATION 1: scPredModel (perform_preprocessing=True) ========")

# --- 3a. Training the scPred Model (Full Preprocessing) ---
print("\n--- Step 3a: Training scPred Model (perform_preprocessing=True) ---")
scpred_model_full = ScPredModel()

scpred_model_full.train(
    ref_adata=ref_adata_raw_split, # Pass the raw reference data
    cell_type_key='cell_type',
    n_components=30,
    hvg_n_top_genes=2000,
    hvg_flavor='seurat',
    svm_kernel='rbf',
    svm_c=1.0,
    svm_random_state=RANDOM_STATE, # Ensuring SVM training is reproducible
)

print("\nModel Trained (Full Preprocessing)!")
print(f"Scaler: {scpred_model_full.scaler_}")
print(f"PCA Model: {scpred_model_full.pca_model_}")
print(f"Classifier: {scpred_model_full.classifier_}")
print(f"Reference HVGs learned: {len(scpred_model_full.reference_hvg_genes_)} genes")

# Save the trained ScPredModel object using its new method
model_full_path = os.path.join(MODELS_DIR, 'scpred_model_paul15_full_preprocessing.pkl')
scpred_model_full.save(model_full_path)

# Save the preprocessed reference data for scenario 1
if hasattr(scpred_model_full, 'ref_adata_processed_') and scpred_model_full.ref_adata_processed_ is not None:
    preprocessed_ref_full_path = os.path.join(PREPROCESSED_DATA_DIR, 'paul15_ref_preprocessed_full.h5ad')
    scpred_model_full.ref_adata_processed_.write(preprocessed_ref_full_path)
    print(f"Preprocessed reference data (Full Preprocessing) saved to: {preprocessed_ref_full_path}")
else:
    print("Warning: scpred_model_full.ref_adata_processed_ is not accessible. Preprocessed reference data not saved for Demo 1.")


# --- 4a. Predicting with the scPred Model (Full Preprocessing) ---
print("\n--- Step 4a: Predicting on Query Data (perform_preprocessing=True) ---")
query_adata_pred_full = scpred_model_full.predict(query_adata_raw_split) # Simplified call now, perform_preprocessing=True is default.

print("\nQuery Data with Predictions (Full Preprocessing, first 5 rows):")
print(query_adata_pred_full.obs[['cell_type', 'scpred_prediction']].head())

# Save the preprocessed query data with predictions for scenario 1
preprocessed_query_full_path = os.path.join(PREPROCESSED_DATA_DIR, 'paul15_query_pred_preprocessed_full.h5ad')
query_adata_pred_full.write(preprocessed_query_full_path)
print(f"Preprocessed query data with predictions (Full Preprocessing) saved to: {preprocessed_query_full_path}")


# --- 5a. Evaluating Predictions (Full Preprocessing) ---
true_labels_full = query_adata_pred_full.obs['cell_type']
predicted_labels_full = query_adata_pred_full.obs['scpred_prediction']
pred_prob_cols_full = [col for col in query_adata_pred_full.obs.columns if col.startswith('scpred_prob_')]
y_pred_probs_df_full = None
if len(pred_prob_cols_full) > 0:
    y_pred_probs_df_full = query_adata_pred_full.obs[pred_prob_cols_full].copy()

print("\n--- Evaluation for Full Preprocessing Model ---")
metrics_results_full = _analysis_utils.evaluate_and_report_metrics(
    true_labels=true_labels_full,
    predicted_labels=predicted_labels_full,
    classifier_classes=scpred_model_full.classifier_.classes_,
    y_pred_probs=y_pred_probs_df_full
)


# --- 6a. Visualizing Results (Full Preprocessing) ---
print("\n--- Step 6a: Visualizing Results (Full Preprocessing) ---")

if 'X_scpred_pca' in query_adata_pred_full.obsm and 'X_umap' not in query_adata_pred_full.obsm:
    print("Computing neighbors and UMAP based on X_scpred_pca for visualization...")
    sc.pp.neighbors(query_adata_pred_full, n_neighbors=10, use_rep='X_scpred_pca', random_state=RANDOM_STATE) # Added random_state
    sc.tl.umap(query_adata_pred_full, random_state=RANDOM_STATE) # Added random_state
    print(f"Query data .obsm keys after UMAP: {list(query_adata_pred_full.obsm.keys())}")
elif 'X_umap' not in query_adata_pred_full.obsm:
    print("X_scpred_pca or X_umap not found in .obsm. UMAP plots might not be generated.")

# Plot 1: Confusion Matrix
print("\n--- Confusion Matrix (Full Preprocessing) ---")
true_labels_str_full = true_labels_full.astype(str)
predicted_labels_str_full = predicted_labels_full.astype(str)
cm_labels_str_full = sorted(true_labels_str_full.unique().tolist())
cm_full = confusion_matrix(true_labels_str_full, predicted_labels_str_full, labels=cm_labels_str_full)
cm_df_full = pd.DataFrame(cm_full, index=cm_labels_str_full, columns=cm_labels_str_full)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df_full, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Cells'})
plt.title('Confusion Matrix (Paul15 - Full Preprocessing)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot 2: Per-Class ROC AUC (Full Preprocessing)
print("\n--- Per-Class ROC AUC (Paul15 - Full Preprocessing) ---")
if hasattr(scpred_model_full.classifier_, 'predict_proba'):
    classes_full = scpred_model_full.classifier_.classes_
    y_true_binary_full = pd.get_dummies(true_labels_str_full, columns=classes_full, drop_first=False).values
    y_scores_full = y_pred_probs_df_full[[f"scpred_prob_{c}" for c in classes_full]].values

    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(classes_full):
        if y_true_binary_full.shape[1] > i and len(np.unique(y_true_binary_full[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary_full[:, i], y_scores_full[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
        else:
            print(f"Skipping ROC plot for Class {class_label}: Not enough unique true labels.")
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Paul15 - Full Preprocessing)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Plot 3: Distribution of Prediction Probabilities (Full Preprocessing)
print("\n--- Distribution of Prediction Probabilities (Paul15 - Full Preprocessing) ---")
if hasattr(scpred_model_full.classifier_, 'predict_proba'):
    prob_df_full = query_adata_pred_full_filtered_for_umap.obs[[f"scpred_prob_{c}" for c in scpred_model_full.classifier_.classes_]].copy()
    prob_df_full['true_cell_type'] = query_adata_pred_full_filtered_for_umap.obs['cell_type'].astype(str)
    prob_melted_full = prob_df_full.melt(
        id_vars='true_cell_type',
        value_vars=[f"scpred_prob_{c}" for c in scpred_model_full.classifier_.classes_],
        var_name='predicted_class_prob_of',
        value_name='probability'
    )
    prob_melted_full['predicted_class_prob_of'] = prob_melted_full['predicted_class_prob_of'].str.replace('scpred_prob_', '')
    for true_type in sorted(prob_melted_full['true_cell_type'].unique()):
        plt.figure(figsize=(10, 5))
        subset_df = prob_melted_full[prob_melted_full['true_cell_type'] == true_type]
        sns.boxplot(data=subset_df, x='predicted_class_prob_of', y='probability', hue='predicted_class_prob_of', palette='viridis', legend=False)
        plt.title(f'Prediction Probabilities for True Cell Type: {true_type} (Paul15 - Full Preprocessing)')
        plt.xlabel('Probability of being Predicted as Class')
        plt.ylabel('Probability')
        plt.ylim(-0.05, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # FIX: Rotate X-axis labels for readability
        plt.xticks(rotation=45, ha='right') # Rotate labels 45 degrees, align right
        plt.show()


# UMAP plots (Full Preprocessing)
if 'X_umap' in query_adata_pred_full.obsm:
    print("\n--- UMAP: True vs Predicted Labels (Paul15 - Full Preprocessing) ---")
    plt.figure(figsize=(12, 6))
    sc.pl.umap(
        query_adata_pred_full,
        color=['cell_type', 'scpred_prediction'],
        title=['True Labels (Paul15 Clusters)', 'scPred Predictions'],
        frameon=False, show=False, ncols=2, wspace=0.3
    )
    plt.suptitle('UMAP of Query Data: True vs Predicted Labels (Paul15 - Full Preprocessing)', y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

    print("\n--- UMAP: Misclassified Cells (Paul15 - Full Preprocessing) ---")
    true_labels_str_full = query_adata_pred_full.obs['cell_type'].astype(str)
    predicted_labels_str_full = query_adata_pred_full.obs['scpred_prediction'].astype(str)
    query_adata_pred_full.obs['misclassified'] = (true_labels_str_full != predicted_labels_str_full).astype(str).astype('category')
    misclassified_palette = {'True': 'red', 'False': 'lightgray'}
    plt.figure(figsize=(8, 6))
    sc.pl.umap(
        query_adata_pred_full,
        color='misclassified', palette=misclassified_palette, size=50, alpha=0.7,
        title='UMAP of Query Data: Misclassified Cells (Paul15 - Full Preprocessing)',
        frameon=False, show=False, legend_loc='upper right' # Fixed legend_loc
    )
    plt.show()

    print("\n--- UMAP: Prediction Confidence (Paul15 - Full Preprocessing) ---")
    if hasattr(scpred_model_full.classifier_, 'predict_proba'):
        def get_predicted_prob(row, classes):
            pred_class = row['scpred_prediction']
            prob_col_name = f'scpred_prob_{pred_class}'
            if prob_col_name in row.index:
                return row[prob_col_name]
            return np.nan
        prob_cols_full_model = [f"scpred_prob_{c}" for c in scpred_model_full.classifier_.classes_]
        if all(col in query_adata_pred_full.obs.columns for col in prob_cols_full_model):
            temp_obs_for_apply_full = query_adata_pred_full.obs[['scpred_prediction'] + prob_cols_full_model]
            query_adata_pred_full.obs['predicted_prob_score'] = temp_obs_for_apply_full.apply(
                lambda row: get_predicted_prob(row, scpred_model_full.classifier_.classes_), axis=1
            )
            plt.figure(figsize=(8, 6))
            sc.pl.umap(
                query_adata_pred_full, color='predicted_prob_score', cmap='viridis',
                title='UMAP of Query Data: Prediction Confidence (Paul15 - Full Preprocessing)',
                frameon=False, vmin=0.0, vmax=1.0, show=False
            )
            plt.show()
    else:
        print("Skipping UMAP by prediction confidence: Classifier does not support `predict_proba`.")
else:
    print("Skipping UMAP plots as 'X_umap' is not available in .obsm.")



print("\n--- Analysis Complete ---")    