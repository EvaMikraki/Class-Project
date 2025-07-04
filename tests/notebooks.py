# 01

# %% [markdown]
# # 1. Data Loading and Preprocessing
# 
# This notebook loads the PBMC3k dataset, performs standard preprocessing,
# and saves it for later use. We will also split it into a "reference"
# and a "query" set for demonstration purposes.

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import os
import sys

# Add project root to path to import our package
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scpred_py_baseline import _preprocessing

# %% [markdown]
# ## Load Data

# %%
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()
print(adata)

# %% [markdown]
# ## Preprocessing
# 
# We apply standard filtering, normalization, log-transform, HVG selection, and scaling.
# We'll use our custom preprocessing function to keep it consistent.

# %%
# Add some cell type annotations for training (using Scanpy's workflow)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_hvg = adata[:, adata.var.highly_variable].copy() # Work on HVG subset
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver='arpack')
sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=40)
sc.tl.louvain(adata_hvg, random_state=42) # Use louvain clusters as 'cell types'

# Map back the cell types to the full (but preprocessed) data
adata.obs['cell_type'] = adata_hvg.obs['louvain']

# Now apply our *intended* preprocessing (without PCA yet) for scPred
adata_proc = _preprocessing.standard_preprocess(adata.copy())
print(adata_proc)


# %% [markdown]
# ## Split Data (Reference vs. Query)
# 
# We'll randomly split the cells into 70% reference and 30% query.

# %%
n_cells = adata_proc.shape[0]
indices = np.arange(n_cells)
np.random.shuffle(indices)

ref_idx = indices[:int(0.7 * n_cells)]
query_idx = indices[int(0.7 * n_cells):]

ref_adata = adata_proc[ref_idx, :].copy()
query_adata = adata_proc[query_idx, :].copy()

print(f"Reference data shape: {ref_adata.shape}")
print(f"Query data shape: {query_adata.shape}")

# %% [markdown]
# ## Save Data
# 
# We'll save these AnnData objects.

# %%
if not os.path.exists('../data/preprocessed'):
    os.makedirs('../data/preprocessed')

ref_adata.write('../data/preprocessed/baseline_pbmc3k_ref.h5ad')
query_adata.write('../data/preprocessed/baseline_pbmc3k_query.h5ad')

print("Data saved.")

# %% [markdown]
# # 2. Training an scPred Model
# 
# This notebook loads the preprocessed reference data and trains
# our `ScPredModel`.

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import os
import sys
import pickle

# Add project root to path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scpred_py_baseline import ScPredModel

# %% [markdown]
# ## Load Reference Data

# %%
ref_adata = ad.read_h5ad('../data/preprocessed/baseline_pbmc3k_ref.h5ad')
print(ref_adata)
print("Cell types:\n", ref_adata.obs['cell_type'].value_counts())

# %% [markdown]
# ## Initialize and Train the Model
# 
# We use the `ScPredModel` class and train it on our reference data.
# We need to specify which column in `.obs` contains the cell type labels.

# %%
scpred_model = ScPredModel()

# Train the model
scpred_model.train(ref_adata, cell_type_key='cell_type', n_components=30)

print("\nModel Trained!")
print("PCA Model:", scpred_model.pca_model_)
print("Classifier:", scpred_model.classifier_)
print("Reference Genes:", len(scpred_model.reference_genes_))


# %% [markdown]
# ## Save the Trained Model
# 
# We can save the trained model object using `pickle` for later use.

# %%
if not os.path.exists('../models'):
    os.makedirs('../models')

with open('../models/scpred_baseline_model_pbmc3k.pkl', 'wb') as f:
    pickle.dump(scpred_model, f)

print("Trained model saved to ../models/scpred_baseline_model_pbmc3k.pkl")

# %% [markdown]
# # 3. Predicting with an scPred Model
# 
# This notebook loads a trained `ScPredModel` and the query data,
# then performs cell type prediction.

# %%
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# %% [markdown]
# ## Load Model and Query Data

# %%
# Load the trained model
with open('../models/scpred_baseline_model_pbmc3k.pkl', 'rb') as f:
    scpred_model = pickle.load(f)

print("Loaded Model:", scpred_model)

# Load the query data
query_adata = ad.read_h5ad('../data/preprocessed/baseline_pbmc3k_query.h5ad')
print("\nQuery Data:", query_adata)

# %% [markdown]
# ## Perform Prediction
# 
# We use the `predict` method of our loaded model.
# **Important**: The current `_core.py` implementation re-fits PCA
# on common genes and scales the query data. This is a simplification
# and a key area to refine based on the original scPred paper for
# maximum accuracy.

# %%
query_adata_pred = scpred_model.predict(query_adata)

print("\nQuery Data with Predictions:")
print(query_adata_pred.obs[['cell_type', 'scpred_prediction']].head())

# %% [markdown]
# ## Evaluate Predictions
# 
# Since our query data *does* have true labels (because we split it),
# we can evaluate the performance.

# %%
true_labels = query_adata_pred.obs['cell_type']
predicted_labels = query_adata_pred.obs['scpred_prediction']

print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels))

# %% [markdown]
# ## Visualize Results
# 
# Let's visualize the confusion matrix.

# %%
cm = confusion_matrix(true_labels, predicted_labels, labels=scpred_model.classifier_.classes_)
cm_df = pd.DataFrame(cm, index=scpred_model.classifier_.classes_, columns=scpred_model.classifier_.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# %% [markdown]
# We can also visualize the UMAP of the query data, colored by true
# and predicted labels.

# %%
# We need to compute UMAP on the query data using its projected PCs
query_adata_pred.obsm['X_scpred_pca'] = scpred_model.predict(query_adata_pred.copy()).obsm['X_scpred_pca'] # Re-run predict to get PCs

# Calculate UMAP based on *our projected* PCs
sc.pp.neighbors(query_adata_pred, n_neighbors=10, use_rep='X_scpred_pca')
sc.tl.umap(query_adata_pred)

# %%
sc.pl.umap(query_adata_pred, color=['cell_type', 'scpred_prediction'], title=['True Labels', 'scPred Predictions'])

# %% [markdown]
# ## Next Steps
# 
# This shows the basic workflow. To improve this, you should focus on:
# 1.  **Hyperparameter Tuning**: Implement `GridSearchCV` in `_training.py`.
# 2.  **PCA Projection Accuracy**: Ensure the scaling and gene handling *exactly* match `scPred`'s method before PCA projection. This might involve saving scaling factors from the reference.
# 3.  **Feature Selection**: Implement the specific informative gene selection used by `scPred`.
# 4.  **Probability Thresholding**: `scPred` includes steps to handle "unassigned" cells based on probability thresholds.
# 5.  **Robustness & Error Handling**: Add more checks and balances.
# 6.  **Testing**: Implement `pytest` tests in the `tests/` directory.




# 02

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
from sklearn.metrics import (
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
from scpred_py_revised import _analysis_utils # Assuming _analysis_utils is part of the package


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


# --- 1. Data Loading and Initial Annotation ---
print("\n--- Step 1: Loading and Annotating Data ---")
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()

print(f"Initial raw AnnData object shape: {adata.shape}")
print(f"Initial raw AnnData .obs keys: {adata.obs.keys().tolist()}")
print(f"Initial raw AnnData .var keys: {adata.var.keys().tolist()}\n")


# Generate 'cell_type' labels using a temporary preprocessing pipeline
# This prepares the `adata` object with raw counts and the `cell_type` column
print("Generating temporary cell type labels via Louvain clustering...")
temp_adata_for_labels = adata.copy()
sc.pp.filter_cells(temp_adata_for_labels, min_genes=200)
sc.pp.filter_genes(temp_adata_for_labels, min_cells=3)
sc.pp.normalize_total(temp_adata_for_labels, target_sum=1e4)
sc.pp.log1p(temp_adata_for_labels)
sc.pp.highly_variable_genes(temp_adata_for_labels, min_mean=0.0125, max_mean=3, min_disp=0.5, flavor='seurat')
temp_adata_hvg_for_labels = temp_adata_for_labels[:, temp_adata_for_labels.var.highly_variable].copy()
sc.pp.scale(temp_adata_hvg_for_labels, max_value=10)
sc.tl.pca(temp_adata_hvg_for_labels, svd_solver='arpack', random_state=RANDOM_STATE) # Added random_state
sc.pp.neighbors(temp_adata_hvg_for_labels, n_neighbors=10, n_pcs=40, random_state=RANDOM_STATE) # Added random_state
sc.tl.louvain(temp_adata_hvg_for_labels, random_state=RANDOM_STATE, key_added='cell_type') # Added random_state

# Assign the generated cell_type labels back to the original raw adata object
adata.obs['cell_type'] = temp_adata_hvg_for_labels.obs['cell_type'].reindex(adata.obs_names).astype('category') # Ensure category type

print(f"Full dataset (raw counts + Louvain labels) shape: {adata.shape}")
print(f"Full dataset .obs keys after labeling: {adata.obs.keys().tolist()}")
print("Cell type distribution (all data):\n", adata.obs['cell_type'].value_counts())


# --- 2. Train-Test Split (Stratified) ---
# We split the raw data (with labels) into reference and query sets.
print("\n--- Step 2: Splitting Data (Stratified) ---")
indices = range(adata.n_obs)

ref_idx, query_idx = train_test_split(
    indices,
    test_size=0.3,
    random_state=RANDOM_STATE, # Ensuring split is reproducible
    stratify=adata.obs['cell_type'] # CRITICAL for balanced classes
)

# These AnnData objects contain raw counts and cell_type labels.
# The ScPredModel will handle their internal preprocessing.
ref_adata_raw = adata[ref_idx, :].copy()
query_adata_raw = adata[query_idx, :].copy()

print(f"Reference data (raw + labels) shape: {ref_adata_raw.shape}")
print(f"Query data (raw + labels) shape: {query_adata_raw.shape}")
print("\nReference cell type distribution:\n", ref_adata_raw.obs['cell_type'].value_counts())
print("\nQuery cell type distribution:\n", query_adata_raw.obs['cell_type'].value_counts())


# Calculate and report class imbalance
print("\n--- Class Imbalance Analysis ---")
print("Reference data class imbalance (proportion):\n", ref_adata_raw.obs['cell_type'].value_counts(normalize=True).sort_index())
print("Query data class imbalance (proportion):\n", query_adata_raw.obs['cell_type'].value_counts(normalize=True).sort_index())


# --- 3. Training the scPred Model ---
# The ScPredModel will now handle all preprocessing (normalization, log1p, HVG, scaling, PCA) internally.
print("\n--- Step 3: Training scPred Model ---")
scpred_model = ScPredModel()

# Train the model on the raw reference data.
# We are also passing some configuration options for HVG selection and SVM.
scpred_model.train(
    ref_adata=ref_adata_raw, # Pass the raw reference data
    cell_type_key='cell_type',
    n_components=30,
    hvg_n_top_genes=1000, # Example: Select top 1000 HVGs
    hvg_flavor='seurat',  # Using 'seurat' flavor for HVG selection
    svm_kernel='rbf',     # Using RBF kernel for SVM (more common in scPred)
    svm_c=1.0,            # Default C parameter for SVM
    svm_random_state=RANDOM_STATE # Ensuring SVM training is reproducible
)

print("\nModel Trained!")
print(f"Scaler: {scpred_model.scaler_}")
print(f"PCA Model: {scpred_model.pca_model_}")
print(f"Classifier: {scpred_model.classifier_}")
print(f"Reference HVGs learned: {len(scpred_model.reference_hvg_genes_)} genes")

# Save the trained ScPredModel object using its new method
model_path = os.path.join(MODELS_DIR, 'scpred_revised_model_pbmc3k.pkl')
scpred_model.save(model_path)

# Save the preprocessed reference data (now accessible via scpred_model.ref_adata_processed_)
if scpred_model.ref_adata_processed_ is not None:
    preprocessed_ref_path = os.path.join(PREPROCESSED_DATA_DIR, 'revised_pbmc3k_ref_preprocessed.h5ad')
    scpred_model.ref_adata_processed_.write(preprocessed_ref_path)
    print(f"Preprocessed reference data saved to: {preprocessed_ref_path}")
else:
    print("Warning: scpred_model.ref_adata_processed_ is None. Preprocessed reference data not saved.")


# --- 4. Predicting with the scPred Model ---
# The model will use its saved preprocessing steps and trained classifier to predict.
print("\n--- Step 4: Predicting on Query Data ---")
query_adata_pred = scpred_model.predict(query_adata_raw) # Pass the raw query data

print("\nQuery Data with Predictions (first 5 rows):")
print(query_adata_pred.obs[['cell_type', 'scpred_prediction']].head())
print(f"\nQuery data .obs keys after prediction: {query_adata_pred.obs.keys().tolist()}")
print(f"Query data .obsm keys after prediction: {list(query_adata_pred.obsm.keys())}")

# Save the preprocessed query data with predictions
preprocessed_query_path = os.path.join(PREPROCESSED_DATA_DIR, 'revised_pbmc3k_query_pred_preprocessed.h5ad')
query_adata_pred.write(preprocessed_query_path)
print(f"Preprocessed query data with predictions saved to: {preprocessed_query_path}")


# --- 5. Evaluating Predictions (using _analysis_utils for metrics) ---
true_labels = query_adata_pred.obs['cell_type']
predicted_labels = query_adata_pred.obs['scpred_prediction']

# Extract prediction probabilities (columns starting with 'scpred_prob_')
# and ensure they are ordered according to the classifier's classes
pred_prob_cols = [col for col in query_adata_pred.obs.columns if col.startswith('scpred_prob_')]
y_pred_probs_df = None
if len(pred_prob_cols) > 0:
    # Ensure columns are sorted by class name to match classifier_.classes_ order if needed
    # For now, assuming scpred_prob_0, scpred_prob_1, etc. directly correspond to classes_ order
    y_pred_probs_df = query_adata_pred.obs[pred_prob_cols].copy()

# Use the helper function for metrics
metrics_results = _analysis_utils.evaluate_and_report_metrics(
    true_labels=true_labels,
    predicted_labels=predicted_labels,
    classifier_classes=scpred_model.classifier_.classes_, # Pass classifier classes for robust reporting
    y_pred_probs=y_pred_probs_df
)


# --- 6. Visualizing Results (plotting code directly in notebook) ---
print("\n--- Step 6: Visualizing Results ---")

# Ensure UMAP embedding is computed on the projected PCs if not already present
if 'X_scpred_pca' in query_adata_pred.obsm and 'X_umap' not in query_adata_pred.obsm:
    print("Computing neighbors and UMAP based on X_scpred_pca for visualization...")
    sc.pp.neighbors(query_adata_pred, n_neighbors=10, use_rep='X_scpred_pca', random_state=RANDOM_STATE) # Added random_state
    sc.tl.umap(query_adata_pred, random_state=RANDOM_STATE) # Added random_state
    print(f"Query data .obsm keys after UMAP: {list(query_adata_pred.obsm.keys())}")
elif 'X_umap' not in query_adata_pred.obsm:
    print("X_scpred_pca or X_umap not found in .obsm. UMAP plots might not be generated.")


# Plot 1: Confusion Matrix
print("\n--- Confusion Matrix ---")
# Ensure labels are string type consistently
true_labels_str = true_labels.astype(str)
predicted_labels_str = predicted_labels.astype(str)
cm_labels_str = sorted(true_labels_str.unique().tolist()) # Get all true labels for consistency
cm = confusion_matrix(true_labels_str, predicted_labels_str, labels=cm_labels_str)
cm_df = pd.DataFrame(cm, index=cm_labels_str, columns=cm_labels_str)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Cells'})
plt.title('Confusion Matrix (True vs. Predicted Labels)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Plot 2: Per-Class ROC AUC (One-vs-Rest)
print("\n--- Per-Class ROC AUC (One-vs-Rest) ---")
# ROC AUC is only meaningful if classifier.predict_proba is available
if hasattr(scpred_model.classifier_, 'predict_proba'):
    # Prepare data for one-vs-rest ROC AUC
    classes = scpred_model.classifier_.classes_
    
    # Create binary labels for each class (one-hot encoding style)
    y_true_binary = pd.get_dummies(true_labels_str, columns=classes, drop_first=False).values
    
    # Get prediction probabilities
    y_scores = y_pred_probs_df[[f"scpred_prob_{c}" for c in classes]].values # Use the df we already prepared

    plt.figure(figsize=(10, 8))
    for i, class_label in enumerate(classes):
        # Check if class has both 0 and 1 in true labels for ROC calculation
        if y_true_binary.shape[1] > i and len(np.unique(y_true_binary[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')
        else:
            print(f"Skipping ROC plot for Class {class_label}: Not enough unique true labels.")

    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - One-vs-Rest')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
else:
    print("Classifier does not support `predict_proba`, cannot compute ROC AUC.")


# Plot 3: Distribution of Prediction Probabilities by True Class
print("\n--- Distribution of Prediction Probabilities by True Class ---")
if hasattr(scpred_model.classifier_, 'predict_proba'):
    prob_df = query_adata_pred.obs[[f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_]].copy()
    prob_df['true_cell_type'] = query_adata_pred.obs['cell_type'].astype(str)

    # Melt the DataFrame for easier plotting with seaborn
    prob_melted = prob_df.melt(
        id_vars='true_cell_type',
        value_vars=[f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_],
        var_name='predicted_class_prob_of',
        value_name='probability'
    )
    prob_melted['predicted_class_prob_of'] = prob_melted['predicted_class_prob_of'].str.replace('scpred_prob_', '')

    # Plotting loop for each true cell type
    for true_type in sorted(prob_melted['true_cell_type'].unique()):
        plt.figure(figsize=(10, 5))
        subset_df = prob_melted[prob_melted['true_cell_type'] == true_type]
        sns.boxplot(
            data=subset_df,
            x='predicted_class_prob_of',
            y='probability',
            hue='predicted_class_prob_of',
            palette='viridis',
            legend=False
        )
        plt.title(f'Prediction Probabilities for True Cell Type: {true_type}')
        plt.xlabel('Probability of being Predicted as Class')
        plt.ylabel('Probability')
        plt.ylim(-0.05, 1.05) # Consistent y-axis limits
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
else:
    print("Classifier does not support `predict_proba`, cannot visualize probability distributions.")


# Plot 4: True vs Predicted Labels UMAP
if 'X_umap' in query_adata_pred.obsm:
    print("\n--- Plotting UMAP: True vs Predicted Labels ---")
    plt.figure(figsize=(12, 6)) # Use figsize here to control the overall figure size
    
    sc.pl.umap(
        query_adata_pred,
        color=['cell_type', 'scpred_prediction'],
        title=['True Labels (Louvain Clusters)', 'scPred Predictions'],
        frameon=False,
        show=False, # Set show=False to control display manually
        ncols=2,
        wspace=0.3 # Adjust horizontal space between subplots
    )
    plt.suptitle('UMAP of Query Data: True vs Predicted Labels', y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
else:
    print("Skipping UMAP: True vs Predicted Labels as 'X_umap' is not available in .obsm.")


# Plot 5: Misclassified Cells UMAP
if 'X_umap' in query_adata_pred.obsm:
    print("\n--- Plotting Misclassified Cells UMAP ---")
    # Ensure true_labels_str and predicted_labels_str are available (as they are in notebook scope)
    true_labels_str = query_adata_pred.obs['cell_type'].astype(str)
    predicted_labels_str = query_adata_pred.obs['scpred_prediction'].astype(str)

    query_adata_pred.obs['misclassified'] = (true_labels_str != predicted_labels_str).astype(str).astype('category')
    misclassified_palette = {'True': 'red', 'False': 'lightgray'}
    
    plt.figure(figsize=(8, 6))
    sc.pl.umap(
        query_adata_pred,
        color='misclassified',
        palette=misclassified_palette,
        size=50,
        alpha=0.7,
        title='UMAP of Query Data: Misclassified Cells',
        frameon=False,
        show=False,
        # FIX: Change legend_loc from 'on data' to 'upper right' for better visibility
        legend_loc='upper right' # Changed legend location
    )
    plt.show()
else:
    print("Skipping Misclassified Cells UMAP as 'X_umap' is not available in .obsm.")


# Plot 6: UMAP colored by Prediction Confidence
if 'X_umap' in query_adata_pred.obsm and hasattr(scpred_model.classifier_, 'predict_proba'):
    print("\n--- Plotting UMAP by Prediction Confidence ---")
    # Get the probability for the *predicted* class for each cell
    def get_predicted_prob(row, classes):
        pred_class = row['scpred_prediction']
        prob_col_name = f'scpred_prob_{pred_class}'
        if prob_col_name in row.index:
            return row[prob_col_name]
        return np.nan

    prob_cols = [f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_]
    
    if not all(col in query_adata_pred.obs.columns for col in prob_cols):
        print("Skipping UMAP by prediction confidence: Required probability columns not found in .obs.")
    else:
        temp_obs_for_apply = query_adata_pred.obs[['scpred_prediction'] + prob_cols]
        query_adata_pred.obs['predicted_prob_score'] = temp_obs_for_apply.apply(
            lambda row: get_predicted_prob(row, scpred_model.classifier_.classes_), axis=1
        )

        plt.figure(figsize=(8, 6))
        sc.pl.umap(
            query_adata_pred,
            color='predicted_prob_score',
            cmap='viridis',
            title='UMAP of Query Data: Prediction Confidence',
            frameon=False,
            vmin=0.0, vmax=1.0,
            show=False
        )
        plt.show()
else:
    print("Skipping UMAP by prediction confidence: 'X_umap' not available or classifier does not support `predict_proba`.")

print("\n--- Analysis Complete ---")





# 03

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
model_full_path = os.path.join(MODELS_DIR, 'scpred_revised_model_paul15_full_preprocessing.pkl')
scpred_model_full.save(model_full_path)

# Save the preprocessed reference data for scenario 1
if hasattr(scpred_model_full, 'ref_adata_processed_') and scpred_model_full.ref_adata_processed_ is not None:
    preprocessed_ref_full_path = os.path.join(PREPROCESSED_DATA_DIR, 'revised_paul15_ref_preprocessed_full.h5ad')
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
preprocessed_query_full_path = os.path.join(PREPROCESSED_DATA_DIR, 'revised_paul15_query_pred_preprocessed_full.h5ad')
query_adata_pred_full.write(preprocessed_query_full_path)
print(f"Preprocessed query data with predictions (Full Preprocessing) saved to: {preprocessed_query_full_path}")


# --- 5a. Evaluating Predictions (Full Preprocessing) ---
# FIX: Filter out NaN predictions for evaluation to avoid ValueError
valid_cells_full = query_adata_pred_full.obs['scpred_prediction'].notna()
true_labels_full = query_adata_pred_full.obs.loc[valid_cells_full, 'cell_type']
predicted_labels_full = query_adata_pred_full.obs.loc[valid_cells_full, 'scpred_prediction']

pred_prob_cols_full = [col for col in query_adata_pred_full.obs.columns if col.startswith('scpred_prob_')]
y_pred_probs_df_full = None
if len(pred_prob_cols_full) > 0:
    y_pred_probs_df_full = query_adata_pred_full.obs.loc[valid_cells_full, pred_prob_cols_full].copy()
    # Fill any remaining NaNs in probabilities with 0 (should ideally not happen if filtering by notna 'scpred_prediction')
    y_pred_probs_df_full = y_pred_probs_df_full.fillna(0.0)


print("\n--- Evaluation for Full Preprocessing Model ---")
metrics_results_full = _analysis_utils.evaluate_and_report_metrics(
    true_labels=true_labels_full,
    predicted_labels=predicted_labels_full,
    classifier_classes=scpred_model_full.classifier_.classes_,
    y_pred_probs=y_pred_probs_df_full
)


# --- 6a. Visualizing Results (Full Preprocessing) ---
print("\n--- Step 6a: Visualizing Results (Full Preprocessing) ---")

# For UMAP plots, we need the AnnData object to be filtered to only valid cells or handle NaNs in plots.
# It's safer to create a temporary filtered AnnData for plotting to avoid issues with NaN coordinates.
query_adata_pred_full_filtered_for_umap = query_adata_pred_full[valid_cells_full, :].copy()

# FIX: Remove potentially stale 'iroot' and 'neighbors' graph info from .uns of the subsetted AnnData
# This helps prevent warnings like "Root cell index X does not exist..."
if 'iroot' in query_adata_pred_full_filtered_for_umap.uns:
    del query_adata_pred_full_filtered_for_umap.uns['iroot']
if 'neighbors' in query_adata_pred_full_filtered_for_umap.uns:
    del query_adata_pred_full_filtered_for_umap.uns['neighbors']


if 'X_scpred_pca' in query_adata_pred_full_filtered_for_umap.obsm and 'X_umap' not in query_adata_pred_full_filtered_for_umap.obsm:
    print("Computing neighbors and UMAP based on X_scpred_pca for visualization...")
    sc.pp.neighbors(query_adata_pred_full_filtered_for_umap, n_neighbors=10, use_rep='X_scpred_pca', random_state=RANDOM_STATE) # Added random_state
    sc.tl.umap(query_adata_pred_full_filtered_for_umap, random_state=RANDOM_STATE) # Added random_state
    print(f"Query data .obsm keys after UMAP: {list(query_adata_pred_full_filtered_for_umap.obsm.keys())}")
elif 'X_umap' not in query_adata_pred_full_filtered_for_umap.obsm:
    print("X_scpred_pca or X_umap not found in .obsm. UMAP plots might not be generated.")

# Plot 1: Confusion Matrix
print("\n--- Confusion Matrix (Full Preprocessing) ---")
true_labels_str_full = true_labels_full.astype(str) # Use already filtered labels
predicted_labels_str_full = predicted_labels_full.astype(str) # Use already filtered labels
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
if 'X_umap' in query_adata_pred_full_filtered_for_umap.obsm: # Use filtered adata for UMAP
    print("\n--- UMAP: True vs Predicted Labels (Paul15 - Full Preprocessing) ---")
    plt.figure(figsize=(12, 6))
    sc.pl.umap(
        query_adata_pred_full_filtered_for_umap, # Use filtered adata for UMAP
        color=['cell_type', 'scpred_prediction'],
        title=['True Labels (Paul15 Clusters)', 'scPred Predictions'],
        frameon=False, show=False, ncols=2, wspace=0.3
    )
    plt.suptitle('UMAP of Query Data: True vs Predicted Labels (Paul15 - Full Preprocessing)', y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

    print("\n--- UMAP: Misclassified Cells (Paul15 - Full Preprocessing) ---")
    # true_labels_str_full and predicted_labels_str_full are already filtered
    query_adata_pred_full_filtered_for_umap.obs['misclassified'] = (true_labels_str_full != predicted_labels_str_full).astype(str).astype('category')
    misclassified_palette = {'True': 'red', 'False': 'lightgray'}
    plt.figure(figsize=(8, 6))
    sc.pl.umap(
        query_adata_pred_full_filtered_for_umap, # Use filtered adata for UMAP
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
        prob_cols_full_model = [col for col in query_adata_pred_full_filtered_for_umap.obs.columns if col.startswith('scpred_prob_')] # Filter cols based on filtered adata
        if all(col in query_adata_pred_full_filtered_for_umap.obs.columns for col in prob_cols_full_model): # Check against filtered adata
            temp_obs_for_apply_full = query_adata_pred_full_filtered_for_umap.obs[['scpred_prediction'] + prob_cols_full_model]
            query_adata_pred_full_filtered_for_umap.obs['predicted_prob_score'] = temp_obs_for_apply_full.apply(
                lambda row: get_predicted_prob(row, scpred_model_full.classifier_.classes_), axis=1
            )
            plt.figure(figsize=(8, 6))
            sc.pl.umap(
                query_adata_pred_full_filtered_for_umap, # Use filtered adata for UMAP
                color='predicted_prob_score', cmap='viridis',
                title='UMAP of Query Data: Prediction Confidence (Paul15 - Full Preprocessing)',
                frameon=False, vmin=0.0, vmax=1.0, show=False
            )
            plt.show()
    else:
        print("Skipping UMAP by prediction confidence: Classifier does not support `predict_proba`.")
else:
    print("Skipping UMAP plots as 'X_umap' is not available in .obsm.")


# ==============================================================================
# DEMONSTRATION 2: scPredModel with perform_preprocessing=False
# User explicitly preprocesses data before passing to model.
# This will *avoid* the redundant log1p application.
# ==============================================================================
print("\n======== DEMONSTRATION 2: scPredModel (perform_preprocessing=False) ========")

# --- Preprocessing Data MANUALLY for perform_preprocessing=False mode ---
print("\n--- Manually Preprocessing Data for perform_preprocessing=False ---")
# Create copies to avoid modifying original raw splits
ref_adata_manual_prep = ref_adata_raw_split.copy()
query_adata_manual_prep = query_adata_raw_split.copy()

# Step 1: Normalize and Log1p (Manually)
sc.pp.normalize_total(ref_adata_manual_prep, target_sum=1e4)
sc.pp.log1p(ref_adata_manual_prep)
# IMPORTANT: Manually set the 'log1p' flag in adata.uns to inform downstream tools
ref_adata_manual_prep.uns['log1p'] = {'base': None} # Indicate it's log-transformed

sc.pp.normalize_total(query_adata_manual_prep, target_sum=1e4)
sc.pp.log1p(query_adata_manual_prep)
query_adata_manual_prep.uns['log1p'] = {'base': None}

print("  Manual normalization and log1p finished for reference and query.")

# Step 2: Highly Variable Genes selection (on reference)
# For simplicity, we'll apply it after log1p here to get HVGs on the manually processed data.
ref_adata_manual_prep, ref_hvg_genes_manual = _preprocessing.select_highly_variable_genes(
    ref_adata_manual_prep, n_top_genes=2000, flavor='seurat' # Added flavor and n_top_genes
)
print(f"  Manually selected {ref_adata_manual_prep.shape[1]} HVGs for reference.")


# Step 3: Align query genes to reference HVGs
# The _preprocessing.align_genes_to_reference function will handle this.
# This aligns the genes in query_adata_manual_prep to the common HVGs found in reference.
query_expr_df_manual = pd.DataFrame(
    query_adata_manual_prep.X.toarray() if hasattr(query_adata_manual_prep.X, 'toarray') else query_adata_manual_prep.X,
    index=query_adata_manual_prep.obs_names,
    columns=query_adata_manual_prep.var_names
)
aligned_query_expr_df_manual = query_expr_df_manual.reindex(columns=ref_hvg_genes_manual, fill_value=0.0)

aligned_query_manual_prep = ad.AnnData(
    X=aligned_query_expr_df_manual.values,
    obs=query_adata_manual_prep.obs.loc[aligned_query_expr_df_manual.index],
    var=pd.DataFrame(index=aligned_query_expr_df_manual.columns)
)
print(f"  Manually aligned query data shape: {aligned_query_manual_prep.shape}")

# Save the manually preprocessed reference data for scenario 2
preprocessed_ref_manual_path = os.path.join(PREPROCESSED_DATA_DIR, 'revised_paul15_ref_preprocessed_manual.h5ad')
ref_adata_manual_prep.write(preprocessed_ref_manual_path)
print(f"Manually preprocessed reference data (Manual Preprocessing) saved to: {preprocessed_ref_manual_path}")


# --- 3b. Training the scPred Model (perform_preprocessing=False) ---
print("\n--- Step 3b: Training scPred Model (perform_preprocessing=False) ---")
scpred_model_partial = ScPredModel()

# Pass the manually preprocessed data (normalized, log1p, HVG-selected)
# Model will then only perform scaling and PCA internally.
scpred_model_partial.train(
    ref_adata=ref_adata_manual_prep, # Pass the manually preprocessed data
    cell_type_key='cell_type',
    n_components=30, # Same as before
    # hvg parameters are now ignored by ScPredModel's train when perform_preprocessing=False
    svm_kernel='rbf',
    svm_c=1.0,
    svm_random_state=RANDOM_STATE, # Ensuring SVM training is reproducible
)

print("\nModel Trained (Manual Preprocessing)!")
print(f"Scaler: {scpred_model_partial.scaler_}")
print(f"PCA Model: {scpred_model_partial.pca_model_}")
print(f"Classifier: {scpred_model_partial.classifier_}")
print(f"Reference HVGs learned: {len(scpred_model_partial.reference_hvg_genes_)} genes")

# Save the trained ScPredModel object for scenario 2
model_partial_path = os.path.join(MODELS_DIR, 'scpred_revised_model_paul15_manual_preprocessing.pkl')
scpred_model_partial.save(model_partial_path)


# --- 4b. Predicting with the scPred Model (perform_preprocessing=False) ---
print("\n--- Step 4b: Predicting on Query Data (perform_preprocessing=False) ---")
# Pass the manually preprocessed query data (normalized, log1p, aligned to reference HVGs)
query_adata_pred_partial = scpred_model_partial.predict(aligned_query_manual_prep) # Simplified call now, perform_preprocessing=False is default.

print("\nQuery Data with Predictions (Manual Preprocessing, first 5 rows):")
print(query_adata_pred_partial.obs[['cell_type', 'scpred_prediction']].head())

# Save the preprocessed query data with predictions for scenario 2
preprocessed_query_manual_path = os.path.join(PREPROCESSED_DATA_DIR, 'revised_paul15_query_pred_preprocessed_manual.h5ad')
query_adata_pred_partial.write(preprocessed_query_manual_path)
print(f"Manually preprocessed query data with predictions (Manual Preprocessing) saved to: {preprocessed_query_manual_path}")


# --- 5b. Evaluating Predictions (Manual Preprocessing) ---
# FIX: Filter out NaN predictions for evaluation to avoid ValueError
valid_cells_partial = query_adata_pred_partial.obs['scpred_prediction'].notna()
true_labels_partial = query_adata_pred_partial.obs.loc[valid_cells_partial, 'cell_type']
predicted_labels_partial = query_adata_pred_partial.obs.loc[valid_cells_partial, 'scpred_prediction']

pred_prob_cols_partial = [col for col in query_adata_pred_partial.obs.columns if col.startswith('scpred_prob_')]
y_pred_probs_df_partial = None
if len(pred_prob_cols_partial) > 0:
    y_pred_probs_df_partial = query_adata_pred_partial.obs.loc[valid_cells_partial, pred_prob_cols_partial].copy()
    # Fill any remaining NaNs in probabilities with 0
    y_pred_probs_df_partial = y_pred_probs_df_partial.fillna(0.0)


print("\n--- Evaluation for Manual Preprocessing Model ---")
metrics_results_partial = _analysis_utils.evaluate_and_report_metrics(
    true_labels=true_labels_partial,
    predicted_labels=predicted_labels_partial,
    classifier_classes=scpred_model_partial.classifier_.classes_,
    y_pred_probs=y_pred_probs_df_partial
)

# --- 6b. Visualizing Results (Manual Preprocessing) ---
# For brevity, we'll only compute UMAP for this run and assume similar plots.
# You can copy/paste/adapt plotting code from 6a here if you want to see all plots for this mode.
print("\n--- Step 6b: Visualizing Results (Manual Preprocessing) ---")

# Create a temporary filtered AnnData for plotting for scenario 2
query_adata_pred_partial_filtered_for_umap = query_adata_pred_partial[valid_cells_partial, :].copy()

# FIX: Remove potentially stale 'iroot' and 'neighbors' graph info from .uns of the subsetted AnnData
# This helps prevent warnings like "Root cell index X does not exist..."
if 'iroot' in query_adata_pred_partial_filtered_for_umap.uns:
    del query_adata_pred_partial_filtered_for_umap.uns['iroot']
if 'neighbors' in query_adata_pred_partial_filtered_for_umap.uns:
    del query_adata_pred_partial_filtered_for_umap.uns['neighbors']


if 'X_scpred_pca' in query_adata_pred_partial_filtered_for_umap.obsm and 'X_umap' not in query_adata_pred_partial_filtered_for_umap.obsm:
    print("Computing neighbors and UMAP based on X_scpred_pca for visualization...")
    sc.pp.neighbors(query_adata_pred_partial_filtered_for_umap, n_neighbors=10, use_rep='X_scpred_pca', random_state=RANDOM_STATE) # Added random_state
    sc.tl.umap(query_adata_pred_partial_filtered_for_umap, random_state=RANDOM_STATE) # Added random_state
    print(f"Query data .obsm keys after UMAP: {list(query_adata_pred_partial_filtered_for_umap.obsm.keys())}")
elif 'X_umap' not in query_adata_pred_partial_filtered_for_umap.obsm:
    print("X_scpred_pca or X_umap not found in .obsm. UMAP plots might not be generated.")

if 'X_umap' in query_adata_pred_partial_filtered_for_umap.obsm:
    print("\n--- UMAP: True vs Predicted Labels (Paul15 - Manual Preprocessing) ---")
    plt.figure(figsize=(12, 6))
    sc.pl.umap(
        query_adata_pred_partial_filtered_for_umap, # Use filtered adata for UMAP
        color=['cell_type', 'scpred_prediction'],
        title=['True Labels (Paul15 Clusters)', 'scPred Predictions'],
        frameon=False, show=False, ncols=2, wspace=0.3
    )
    plt.suptitle('UMAP of Query Data: True vs Predicted Labels (Paul15 - Manual Preprocessing)', y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
else:
    print("Skipping UMAP: True vs Predicted Labels as 'X_umap' is not available in .obsm.")

print("\n--- Analysis Complete --")




# 04

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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import LabelBinarizer

# Adjust the path to import your scpred_py_revised package
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Importing the main ScPredModel class and utilities
from scpred_py_final_tmp import ScPredModel # Adjusted to scpred_py_revised as per context
from scpred_py_final_tmp import _analysis_utils # Assuming _analysis_utils is part of the package


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


# --- 1. Data Loading and Initial Annotation ---
print("\n--- Step 1: Loading and Annotating Data ---")
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()

print(f"Initial raw AnnData object shape: {adata.shape}")
print(f"Initial raw AnnData .obs keys: {adata.obs.keys().tolist()}")
print(f"Initial raw AnnData .var keys: {adata.var.keys().tolist()}\n")


# Generate 'cell_type' labels using a temporary preprocessing pipeline
# This prepares the `adata` object with raw counts and the `cell_type` column
print("Generating temporary cell type labels via Louvain clustering...")
temp_adata_for_labels = adata.copy()
sc.pp.filter_cells(temp_adata_for_labels, min_genes=200)
sc.pp.filter_genes(temp_adata_for_labels, min_cells=3)
sc.pp.normalize_total(temp_adata_for_labels, target_sum=1e4)
sc.pp.log1p(temp_adata_for_labels)
sc.pp.highly_variable_genes(temp_adata_for_labels, min_mean=0.0125, max_mean=3, min_disp=0.5, flavor='seurat')
temp_adata_hvg_for_labels = temp_adata_for_labels[:, temp_adata_for_labels.var.highly_variable].copy()
sc.pp.scale(temp_adata_hvg_for_labels, max_value=10)
sc.tl.pca(temp_adata_hvg_for_labels, svd_solver='arpack', random_state=RANDOM_STATE) # Added random_state
sc.pp.neighbors(temp_adata_hvg_for_labels, n_neighbors=10, n_pcs=40, random_state=RANDOM_STATE) # Added random_state
sc.tl.louvain(temp_adata_hvg_for_labels, random_state=RANDOM_STATE, key_added='cell_type') # Added random_state

# Assign the generated cell_type labels back to the original raw adata object
adata.obs['cell_type'] = temp_adata_hvg_for_labels.obs['cell_type'].reindex(adata.obs_names).astype('category') # Ensure category type

print(f"Full dataset (raw counts + Louvain labels) shape: {adata.shape}")
print(f"Full dataset .obs keys after labeling: {adata.obs.keys().tolist()}")
print("Cell type distribution (all data):\n", adata.obs['cell_type'].value_counts())


# --- 2. Train-Test Split (Stratified) ---
# We split the raw data (with labels) into reference and query sets.
print("\n--- Step 2: Splitting Data (Stratified) ---")
indices = range(adata.n_obs)

ref_idx, query_idx = train_test_split(
    indices,
    test_size=0.3,
    random_state=RANDOM_STATE, # Ensuring split is reproducible
    stratify=adata.obs['cell_type'] # CRITICAL for balanced classes
)

# These AnnData objects contain raw counts and cell_type labels.
# The ScPredModel will handle their internal preprocessing.
ref_adata_raw = adata[ref_idx, :].copy()
query_adata_raw = adata[query_idx, :].copy()

print(f"Reference data (raw + labels) shape: {ref_adata_raw.shape}")
print(f"Query data (raw + labels) shape: {query_adata_raw.shape}")
print("\nReference cell type distribution:\n", ref_adata_raw.obs['cell_type'].value_counts())
print("\nQuery cell type distribution:\n", query_adata_raw.obs['cell_type'].value_counts())


# Calculate and report class imbalance
print("\n--- Class Imbalance Analysis ---")
print("Reference data class imbalance (proportion):\n", ref_adata_raw.obs['cell_type'].value_counts(normalize=True).sort_index())
print("Query data class imbalance (proportion):\n", query_adata_raw.obs['cell_type'].value_counts(normalize=True).sort_index())


# --- 3. Training the scPred Model ---
# The ScPredModel will now handle all preprocessing (normalization, log1p, HVG, scaling, PCA) internally.
# It will also use class_weight='balanced' internally for the SVM.
print("\n--- Step 3: Training scPred Model ---")
scpred_model = ScPredModel()

# Train the model on the raw reference data.
# We are also passing some configuration options for HVG selection and SVM.
scpred_model.train(
    ref_adata=ref_adata_raw, # Pass the raw reference data
    cell_type_key='cell_type',
    n_components=30,
    hvg_n_top_genes=1000, # Example: Select top 1000 HVGs
    hvg_flavor='seurat',  # Using 'seurat' flavor for HVG selection
    svm_kernel='rbf',     # Using RBF kernel for SVM (more common in scPred)
    svm_c=1.0,            # Default C parameter for SVM
    svm_random_state=RANDOM_STATE # Ensuring SVM training is reproducible
    # Note: class_weight='balanced' is now set directly inside _training.py's train_svm function,
    # so no need to pass it here.
)

print("\nModel Trained!")
print(f"Scaler: {scpred_model.scaler_}")
print(f"PCA Model: {scpred_model.pca_model_}")
print(f"Classifier: {scpred_model.classifier_}")
print(f"Reference HVGs learned: {len(scpred_model.reference_hvg_genes_)} genes")

# Save the trained ScPredModel object using its new method
model_path = os.path.join(MODELS_DIR, 'scpred_final_model_pbmc3k.pkl')
scpred_model.save(model_path)

# Save the preprocessed reference data (now accessible via scpred_model.ref_adata_processed_)
if scpred_model.ref_adata_processed_ is not None:
    preprocessed_ref_path = os.path.join(PREPROCESSED_DATA_DIR, 'final_pbmc3k_ref_preprocessed.h5ad')
    scpred_model.ref_adata_processed_.write(preprocessed_ref_path)
    print(f"Preprocessed reference data saved to: {preprocessed_ref_path}")
else:
    print("Warning: scpred_model.ref_adata_processed_ is None. Preprocessed reference data not saved.")


# --- 4. Predicting with the scPred Model ---
# The model will use its saved preprocessing steps and trained classifier to predict.
print("\n--- Step 4: Predicting on Query Data ---")
# FIX: Add threshold parameter for "unassigned" cells
PREDICTION_THRESHOLD = 0.8 # Example threshold: predictions below 80% confidence will be 'unassigned'
query_adata_pred = scpred_model.predict(query_adata_raw, threshold=PREDICTION_THRESHOLD)

print("\nQuery Data with Predictions (first 5 rows):")
print(query_adata_pred.obs[['cell_type', 'scpred_prediction']].head())
print(f"\nQuery data .obs keys after prediction: {query_adata_pred.obs.keys().tolist()}")
print(f"Query data .obsm keys after prediction: {list(query_adata_pred.obsm.keys())}")
print("\nPredicted label distribution (including 'unassigned'):\n", query_adata_pred.obs['scpred_prediction'].value_counts(dropna=False))


# Save the preprocessed query data with predictions
preprocessed_query_path = os.path.join(PREPROCESSED_DATA_DIR, 'final_pbmc3k_query_pred_preprocessed.h5ad')
query_adata_pred.write(preprocessed_query_path)
print(f"Preprocessed query data with predictions saved to: {preprocessed_query_path}")


# --- 5. Evaluating Predictions (using _analysis_utils for metrics) ---
true_labels = query_adata_pred.obs['cell_type']
predicted_labels = query_adata_pred.obs['scpred_prediction']

# Extract prediction probabilities (columns starting with 'scpred_prob_')
# Ensure they are ordered according to the classifier's classes for ROC AUC if needed.
# Note: y_pred_probs_df should contain probabilities ONLY for the classes, not 'unassigned'.
pred_prob_cols = [col for col in query_adata_pred.obs.columns if col.startswith('scpred_prob_')]
y_pred_probs_df = None
if len(pred_prob_cols) > 0:
    # Filter probabilities to only those cells that were not assigned "unassigned" due to threshold,
    # as evaluation metrics like Balanced Accuracy and MCC are typically for assigned predictions.
    # The _analysis_utils function will handle filtering based on `predicted_labels` directly.
    y_pred_probs_df = query_adata_pred.obs[pred_prob_cols].copy()

# Use the helper function for metrics
metrics_results = _analysis_utils.evaluate_and_report_metrics(
    true_labels=true_labels,
    predicted_labels=predicted_labels,
    classifier_classes=scpred_model.classifier_.classes_, # Pass classifier classes for robust reporting
    y_pred_probs=y_pred_probs_df
)


# --- 6. Visualizing Results (plotting code directly in notebook) ---
print("\n--- Step 6: Visualizing Results ---")

# Ensure UMAP embedding is computed on the projected PCs if not already present
if 'X_scpred_pca' in query_adata_pred.obsm and 'X_umap' not in query_adata_pred.obsm:
    print("Computing neighbors and UMAP based on X_scpred_pca for visualization...")
    sc.pp.neighbors(query_adata_pred, n_neighbors=10, use_rep='X_scpred_pca', random_state=RANDOM_STATE) # Added random_state
    sc.tl.umap(query_adata_pred, random_state=RANDOM_STATE) # Added random_state
    print(f"Query data .obsm keys after UMAP: {list(query_adata_pred.obsm.keys())}")
elif 'X_umap' not in query_adata_pred.obsm:
    print("X_scpred_pca or X_umap not found in .obsm. UMAP plots might not be generated.")


# Plot 1: Confusion Matrix
print("\n--- Confusion Matrix ---")
# Ensure labels are string type consistently, including "unassigned" if present
true_labels_str = query_adata_pred.obs['cell_type'].astype(str)
predicted_labels_str = query_adata_pred.obs['scpred_prediction'].astype(str)

# FIX: Ensure all unique labels (true and predicted, including 'unassigned') are used for CM labels
cm_labels_str = sorted(list(set(true_labels_str.unique()) | set(predicted_labels_str.unique())))
cm = confusion_matrix(true_labels_str, predicted_labels_str, labels=cm_labels_str)
cm_df = pd.DataFrame(cm, index=cm_labels_str, columns=cm_labels_str)

plt.figure(figsize=(10, 8)) # Slightly larger figure for clarity with potentially more labels
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Number of Cells'})
plt.title('Confusion Matrix (True vs. Predicted Labels, incl. Unassigned)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Plot 2: Per-Class ROC AUC (One-vs-Rest)
print("\n--- Per-Class ROC AUC (One-vs-Rest) ---")
# ROC AUC is only meaningful if classifier.predict_proba is available
if hasattr(scpred_model.classifier_, 'predict_proba'):
    # We use the filtered true_labels and predicted_labels from evaluation_metrics for ROC,
    # as ROC is not typically defined for "unassigned" as a class itself.
    # The `_analysis_utils.evaluate_and_report_metrics` already handles this internally for its printout,
    # but for manual plotting here, we need to ensure labels are consistent.
    
    # Filter out unassigned cells from prediction results for ROC plotting
    non_unassigned_mask = (query_adata_pred.obs['scpred_prediction'].astype(str) != "unassigned")
    true_labels_for_roc = query_adata_pred.obs.loc[non_unassigned_mask, 'cell_type'].astype(str)
    
    # Ensure y_pred_probs_df is also filtered to match
    y_pred_probs_for_roc = query_adata_pred.obs.loc[non_unassigned_mask, pred_prob_cols].copy()

    if len(true_labels_for_roc) > 0 and not y_pred_probs_for_roc.empty:
        classes = scpred_model.classifier_.classes_
        
        # Create binary labels for each class (one-hot encoding style)
        # Use LabelBinarizer to ensure correct order and handling of classes
        label_binarizer = LabelBinarizer()
        label_binarizer.fit([str(c) for c in classes]) # Fit on all possible classes from classifier
        y_true_binary = label_binarizer.transform(true_labels_for_roc)
        
        y_scores = y_pred_probs_for_roc[[f"scpred_prob_{str(c)}" for c in classes]].values # Use the df we already prepared

        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(classes):
            class_label_str = str(class_label)
            # Check if class has both 0 and 1 in true labels for ROC calculation
            # and if the corresponding probability column exists.
            if y_true_binary.shape[1] > i and len(np.unique(y_true_binary[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {class_label_str} (AUC = {roc_auc:.2f})')
            else:
                print(f"Skipping ROC plot for Class {class_label_str}: Not enough unique true labels in the assigned set.")

        plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - One-vs-Rest (Excluding Unassigned)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        print("Not enough assigned cells with probabilities to compute ROC AUC for any class.")
else:
    print("Classifier does not support `predict_proba`, cannot compute ROC AUC.")


# Plot 3: Distribution of Prediction Probabilities by True Class
print("\n--- Distribution of Prediction Probabilities by True Class ---")
if hasattr(scpred_model.classifier_, 'predict_proba'):
    # Use the probabilities for all cells (including those that might become unassigned)
    prob_df = query_adata_pred.obs[[f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_]].copy()
    prob_df['true_cell_type'] = query_adata_pred.obs['cell_type'].astype(str)
    
    # Add a column for the actual predicted class (which might be 'unassigned')
    prob_df['predicted_cell_type'] = query_adata_pred.obs['scpred_prediction'].astype(str)

    # Melt the DataFrame for easier plotting with seaborn
    prob_melted = prob_df.melt(
        id_vars=['true_cell_type', 'predicted_cell_type'], # Include predicted_cell_type
        value_vars=[f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_],
        var_name='probability_of_class', # Renamed to avoid confusion
        value_name='probability'
    )
    prob_melted['probability_of_class'] = prob_melted['probability_of_class'].str.replace('scpred_prob_', '')

    # Filter to only show probabilities of the class that was actually predicted for assigned cells,
    # or the max probability if it led to 'unassigned'.
    # For this plot, we're interested in the distribution of the *highest* probability
    # per cell, or probabilities *towards specific classes*.
    # Let's show the probability distribution for each true cell type towards *all* predicted classes,
    # and also a separate plot for the confidence score (max prob) itself.

    # Option A: Distribution of probabilities for *all* classes, grouped by true type
    for true_type in sorted(prob_melted['true_cell_type'].unique()):
        plt.figure(figsize=(12, 6)) # Increased figure size
        subset_df = prob_melted[prob_melted['true_cell_type'] == true_type]
        sns.boxplot(
            data=subset_df,
            x='probability_of_class',
            y='probability',
            hue='probability_of_class',
            palette='viridis',
            legend=False
        )
        plt.title(f'Prediction Probabilities for True Cell Type: {true_type}')
        plt.xlabel('Probability of being Predicted as Class')
        plt.ylabel('Probability')
        plt.ylim(-0.05, 1.05) # Consistent y-axis limits
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # For PBMC3K, labels are usually numbers, so rotation might not be needed.
        # plt.xticks(rotation=45, ha='right') # Re-enable if needed for overlap
        plt.show()

    # Option B: Distribution of the *max* prediction probability for each cell
    # This shows the confidence of the assigned or unassigned label
    # Filter out NaNs that might occur if prob_df was somehow empty
    max_probs_df = prob_df.loc[prob_df.index.intersection(query_adata_pred.obs.index)][[f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_]].max(axis=1).to_frame(name='max_probability')
    max_probs_df['true_cell_type'] = query_adata_pred.obs['cell_type'].astype(str)
    max_probs_df['predicted_label'] = query_adata_pred.obs['scpred_prediction'].astype(str)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=max_probs_df,
        x='true_cell_type',
        y='max_probability',
        hue='true_cell_type', # Hue by true cell type
        palette='viridis',
        legend=False
    )
    plt.axhline(y=PREDICTION_THRESHOLD, color='r', linestyle='--', label=f'Unassigned Threshold ({PREDICTION_THRESHOLD})')
    plt.title('Max Prediction Probability per True Cell Type (Confidence)')
    plt.xlabel('True Cell Type')
    plt.ylabel('Max Predicted Probability')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    # Boxplot of max_probability, grouped by *predicted* label (including unassigned)
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=max_probs_df,
        x='predicted_label',
        y='max_probability',
        hue='predicted_label', # Hue by predicted label
        palette='magma',
        legend=False
    )
    plt.axhline(y=PREDICTION_THRESHOLD, color='r', linestyle='--', label=f'Unassigned Threshold ({PREDICTION_THRESHOLD})')
    plt.title('Max Prediction Probability per Predicted Label (Confidence)')
    plt.xlabel('Predicted Label (including Unassigned)')
    plt.ylabel('Max Predicted Probability')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


else:
    print("Classifier does not support `predict_proba`, cannot visualize probability distributions.")


# Plot 4: True vs Predicted Labels UMAP
if 'X_umap' in query_adata_pred.obsm:
    print("\n--- Plotting UMAP: True vs Predicted Labels (incl. Unassigned) ---")
    plt.figure(figsize=(12, 6)) # Use figsize here to control the overall figure size
    
    sc.pl.umap(
        query_adata_pred,
        color=['cell_type', 'scpred_prediction'],
        title=['True Labels (Louvain Clusters)', 'scPred Predictions (incl. Unassigned)'],
        frameon=False,
        show=False, # Set show=False to control display manually
        ncols=2,
        wspace=0.3 # Adjust horizontal space between subplots
    )
    plt.suptitle('UMAP of Query Data: True vs Predicted Labels (PBMC3K)', y=1.02, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
else:
    print("Skipping UMAP: True vs Predicted Labels as 'X_umap' is not available in .obsm.")


# Plot 5: Misclassified Cells UMAP (with 'unassigned' considered as 'not correct')
if 'X_umap' in query_adata_pred.obsm:
    print("\n--- Plotting Misclassified Cells UMAP ---")
    # true_labels_str and predicted_labels_str are already available (as they are in notebook scope)
    true_labels_str = query_adata_pred.obs['cell_type'].astype(str)
    predicted_labels_str = query_adata_pred.obs['scpred_prediction'].astype(str)

    # A cell is "misclassified" if its true label is not equal to its predicted label.
    # This will correctly mark cells predicted as "unassigned" as 'True' for misclassified,
    # which is often desired for this type of visualization.
    query_adata_pred.obs['misclassified'] = (true_labels_str != predicted_labels_str).astype(str).astype('category')
    
    # Custom palette to highlight misclassified/unassigned vs correctly classified
    misclassified_palette = {'True': 'red', 'False': 'lightgray'}
    
    plt.figure(figsize=(8, 6))
    sc.pl.umap(
        query_adata_pred,
        color='misclassified',
        palette=misclassified_palette,
        size=50,
        alpha=0.7,
        title='UMAP of Query Data: Misclassified/Unassigned Cells',
        frameon=False,
        show=False,
        legend_loc='upper right' # Changed legend location
    )
    plt.show()
else:
    print("Skipping Misclassified Cells UMAP as 'X_umap' is not available in .obsm.")


# Plot 6: UMAP colored by Prediction Confidence (Max Probability of the Assigned Class)
if 'X_umap' in query_adata_pred.obsm and hasattr(scpred_model.classifier_, 'predict_proba'):
    print("\n--- Plotting UMAP by Prediction Confidence ---")
    # Get the probability for the *predicted* class for each cell
    # If the cell is 'unassigned', its predicted_prob_score will be NaN,
    # which Scanpy usually handles by making those points invisible or a default color.
    def get_predicted_prob(row, classes):
        pred_class = row['scpred_prediction']
        if pred_class == "unassigned": # Handle unassigned explicitly
            return np.nan
        prob_col_name = f'scpred_prob_{pred_class}'
        if prob_col_name in row.index:
            return row[prob_col_name]
        return np.nan

    prob_cols = [f"scpred_prob_{c}" for c in scpred_model.classifier_.classes_]
    
    if not all(col in query_adata_pred.obs.columns for col in prob_cols):
        print("Skipping UMAP by prediction confidence: Required probability columns not found in .obs.")
    else:
        temp_obs_for_apply = query_adata_pred.obs[['scpred_prediction'] + prob_cols]
        query_adata_pred.obs['predicted_prob_score'] = temp_obs_for_apply.apply(
            lambda row: get_predicted_prob(row, scpred_model.classifier_.classes_), axis=1
        )

        plt.figure(figsize=(8, 6))
        sc.pl.umap(
            query_adata_pred,
            color='predicted_prob_score',
            cmap='viridis',
            title='UMAP of Query Data: Prediction Confidence (Assigned Cells)',
            frameon=False,
            vmin=0.0, vmax=1.0, # Consistent color bar limits
            show=False
        )
        plt.show()
else:
    print("Skipping UMAP by prediction confidence: 'X_umap' not available or classifier does not support `predict_proba`.")

print("\n--- Analysis Complete ---")





# _core.py:

import anndata as ad
import pandas as pd
import scanpy as sc
from . import _utils, _preprocessing, _training, _prediction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # Import SVC for kernel flexibility
import pickle # For saving/loading the model


class ScPredModel:
    """
    A class to encapsulate the scPred workflow.
    This version handles preprocessing internally (Strategy A) but offers an option
    to skip it for pre-processed inputs.
    """
    def __init__(self):
        self.pca_model_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.reference_hvg_genes_ = None # Stores the HVGs identified during reference preprocessing
        self.ref_adata_processed_ = None # New: Stores the preprocessed reference adata (HVG-selected)

    def train(self, ref_adata, cell_type_key, n_components=30, 
              hvg_min_mean=0.0125, hvg_max_mean=3, hvg_min_disp=0.5, hvg_n_top_genes=None, hvg_flavor='seurat',
              svm_kernel='linear', svm_c=1.0, svm_random_state=42, class_weight=None,
              perform_preprocessing=True):
        """
        Trains the scPred model on reference data.
        Can perform all preprocessing internally or skip it if data is already prepared.

        Args:
            ref_adata (ad.AnnData): Reference AnnData object. Expected raw counts if perform_preprocessing=True,
                                    else expected to be normalized, log1p, and HVG-selected.
            cell_type_key (str): Key in `ref_adata.obs` for cell type labels.
            n_components (int): Number of PCs to compute.
            hvg_min_mean (float): Min mean for HVG selection.
            hvg_max_mean (float): Max mean for HVG selection.
            hvg_min_disp (float): Min dispersion for HVG selection.
            hvg_n_top_genes (int, optional): Number of top HVGs to select.
            hvg_flavor (str): HVG selection method ('seurat', 'cell_ranger', 'seurat_v3').
            svm_kernel (str): SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid').
            svm_c (float): Regularization parameter for SVM.
            svm_random_state (int): Random state for SVM for reproducibility.
            perform_preprocessing (bool): If True, model performs initial filtering, normalization,
                                          log1p, and HVG selection. If False, assumes ref_adata
                                          is already normalized, log1p-transformed, and HVG-selected.
                                          Scaling and PCA are *always* performed internally.
        """
        _utils.check_adata(ref_adata)
        if cell_type_key not in ref_adata.obs:
            raise ValueError(f"'{cell_type_key}' not found in ref_adata.obs.")

        print("--- Starting ScPred Training ---")

        if perform_preprocessing:
            print("  `perform_preprocessing` is True. Performing full internal preprocessing of reference data...")
            # 1. Initial Preprocessing (Filter, Normalize, Log1p)
            processed_ref_adata = _preprocessing.initial_preprocessing_steps(ref_adata.copy())
            
            # 2. Select Highly Variable Genes (HVGs)
            self.ref_adata_processed_, self.reference_hvg_genes_ = _preprocessing.select_highly_variable_genes(
                processed_ref_adata, # Use the initially processed data
                min_mean=hvg_min_mean, max_mean=hvg_max_mean, min_disp=hvg_min_disp, 
                n_top_genes=hvg_n_top_genes, flavor=hvg_flavor
            )
        else:
            print("  `perform_preprocessing` is False. Skipping initial filtering, normalization, log1p, and HVG selection.")
            print("  Assuming reference data is already normalized, log1p-transformed, and HVG-selected.")
            self.ref_adata_processed_ = ref_adata.copy() # Use provided ref_adata as is
            self.reference_hvg_genes_ = self.ref_adata_processed_.var_names.tolist() # Assume input is already subsetted to relevant HVGs
            
            # Safety check: If not performing preprocessing, ensure a 'log1p' flag exists for consistency
            if 'log1p' not in self.ref_adata_processed_.uns:
                print("  Warning: `perform_preprocessing` is False, but `adata.uns['log1p']` not found in reference data. "
                      "  Ensure data is indeed log-transformed if required by your pipeline.")


        # 3. Fit StandardScaler on the (preprocessed or pre-provided) HVG-selected data
        self.scaler_ = _preprocessing.get_fitted_scaler(self.ref_adata_processed_)
        
        # 4. Scale the HVG-selected data
        X_scaled_ref = _preprocessing.transform_data_with_scaler(self.ref_adata_processed_, self.scaler_)

        # 5. Perform PCA on the scaled data
        self.pca_model_, X_pca = _preprocessing.get_fitted_pca(X_scaled_ref, n_components)
        
        # 6. Train Classifier
        labels = self.ref_adata_processed_.obs[cell_type_key]
        self.classifier_ = _training.train_svm(
            X_pca, labels, 
            kernel=svm_kernel, c=svm_c, random_state=svm_random_state,
            class_weight=class_weight # FIX: Add class_weight='balanced' here
        )
        print("--- ScPred Training Complete ---")

    def predict(self, query_adata, threshold=0.0, perform_preprocessing=True):
        """
        Predicts cell types on query data.
        Can perform preprocessing and projection consistently with training, or skip initial steps.

        Args:
            query_adata (ad.AnnData): Query AnnData object. Expected raw counts if perform_preprocessing=True,
                                      else expected to be normalized, log1p, and aligned to reference HVGs.
            threshold (float): Minimum probability for a prediction.
                               Predictions below this will be flagged as "unassigned".
            perform_preprocessing (bool): If True, model performs initial filtering, normalization,
                                          log1p, and gene alignment to reference HVGs. If False,
                                          assumes query_adata is already normalized, log1p-transformed,
                                          and aligned to reference HVGs.
                                          Scaling and PCA transformation are *always* performed internally.
        Returns:
            ad.AnnData: Query AnnData object with prediction results added.
        """
        if self.pca_model_ is None or self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("Model must be trained before prediction.")

        _utils.check_adata(query_adata)
        print("--- Starting ScPred Prediction ---")

        if perform_preprocessing:
            print("  `perform_preprocessing` is True. Performing full internal preprocessing of query data...")
            # 1. Initial Preprocessing (Filter, Normalize, Log1p)
            processed_query_adata = _preprocessing.initial_preprocessing_steps(query_adata.copy())

            # 2. Align query genes to reference HVGs (learned during training)
            aligned_query_adata = _preprocessing.align_genes_to_reference(
                processed_query_adata, self.reference_hvg_genes_
            )
        else:
            print("  `perform_preprocessing` is False. Skipping initial filtering, normalization, log1p, and gene alignment.")
            print("  Assuming query data is already normalized, log1p-transformed, and aligned to reference HVGs.")
            aligned_query_adata = query_adata.copy() # Use provided query_adata as is
            
            # Additional check: If not performing preprocessing, ensure a 'log1p' flag exists for consistency
            if 'log1p' not in aligned_query_adata.uns:
                print("  Warning: `perform_preprocessing` is False, but `adata.uns['log1p']` not found in query data. "
                      "  Ensure data is indeed log-transformed if required by your pipeline.")
            # The gene count mismatch warning is already there, which is good:
            if aligned_query_adata.shape[1] != len(self.reference_hvg_genes_):
                print("  Warning: `perform_preprocessing` is False, but query_adata gene count doesn't match reference HVGs. "
                      "  Ensure query_adata is aligned to reference HVGs before passing.")

        # 3. Scale query data using the *fitted reference scaler*
        X_scaled_query = _preprocessing.transform_data_with_scaler(aligned_query_adata, self.scaler_)

        # 4. Project scaled query data onto the *existing PCA space*
        X_projected = _preprocessing.transform_data_with_pca(X_scaled_query, self.pca_model_)

        # 5. Predict cell types and probabilities, applying threshold
        labels, probs = _prediction.predict_cells(X_projected, self.classifier_, threshold=threshold) # FIX: Pass threshold

        # 6. Add results back to the original query_adata object
        query_adata.obs['scpred_prediction'] = pd.Series(
            labels, index=aligned_query_adata.obs_names # Labels from aligned adata cells
        ).reindex(query_adata.obs_names).astype('category') # Reindex to original query_adata for consistency
        
        # Add probability columns
        if probs is not None:
            prob_df_for_reindex = pd.DataFrame(
                probs.values, index=aligned_query_adata.obs_names, columns=self.classifier_.classes_
            )
            for col in prob_df_for_reindex.columns:
                query_adata.obs[f"scpred_prob_{col}"] = prob_df_for_reindex[col].reindex(query_adata.obs_names)

        # Store the projected PCs in .obsm.
        if X_projected.shape[0] == aligned_query_adata.shape[0]:
            pca_df_for_reindex = pd.DataFrame(X_projected, index=aligned_query_adata.obs_names)
            query_adata.obsm['X_scpred_pca'] = pca_df_for_reindex.reindex(query_adata.obs_names).values
        else:
            print("  Warning: Number of projected cells does not match aligned query data. X_scpred_pca not stored in .obsm.")

        print("--- ScPred Prediction Complete ---")
        return query_adata

    def save(self, filepath):
        """
        Saves the trained ScPredModel instance to a file using pickle.
        
        Args:
            filepath (str): The path to the file where the model should be saved.
        """
        print(f"Saving ScPredModel to {filepath}...")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print("ScPredModel saved successfully.")
        except Exception as e:
            print(f"Error saving ScPredModel: {e}")

    @staticmethod
    def load(filepath):
        """
        Loads a trained ScPredModel instance from a file using pickle.
        
        Args:
            filepath (str): The path to the file from which the model should be loaded.
            
        Returns:
            ScPredModel: The loaded ScPredModel instance.
        """
        print(f"Loading ScPredModel from {filepath}...")
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print("ScPredModel loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading ScPredModel: {e}")
            return None



# _prediction.py

import pandas as pd
import numpy as np

def predict_cells(X_projected_pca, classifier, threshold=0.0):
    """
    Predicts cell types using the trained classifier, with an option for probability thresholding.

    Args:
        X_projected_pca (np.ndarray): Projected PCA data for query cells.
        classifier (sklearn.base.BaseEstimator): The trained classifier.
        threshold (float): Minimum probability for a prediction.
                           Predictions below this will be flagged as "unassigned".

    Returns:
        tuple: (np.ndarray, pd.DataFrame) - Predicted labels (as numpy array, possibly with "unassigned")
                                           and prediction probabilities (DataFrame).
    """
    print("Predicting cell types...")
    
    # Get raw predictions and probabilities
    predicted_labels_raw = classifier.predict(X_projected_pca)
    
    prob_df = None
    try:
        predicted_probs_array = classifier.predict_proba(X_projected_pca)
        # Create a DataFrame for probabilities with class names
        prob_df = pd.DataFrame(predicted_probs_array, columns=classifier.classes_)
    except AttributeError:
        print("Classifier does not support predict_proba. Probability thresholding will not be applied.")
        # If no probabilities, thresholding cannot be applied.
        threshold = 0.0 # Effectively disable thresholding

    # Apply probability thresholding
    final_predicted_labels = predicted_labels_raw.copy().astype(object) # Use object dtype to allow mixed types (str and original labels)
    
    if threshold > 0.0 and prob_df is not None:
        # For each cell, find the maximum probability and its corresponding predicted class
        max_probs = prob_df.max(axis=1)
        
        # Identify cells where max probability is below threshold
        low_confidence_mask = (max_probs < threshold)
        
        # Set predictions to "unassigned" for low-confidence cells
        final_predicted_labels[low_confidence_mask] = "unassigned"
        print(f"Applied probability thresholding: {np.sum(low_confidence_mask)} cells set to 'unassigned' (threshold={threshold}).")
    elif threshold > 0.0 and prob_df is None:
        print("Warning: Probability thresholding requested, but classifier does not support `predict_proba`. Thresholding skipped.")

    print("Prediction finished.")
    return final_predicted_labels, prob_df



# _analysis_utils.py

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
                                               Must have the same index as true_labels and predicted_labels initially.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    print("\n--- Evaluating Predictions ---")

    # Ensure labels are string type and are Pandas Series for consistent indexing operations
    # This also handles cases where inputs might be NumPy arrays, converting them to Series
    true_labels_str = pd.Series(true_labels, dtype=str)
    predicted_labels_str = pd.Series(predicted_labels, dtype=str)

    # If y_pred_probs is provided, ensure its index matches that of predicted_labels_str
    if y_pred_probs is not None:
        if not y_pred_probs.index.equals(predicted_labels_str.index):
            print("Warning: y_pred_probs index does not match predicted_labels index. Reindexing y_pred_probs to match.")
            # Reindex y_pred_probs to match the index of predicted_labels_str.
            # This is crucial for consistent slicing with boolean masks later.
            y_pred_probs = y_pred_probs.reindex(predicted_labels_str.index)
            # Fill NaNs that might result from reindexing if some cells are missing probabilities
            y_pred_probs = y_pred_probs.fillna(0.0) # Using 0.0 for probabilities is a safe default

    # Mask for cells that are NOT "unassigned" from the *predicted* labels
    # This mask will have the same index as predicted_labels_str
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
        
        # Use the mask to filter the probability DataFrame.
        # Since y_pred_probs has been reindexed to match predicted_labels_str,
        # this slicing will ensure index and length consistency.
        y_scores_filtered = y_pred_probs.loc[non_unassigned_mask, :].copy()

        # Convert true_labels_filtered to binary for ROC calculation
        label_binarizer = LabelBinarizer()
        # Fit on classifier_classes as these are the "true" classes the classifier knows about
        # Convert classifier_classes to string to match labels
        label_binarizer.fit([str(c) for c in classifier_classes]) 
        y_true_binary = label_binarizer.transform(true_labels_filtered)

        # Check if y_true_binary is empty or has only one class after filtering
        if y_true_binary.size == 0 or np.all(y_true_binary == y_true_binary[0, :]):
            print("Not enough assigned cells with diverse true labels to compute ROC AUC for any class.")
            metrics_results['roc_auc_scores'] = {} # No ROC AUC scores
            return metrics_results # Exit early from function if no valid data for ROC

        for i, class_label in enumerate(classifier_classes):
            class_label_str = str(class_label)
            prob_col_name = f"scpred_prob_{class_label_str}"

            if prob_col_name in y_scores_filtered.columns:
                # Check if there's at least one positive and one negative sample for ROC AUC
                # within the filtered set of labels for the current class
                if y_true_binary.shape[1] > i and len(np.unique(y_true_binary[:, i])) > 1:
                    # Ensure y_true_binary column for this class has data and y_scores_filtered has data
                    if y_true_binary[:, i].size > 0 and y_scores_filtered[prob_col_name].size > 0:
                        fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_scores_filtered[prob_col_name])
                        roc_auc = auc(fpr, tpr)
                        roc_auc_scores[class_label_str] = roc_auc
                    else:
                        roc_auc_scores[class_label_str] = np.nan # No data to compute ROC AUC
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
