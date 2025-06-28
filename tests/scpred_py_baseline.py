# __init__.py
# Makes 'scpred_py' a package and can expose key functions/classes.
# For now, we'll import our main class.

"""
scPred-Py: A Python implementation of the scPred single-cell classification tool.
"""

from ._core import ScPredModel

__version__ = "0.1.0"
__all__ = ["ScPredModel"]

print("scPred-Py initialized. Remember this is a starting point!")

# _core.py
import anndata as ad
import pandas as pd
import scanpy as sc
from . import _utils, _preprocessing, _training, _prediction

class ScPredModel:
    """
    A class to encapsulate the scPred workflow.
    """
    def __init__(self):
        self.pca_model_ = None
        self.classifier_ = None
        self.reference_genes_ = None
        self.reference_adata_ = None # Store for reference

    def train(self, ref_adata, cell_type_key, n_components=30):
        """
        Trains the scPred model on reference data.

        Args:
            ref_adata (ad.AnnData): Reference AnnData object (must be preprocessed).
            cell_type_key (str): Key in `ref_adata.obs` for cell type labels.
            n_components (int): Number of PCs to use.
        """
        _utils.check_adata(ref_adata)
        if cell_type_key not in ref_adata.obs:
            raise ValueError(f"'{cell_type_key}' not found in ref_adata.obs.")

        print("--- Starting ScPred Training ---")
        self.reference_adata_ = ref_adata.copy() # Keep a copy
        
        # 1. PCA
        self.pca_model_, X_pca = _preprocessing.get_pca(self.reference_adata_, n_components)
        self.reference_adata_.obsm['X_scpred_pca'] = X_pca
        self.reference_genes_ = self.reference_adata_.var_names.tolist()

        # 2. Train Classifier
        labels = self.reference_adata_.obs[cell_type_key]
        self.classifier_ = _training.train_svm(X_pca, labels)
        print("--- ScPred Training Complete ---")

    def predict(self, query_adata):
        """
        Predicts cell types on query data.

        Args:
            query_adata (ad.AnnData): Query AnnData object (must be preprocessed).

        Returns:
            ad.AnnData: Query AnnData object with prediction results added.
        """
        if self.pca_model_ is None or self.classifier_ is None:
            raise RuntimeError("Model must be trained before prediction.")

        _utils.check_adata(query_adata)
        
        print("--- Starting ScPred Prediction ---")
        
        # 1. Find common genes and subset *both* datasets
        common_genes = _utils.get_common_genes(self.reference_adata_, query_adata)
        
        # Ensure query genes are in the same order as reference for PCA projection
        query_adata_sub = query_adata[:, common_genes].copy()
        ref_adata_sub = self.reference_adata_[:, common_genes].copy()
        
        # Re-fit PCA on the *subsetted* reference data
        print("Re-fitting PCA on common genes for projection consistency...")
        pca_model_sub, _ = _preprocessing.get_pca(ref_adata_sub, n_components=self.pca_model_.n_components_)
        
        # 2. Project Query Data
        # --- REMOVED sc.pp.scale ---
        # sc.pp.scale(query_adata_sub, max_value=10) # <--- THIS LINE IS REMOVED
        
        X_projected = _preprocessing.project_pca(query_adata_sub, pca_model_sub)

        # 3. Predict
        labels, probs = _prediction.predict_cells(X_projected, self.classifier_)

        # 4. Add results to query_adata
        query_adata.obs['scpred_prediction'] = pd.Categorical(labels, categories=self.classifier_.classes_)
        if probs is not None:
            prob_cols = [f"scpred_prob_{c}" for c in self.classifier_.classes_]
            query_adata.obs[prob_cols] = probs.values
            
        # Add projected PCA to .obsm for potential UMAP visualization
        query_adata.obsm['X_scpred_pca'] = X_projected

        print("--- ScPred Prediction Complete ---")
        return query_adata

# _prediction.py
import pandas as pd

def predict_cells(X_projected_pca, classifier):
    """
    Predicts cell types using the trained classifier.

    Args:
        X_projected_pca (np.ndarray): Projected PCA data for query cells.
        classifier (sklearn.base.BaseEstimator): The trained classifier.

    Returns:
        tuple: (pd.Series, pd.DataFrame) - Predicted labels and prediction
               probabilities.
    """
    print("Predicting cell types...")
    predicted_labels = classifier.predict(X_projected_pca)
    
    try:
        predicted_probs = classifier.predict_proba(X_projected_pca)
        # Create a DataFrame for probabilities with class names
        prob_df = pd.DataFrame(predicted_probs, columns=classifier.classes_)
    except AttributeError:
        print("Classifier does not support predict_proba. Returning only labels.")
        prob_df = None

    print("Prediction finished.")
    return predicted_labels, prob_df

# _preprocessing.py
import scanpy as sc
from sklearn.decomposition import PCA
import numpy as np # <--- ADD THIS LINE

def standard_preprocess(adata):
    """Applies standard scRNA-seq preprocessing."""
    print("Applying standard preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10) # Scaling is important for PCA
    print("Preprocessing finished.")
    return adata

def get_pca(adata, n_components=30):
    """
    Performs PCA on the reference data.

    Args:
        adata (ad.AnnData): Preprocessed AnnData object (cells x genes).
        n_components (int): Number of principal components.

    Returns:
        tuple: (sklearn.decomposition.PCA, np.ndarray) - The fitted PCA model
               and the transformed data (PCs).
    """
    print(f"Performing PCA with {n_components} components...")
    pca_model = PCA(n_components=n_components, random_state=42)
    
    # Ensure data is dense for scikit-learn PCA
    X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    transformed_pca = pca_model.fit_transform(X_data)
    print("PCA finished.")
    return pca_model, transformed_pca

def project_pca(adata, pca_model):
    """
    Projects query data onto an existing PCA space.

    Args:
        adata (ad.AnnData): Preprocessed query AnnData object (cells x genes).
                           Must have the *same genes* in the *same order*
                           as the reference data used for PCA.
        pca_model (sklearn.decomposition.PCA): The *fitted* PCA model from
                                               the reference data.

    Returns:
        np.ndarray: The projected data (PCs for query cells).
    """
    print("Projecting query data onto existing PCA space...")
    
    # Ensure data is dense
    X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # --- FIX: Handle potential NaNs ---
    if np.isnan(X_data).any():
        print("Warning: NaNs found in input data for PCA transform. Replacing with 0.")
        X_data = np.nan_to_num(X_data, copy=True, nan=0.0, posinf=0.0, neginf=0.0) # Replace NaN/inf with 0
    # -----------------------------------------------

    # Important: We ONLY use transform, not fit_transform!
    projected_data = pca_model.transform(X_data)
    print("Projection finished.")
    return projected_data

# _training.py
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def train_svm(X_pca, labels):
    """
    Trains a One-vs-Rest SVM classifier on PCA components.

    Args:
        X_pca (np.ndarray): PCA-transformed data (cells x PCs).
        labels (pd.Series or np.ndarray): Cell type labels for each cell.

    Returns:
        sklearn.multiclass.OneVsRestClassifier: The trained classifier.
    """
    print("Training One-vs-Rest SVM classifier...")

    # A simple linear SVM - scPred often uses this.
    # We use probability=True for prediction probabilities later.
    # C=1 is a default, scPred does hyperparameter tuning - A key area for expansion!
    svm = SVC(kernel='linear', probability=True, C=1.0, random_state=42)

    # Use One-vs-Rest strategy for multi-class problems
    clf = OneVsRestClassifier(svm, n_jobs=-1) # Use all available cores

    # --- Optional: Add Cross-Validation (Good Practice) ---
    # cv = StratifiedKFold(n_splits=5)
    # scores = cross_val_score(clf, X_pca, labels, cv=cv, scoring='accuracy')
    # print(f"Cross-validation accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    # --------------------------------------------------------

    # Train the final model on all data
    clf.fit(X_pca, labels)
    print("Training finished.")

    return clf

# _utils.py
import scanpy as sc
import anndata as ad

def check_adata(adata):
    """Checks if the input is an AnnData object."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input must be an AnnData object.")
    print("AnnData object check passed.")

def get_common_genes(ref_adata, query_adata):
    """Finds common genes between reference and query datasets."""
    check_adata(ref_adata)
    check_adata(query_adata)

    ref_genes = ref_adata.var_names
    query_genes = query_adata.var_names

    common_genes = list(set(ref_genes) & set(query_genes))

    if len(common_genes) == 0:
        raise ValueError("No common genes found between reference and query data.")

    print(f"Found {len(common_genes)} common genes.")
    return common_genes

# pbmc3k notebook

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
if not os.path.exists('../data/processed'):
    os.makedirs('../data/processed')

ref_adata.write('../data/processed/pbmc3k_ref.h5ad')
query_adata.write('../data/processed/pbmc3k_query.h5ad')

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
ref_adata = ad.read_h5ad('../data/processed/pbmc3k_ref.h5ad')
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

with open('../models/scpred_pbmc3k_model.pkl', 'wb') as f:
    pickle.dump(scpred_model, f)

print("Trained model saved to ../models/scpred_pbmc3k_model.pkl")

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
with open('../models/scpred_pbmc3k_model.pkl', 'rb') as f:
    scpred_model = pickle.load(f)

print("Loaded Model:", scpred_model)

# Load the query data
query_adata = ad.read_h5ad('../data/processed/pbmc3k_query.h5ad')
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