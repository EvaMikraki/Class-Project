# scpred_py/_preprocessing.py

import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def standard_preprocess(adata):
    """Applies standard scRNA-seq preprocessing."""
    print("Applying standard preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    # Note: We will now handle scaling right before PCA for consistency.
    # sc.pp.scale(adata, max_value=10) <-- This is no longer needed here.
    print("Preprocessing finished.")
    return adata


def get_pca(adata, n_components=30):
    """
    Performs scaling and PCA on the reference data.

    Args:
        adata (ad.AnnData): AnnData object (cells x genes).
        n_components (int): Number of principal components.

    Returns:
        tuple: (pca_model, scaler_model, transformed_pca)
               - The fitted PCA model from scikit-learn.
               - The fitted StandardScaler model from scikit-learn.
               - The transformed data (PCs).
    """
    print(f"Performing scaling and PCA with {n_components} components...")
    
    # Ensure data is dense
    X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # 1. Scale data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # 2. Perform PCA on the scaled data
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print("Scaling and PCA finished.")
    return pca, scaler, X_pca


def project_pca(adata, pca_model, scaler_model):
    """
    Projects query data onto an existing PCA space using a saved scaler.

    Args:
        adata (ad.AnnData): Query AnnData object (cells x genes).
        pca_model (sklearn.decomposition.PCA): The *fitted* PCA model.
        scaler_model (sklearn.preprocessing.StandardScaler): The *fitted* scaler.

    Returns:
        np.ndarray: The projected data (PCs for query cells).
    """
    print("Scaling query data and projecting onto existing PCA space...")
    
    # Ensure data is dense
    X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # 1. Scale query data using the *same* scaler from the reference data
    X_scaled = scaler_model.transform(X_data)
    
    # 2. Project scaled data onto the existing PCA space
    X_projected = pca_model.transform(X_scaled)
    
    print("Projection finished.")
    return X_projected


def select_informative_pcs(X_pca, labels, p_threshold=0.05):
    """
    Selects informative PCs using the Wilcoxon rank-sum test.

    Args:
        X_pca (np.ndarray): PCA-transformed data (cells x PCs).
        labels (pd.Series or np.ndarray): Cell type labels for each cell.
        p_threshold (float): The significance threshold to use after multiple
                             test correction.

    Returns:
        np.ndarray: An array of indices for the informative PCs.
    """
    print("Selecting informative PCs using Wilcoxon rank-sum test...")
    
    unique_labels = np.unique(labels)
    n_pcs = X_pca.shape[1]
    
    p_values = []
    
    # Perform test for each PC against each cell type
    for i in range(n_pcs):
        pc_scores = X_pca[:, i]
        for label in unique_labels:
            group1 = pc_scores[labels == label]
            group2 = pc_scores[labels != label]
            
            # The test requires at least one observation in each group
            if len(group1) == 0 or len(group2) == 0:
                p_val = 1.0
            else:
                _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            p_values.append(p_val)
            
    # Correct for multiple testing
    # We tested n_pcs * n_labels times
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=p_threshold, method='fdr_bh')
    
    # Reshape the rejection array to match our PC-vs-label structure
    reject_matrix = reject.reshape((n_pcs, len(unique_labels)))
    
    # A PC is informative if it's significant for *any* cell type
    is_informative = reject_matrix.any(axis=1)
    informative_indices = np.where(is_informative)[0]
    
    if len(informative_indices) == 0:
        print(f"Warning: No informative PCs found at p < {p_threshold}. Returning all PCs.")
        return np.arange(n_pcs)
        
    print(f"Found {len(informative_indices)} informative PCs.")
    return informative_indices