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