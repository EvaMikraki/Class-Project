# scpred_py/_preprocessing.py

import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import anndata as ad
import pandas as pd # Needed for robust gene alignment via DataFrame reindexing


def initial_preprocessing_steps(adata):
    """
    Applies initial scRNA-seq preprocessing: filtering, normalization, log1p.
    Does NOT perform HVG selection or scaling here.
    """
    print("Applying initial preprocessing (filter, normalize, log1p)...")
    # Work on a copy to avoid modifying the original AnnData object passed in
    adata_copy = adata.copy()

    # Basic filtering steps
    sc.pp.filter_cells(adata_copy, min_genes=200)
    sc.pp.filter_genes(adata_copy, min_cells=3)

    # Normalization to a total count per cell
    sc.pp.normalize_total(adata_copy, target_sum=1e4)

    # Log-transformation
    sc.pp.log1p(adata_copy)

    print("Initial preprocessing finished.")
    return adata_copy


def select_highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=None, flavor='seurat'):
    """
    Identifies and subsets AnnData object to highly variable genes.
    
    Args:
        adata (ad.AnnData): AnnData object after initial preprocessing (normalized, log1p).
        min_mean (float): Minimum mean expression.
        max_mean (float): Maximum mean expression.
        min_disp (float): Minimum dispersion.
        n_top_genes (int, optional): Number of top highly variable genes to select. If None,
                                     selection is based on min/max_disp/mean.
        flavor (str): HVG selection method ('seurat', 'cell_ranger', 'seurat_v3').
                      'seurat' is a good robust choice.
                      'seurat_v3' has known environment dependency issues (skmisc.loess).

    Returns:
        tuple: (ad.AnnData, list) - Subsetted AnnData object containing only HVGs,
                                   and a list of the names of these HVGs.
    """
    print(f"Selecting highly variable genes with flavor='{flavor}'...")
    adata_hvg_copy = adata.copy() # Work on a copy
    
    # Perform HVG computation. This adds 'highly_variable' to adata_hvg_copy.var
    sc.pp.highly_variable_genes(
        adata_hvg_copy,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        n_top_genes=n_top_genes,
        flavor=flavor
    )
    
    # Subset the AnnData object to only include the highly variable genes
    adata_hvg_subset = adata_hvg_copy[:, adata_hvg_copy.var.highly_variable].copy()
    
    hvg_gene_names = adata_hvg_subset.var_names.tolist()
    print(f"Selected {len(hvg_gene_names)} highly variable genes.")
    
    return adata_hvg_subset, hvg_gene_names


def get_fitted_scaler(adata_hvg_subset):
    """
    Fits a StandardScaler on the given AnnData object's expression data.
    Expected input is already normalized, log1p-transformed, and HVG-selected.

    Args:
        adata_hvg_subset (ad.AnnData): AnnData object with only HVGs.

    Returns:
        sklearn.preprocessing.StandardScaler: The fitted StandardScaler model.
    """
    print("Fitting StandardScaler on highly variable genes...")
    X_data = adata_hvg_subset.X.toarray() if hasattr(adata_hvg_subset.X, 'toarray') else adata_hvg_subset.X
    scaler = StandardScaler()
    scaler.fit(X_data)
    print("StandardScaler fitted.")
    return scaler


def transform_data_with_scaler(adata_subset, scaler_model):
    """
    Transforms AnnData object's expression data using a fitted StandardScaler.

    Args:
        adata_subset (ad.AnnData): AnnData object (cells x genes), where genes
                                   match those used to fit the scaler.
        scaler_model (sklearn.preprocessing.StandardScaler): The fitted StandardScaler.

    Returns:
        np.ndarray: The scaled data (NumPy array).
    """
    print("Transforming data with fitted StandardScaler...")
    X_data = adata_subset.X.toarray() if hasattr(adata_subset.X, 'toarray') else adata_subset.X
    
    # Handle potential NaNs before transform, as scaling can be sensitive
    if np.isnan(X_data).any():
        print("Warning: NaNs found in input data for scaler transform. Replacing with 0.")
        X_data = np.nan_to_num(X_data, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    X_transformed = scaler_model.transform(X_data)
    print("Data scaled.")
    return X_transformed


def get_fitted_pca(X_scaled_data, n_components=30):
    """
    Performs PCA on the scaled data.

    Args:
        X_scaled_data (np.ndarray): Scaled expression data (cells x genes).
        n_components (int): Number of principal components.

    Returns:
        tuple: (sklearn.decomposition.PCA, np.ndarray) - The fitted PCA model
               and the transformed data (PCs).
    """
    print(f"Performing PCA with {n_components} components...")
    pca_model = PCA(n_components=n_components, random_state=42)
    
    transformed_pca = pca_model.fit_transform(X_scaled_data)
    print("PCA finished.")
    return pca_model, transformed_pca


def transform_data_with_pca(X_scaled_data, pca_model):
    """
    Projects scaled data onto an existing PCA space.

    Args:
        X_scaled_data (np.ndarray): Scaled expression data (cells x genes), where genes
                                    match those used to fit the PCA model.
        pca_model (sklearn.decomposition.PCA): The *fitted* PCA model.

    Returns:
        np.ndarray: The projected data (PCs).
    """
    print("Projecting data onto existing PCA space...")
    
    # Handle potential NaNs before transform
    if np.isnan(X_scaled_data).any():
        print("Warning: NaNs found in input data for PCA transform. Replacing with 0.")
        X_scaled_data = np.nan_to_num(X_scaled_data, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

    projected_data = pca_model.transform(X_scaled_data)
    print("Projection finished.")
    return projected_data


def align_genes_to_reference(adata_to_align, reference_gene_names):
    """
    Aligns the genes of an AnnData object to a specified list of reference gene names.
    Genes missing in adata_to_align but present in reference_gene_names are filled with 0.
    Genes in adata_to_align not in reference_gene_names are dropped.
    The order of genes in the output matches reference_gene_names.

    Args:
        adata_to_align (ad.AnnData): The AnnData object whose genes need to be aligned.
        reference_gene_names (list): A list of gene names representing the desired order and set.

    Returns:
        ad.AnnData: A new AnnData object with genes aligned and ordered.
    """
    print(f"Aligning {adata_to_align.shape[1]} genes to {len(reference_gene_names)} reference genes...")
    
    # Convert to DataFrame for robust column reindexing
    expr_df = pd.DataFrame(
        adata_to_align.X.toarray() if hasattr(adata_to_align.X, 'toarray') else adata_to_align.X,
        index=adata_to_align.obs_names,
        columns=adata_to_align.var_names
    )

    # Reindex columns to match the reference_gene_names order.
    # Genes not in `expr_df` but in `reference_gene_names` will be added as columns with `fill_value=0.0`.
    # Genes in `expr_df` but not in `reference_gene_names` will be dropped.
    aligned_expr_df = expr_df.reindex(columns=reference_gene_names, fill_value=0.0)

    # Safety check: Ensure the number of features matches
    if aligned_expr_df.shape[1] != len(reference_gene_names):
        raise ValueError(f"Gene alignment failed: Aligned data has {aligned_expr_df.shape[1]} features, "
                         f"but reference genes expect {len(reference_gene_names)}.")

    # Create a new AnnData object from the aligned DataFrame
    aligned_adata = ad.AnnData(
        X=aligned_expr_df.values,
        obs=adata_to_align.obs.loc[aligned_expr_df.index], # Ensure obs aligns if cells were filtered (though not by this fn)
        var=pd.DataFrame(index=aligned_expr_df.columns) # Set var_names from the aligned columns
    )
    print(f"Genes aligned. New shape: {aligned_adata.shape}")
    return aligned_adata
