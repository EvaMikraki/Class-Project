# scpred_py/_core.py

import anndata as ad
import pandas as pd
import scanpy as sc
from . import _utils, _preprocessing, _training, _prediction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # Import SVC for kernel flexibility

class ScPredModel:
    """
    A class to encapsulate the scPred workflow.
    This version handles all preprocessing internally (Strategy A).
    """
    def __init__(self):
        self.pca_model_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.reference_hvg_genes_ = None # Stores the HVGs identified during reference preprocessing

    def train(self, ref_adata, cell_type_key, n_components=30, 
              hvg_min_mean=0.0125, hvg_max_mean=3, hvg_min_disp=0.5, hvg_n_top_genes=None, hvg_flavor='seurat',
              svm_kernel='linear', svm_c=1.0, svm_random_state=42):
        """
        Trains the scPred model on reference data, performing all preprocessing internally.

        Args:
            ref_adata (ad.AnnData): Reference AnnData object (raw counts expected).
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
        """
        _utils.check_adata(ref_adata)
        if cell_type_key not in ref_adata.obs:
            raise ValueError(f"'{cell_type_key}' not found in ref_adata.obs.")

        print("--- Starting ScPred Training ---")

        # 1. Initial Preprocessing (Filter, Normalize, Log1p)
        processed_ref_adata = _preprocessing.initial_preprocessing_steps(ref_adata.copy())
        
        # 2. Select Highly Variable Genes (HVGs)
        processed_ref_adata_hvg, self.reference_hvg_genes_ = _preprocessing.select_highly_variable_genes(
            processed_ref_adata, 
            min_mean=hvg_min_mean, max_mean=hvg_max_mean, min_disp=hvg_min_disp, 
            n_top_genes=hvg_n_top_genes, flavor=hvg_flavor
        )
        
        # 3. Fit StandardScaler on the HVG-selected data
        self.scaler_ = _preprocessing.get_fitted_scaler(processed_ref_adata_hvg)
        
        # 4. Scale the HVG-selected data
        X_scaled_ref = _preprocessing.transform_data_with_scaler(processed_ref_adata_hvg, self.scaler_)

        # 5. Perform PCA on the scaled data
        self.pca_model_, X_pca = _preprocessing.get_fitted_pca(X_scaled_ref, n_components)
        
        # 6. Train Classifier
        labels = processed_ref_adata_hvg.obs[cell_type_key]
        self.classifier_ = _training.train_svm(
            X_pca, labels, 
            kernel=svm_kernel, c=svm_c, random_state=svm_random_state
        )
        print("--- ScPred Training Complete ---")

    def predict(self, query_adata, threshold=0.0): # Added threshold option for future
        """
        Predicts cell types on query data.
        Performs preprocessing and projection consistently with training.

        Args:
            query_adata (ad.AnnData): Query AnnData object (raw counts expected).
            threshold (float): Minimum probability for a prediction.
                               Predictions below this could be "unassigned" (future feature).

        Returns:
            ad.AnnData: Query AnnData object with prediction results added.
        """
        if self.pca_model_ is None or self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("Model must be trained before prediction.")

        _utils.check_adata(query_adata)
        print("--- Starting ScPred Prediction ---")
        
        # 1. Initial Preprocessing (Filter, Normalize, Log1p)
        processed_query_adata = _preprocessing.initial_preprocessing_steps(query_adata.copy())

        # 2. Align query genes to reference HVGs (learned during training)
        # This function handles subsetting and order matching
        aligned_query_adata = _preprocessing.align_genes_to_reference(
            processed_query_adata, self.reference_hvg_genes_
        )
        
        # 3. Scale query data using the *fitted reference scaler*
        X_scaled_query = _preprocessing.transform_data_with_scaler(aligned_query_adata, self.scaler_)

        # 4. Project scaled query data onto the *existing PCA space*
        X_projected = _preprocessing.transform_data_with_pca(X_scaled_query, self.pca_model_)

        # 5. Predict cell types and probabilities
        # Pass threshold to prediction function if it supports it (current _prediction.py doesn't, but can be extended)
        labels, probs = _prediction.predict_cells(X_projected, self.classifier_)

        # 6. Add results back to the original query_adata object
        # Ensure categories are consistent and handle potential unassigned labels (if applicable later)
        # Reindex prediction results to the original query_adata's index to handle any cell filtering
        query_adata.obs['scpred_prediction'] = pd.Series(
            labels, index=aligned_query_adata.obs_names # Labels from aligned adata cells
        ).reindex(query_adata.obs_names).astype('category') # Reindex to original query_adata for consistency
        
        # Add probability columns
        if probs is not None:
            # Create a DataFrame for probabilities, indexed by the aligned query cells
            prob_df_for_reindex = pd.DataFrame(
                probs.values, index=aligned_query_adata.obs_names, columns=self.classifier_.classes_
            )
            
            # Reindex to the original query_adata's obs_names to ensure correct alignment
            # Fill with NaN if cells were filtered out during preprocessing.
            for col in prob_df_for_reindex.columns:
                query_adata.obs[f"scpred_prob_{col}"] = prob_df_for_reindex[col].reindex(query_adata.obs_names)

        # Store the projected PCs in .obsm.
        # Ensure correct indexing and reindexing if cells were filtered during preprocessing.
        if X_projected.shape[0] == aligned_query_adata.shape[0]:
            # Create a DataFrame for X_scpred_pca with the aligned query cell names as index
            pca_df_for_reindex = pd.DataFrame(X_projected, index=aligned_query_adata.obs_names)
            # Reindex to the original query_adata's obs_names, filling NaNs if cells were filtered
            query_adata.obsm['X_scpred_pca'] = pca_df_for_reindex.reindex(query_adata.obs_names).values
        else:
            print("Warning: Number of projected cells does not match aligned query data. X_scpred_pca not stored in .obsm.")

        print("--- ScPred Prediction Complete ---")
        return query_adata