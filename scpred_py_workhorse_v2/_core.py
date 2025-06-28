# scpred_py/_core.py

import anndata as ad
import pandas as pd
import scanpy as sc # Imported for context, not directly used for preprocessing here
from . import _utils, _prediction, _training
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ScPredModel:
    """
    A class to encapsulate the scPred workflow.
    This version assumes input AnnData objects are already normalized, log1p-transformed,
    and highly variable gene (HVG) selected/aligned. It handles scaling and PCA internally.
    """
    def __init__(self):
        self.pca_model_ = None
        self.scaler_ = None
        self.classifier_ = None
        self.reference_genes_ = None # Stores the HVGs from the reference data

    def train(self, ref_adata, cell_type_key, n_components=30):
        """
        Trains the scPred model on reference data.
        Assumes ref_adata is already normalized, log1p-transformed, and HVG-selected.

        Args:
            ref_adata (ad.AnnData): Reference AnnData object (normalized, log1p, HVG-selected).
            cell_type_key (str): Key in `ref_adata.obs` for cell type labels.
            n_components (int): Number of PCs to compute.
        """
        _utils.check_adata(ref_adata)
        if cell_type_key not in ref_adata.obs:
            raise ValueError(f"'{cell_type_key}' not found in ref_adata.obs.")

        print("--- Starting ScPred Training ---")
        
        # 1. Store reference genes (which are assumed to be the HVGs after notebook preprocessing)
        self.reference_genes_ = ref_adata.var_names.tolist()

        # Ensure data is dense for scikit-learn
        X_data = ref_adata.X.toarray() if hasattr(ref_adata.X, 'toarray') else ref_adata.X

        # 2. Scale data and FIT the scaler
        # The data is already log1p-transformed by the notebook, now scale it.
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_data) # Use fit_transform here

        # 3. Perform PCA and FIT the PCA model
        self.pca_model_ = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca_model_.fit_transform(X_scaled) # Use fit_transform here

        # 4. Train Classifier
        labels = ref_adata.obs[cell_type_key]
        self.classifier_ = _training.train_svm(X_pca, labels)
        print("--- ScPred Training Complete ---")

    def predict(self, query_adata):
        """
        Predicts cell types on query data.
        Assumes query_adata is already normalized, log1p-transformed, and
        subsetted/aligned to the reference's HVGs.

        Args:
            query_adata (ad.AnnData): Query AnnData object (normalized, log1p, aligned to reference HVGs).

        Returns:
            ad.AnnData: Query AnnData object with prediction results added.
        """
        if self.pca_model_ is None or self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not fully trained.")

        _utils.check_adata(query_adata)
        print("--- Starting ScPred Prediction ---")
        
        # 1. Align query data to reference genes (HVGs)
        # This step is critical to ensure feature consistency.
        # Create a DataFrame from query data for robust column reindexing
        query_expr_df = pd.DataFrame(
            query_adata.X.toarray() if hasattr(query_adata.X, 'toarray') else query_adata.X,
            index=query_adata.obs_names,
            columns=query_adata.var_names
        )

        # Reindex columns to match the reference_genes_ order.
        # Any genes in reference_genes_ not in query_expr_df will be filled with 0.
        # Any genes in query_expr_df not in reference_genes_ will be dropped.
        aligned_query_expr_df = query_expr_df.reindex(columns=self.reference_genes_, fill_value=0.0)

        # Safety check: Ensure the number of features matches
        if aligned_query_expr_df.shape[1] != len(self.reference_genes_):
            raise ValueError(f"Gene alignment failed: Query data has {aligned_query_expr_df.shape[1]} features, "
                             f"but reference genes expect {len(self.reference_genes_)}.")

        # Convert back to NumPy array for scikit-learn compatibility
        X_query_data_aligned = aligned_query_expr_df.values
        
        # 2. Scale query data using the SAVED scaler
        X_query_scaled = self.scaler_.transform(X_query_data_aligned) # Use ONLY transform

        # 3. Project query data using the SAVED PCA model
        X_projected = self.pca_model_.transform(X_query_scaled) # Use ONLY transform

        # 4. Predict
        labels, probs = _prediction.predict_cells(X_projected, self.classifier_)

        # 5. Add results to original query_adata
        # Ensure categories are consistent and handle potential unassigned labels (if applicable later)
        query_adata.obs['scpred_prediction'] = pd.Categorical(labels, categories=self.classifier_.classes_)
        
        # Add probability columns
        if probs is not None:
            # Reindex probabilities DataFrame to align with original query_adata's index
            # This is important if any cells were filtered in previous preprocessing steps (though not currently in this setup)
            prob_df_reindexed = pd.DataFrame(probs.values, index=aligned_query_expr_df.index, columns=probs.columns)
            for col in prob_df_reindexed.columns:
                query_adata.obs[f"scpred_prob_{col}"] = prob_df_reindexed[col].reindex(query_adata.obs_names)
            
        # Add projected PCA to .obsm
        # Ensure proper reindexing for obsm as well if cells were filtered
        if X_projected.shape[0] == aligned_query_expr_df.shape[0]:
            pca_df_for_reindex = pd.DataFrame(X_projected, index=aligned_query_expr_df.index)
            query_adata.obsm['X_scpred_pca'] = pca_df_for_reindex.reindex(query_adata.obs_names).values
        else:
            print("Warning: Number of projected cells does not match aligned query data. X_scpred_pca not stored in .obsm.")

        print("--- ScPred Prediction Complete ---")
        return query_adata # Return the modified original query_adata
