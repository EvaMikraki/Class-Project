# scpred_py/_core.py

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
        self.scaler_ = None  # To store the StandardScaler
        self.informative_pcs_ = None # To store indices of selected PCs
        self.classifier_ = None
        self.reference_genes_ = None
        self.reference_adata_ = None 

    def train(self, ref_adata, cell_type_key, n_components=30):
        """
        Trains the scPred model on reference data.

        Args:
            ref_adata (ad.AnnData): Reference AnnData object (should NOT be scaled yet).
            cell_type_key (str): Key in `ref_adata.obs` for cell type labels.
            n_components (int): Number of PCs to compute.
        """
        _utils.check_adata(ref_adata)
        if cell_type_key not in ref_adata.obs:
            raise ValueError(f"'{cell_type_key}' not found in ref_adata.obs.")

        print("--- Starting ScPred Training ---")
        self.reference_adata_ = ref_adata.copy()
        self.reference_genes_ = self.reference_adata_.var_names.tolist()
        
        # 1. Perform Scaling and PCA
        self.pca_model_, self.scaler_, X_pca = _preprocessing.get_pca(self.reference_adata_, n_components)
        
        # 2. Select Informative PCs
        labels = self.reference_adata_.obs[cell_type_key]
        self.informative_pcs_ = _preprocessing.select_informative_pcs(X_pca, labels)
        
        # Filter PCs based on selection
        X_pca_selected = X_pca[:, self.informative_pcs_]

        # 3. Train Classifier on selected PCs
        self.classifier_ = _training.train_svm(X_pca_selected, labels)
        print("--- ScPred Training Complete ---")

    # scpred_py/_core.py (predict method only)

    def predict(self, query_adata, threshold=0.5): # Added threshold parameter
        """
        Predicts cell types on query data.

        Args:
            query_adata (ad.AnnData): Query AnnData object.
            threshold (float): Minimum probability for a prediction.
                               Predictions below this are "unassigned".

        Returns:
            ad.AnnData: Query AnnData object with prediction results added.
        """
        if self.pca_model_ is None or self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("Model must be trained before prediction.")

        _utils.check_adata(query_adata)
        
        print("--- Starting ScPred Prediction ---")
        
        common_genes = _utils.get_common_genes(self.reference_adata_, query_adata)
        query_adata_sub = query_adata[:, common_genes].copy()
        
        X_projected = _preprocessing.project_pca(query_adata_sub, self.pca_model_, self.scaler_)

        # Note: We now use ALL PCs as our stable baseline
        X_projected_selected = X_projected[:, self.informative_pcs_]

        # Pass the threshold to the prediction function
        labels, probs = _prediction.predict_cells(self.classifier_, X_projected_selected, threshold=threshold)

        query_adata.obs['scpred_prediction'] = labels # Now includes "unassigned"
        if probs is not None:
            prob_cols = [f"scpred_prob_{c}" for c in self.classifier_.classes_]
            query_adata.obs[prob_cols] = probs.values
            
        query_adata.obsm['X_scpred_pca'] = X_projected_selected

        print("--- ScPred Prediction Complete ---")
        return query_adata