# scpred_py/_core.py
import anndata as ad
import pandas as pd
import scanpy as sc
from . import _prediction, _preprocessing, _training, _utils

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