import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from . import _utils, _preprocessing, _training, _prediction

def predict(self, query_adata):
    """
    Predicts cell types on query data.
    """
    if self.pca_model_ is None or self.classifier_ is None:
        raise RuntimeError("Model must be trained before prediction.")

    _utils.check_adata(query_adata)
    
    print("--- Starting ScPred Prediction ---")
    
    # 1. Align query genes to the reference genes used for training
    # Create a new AnnData with all reference genes, filling missing ones with 0
    missing_genes = set(self.reference_genes_) - set(query_adata.var_names)
    if len(missing_genes) > 0:
        print(f"Warning: {len(missing_genes)} reference genes not found in query data. Filling with zeros.")
        
        # Create a zero-filled matrix for missing genes
        missing_data = np.zeros((query_adata.shape[0], len(missing_genes)))
        
        # Concatenate existing and missing data
        import scipy.sparse
        combined_X = scipy.sparse.hstack([query_adata.X, missing_data], format='csr')
        
        # Create new var DataFrame
        combined_var = pd.concat([
            query_adata.var, 
            pd.DataFrame(index=list(missing_genes))
        ])
        
        # Create a new aligned AnnData object
        query_aligned = ad.AnnData(X=combined_X, obs=query_adata.obs, var=combined_var)
        
        # Ensure the final gene order matches the reference
        query_aligned = query_aligned[:, self.reference_genes_].copy()
    else:
        # If all genes are present, just re-order them to be safe
        query_aligned = query_adata[:, self.reference_genes_].copy()

    # NOTE: This assumes the query data has already been preprocessed and scaled
    # using the parameters from the reference set, as described in Issue 1.
    
    # 2. Project Query Data using the ORIGINAL PCA model
    # DO NOT RE-FIT. ONLY TRANSFORM.
    X_projected = _preprocessing.project_pca(query_aligned, self.pca_model_)

    # 3. Predict
    labels, probs = _prediction.predict_cells(X_projected, self.classifier_)

    # 4. Add results to query_adata
    query_adata.obs['scpred_prediction'] = pd.Categorical(labels, categories=self.classifier_.classes_)
    if probs is not None:
        prob_cols = [f"scpred_prob_{c}" for c in self.classifier_.classes_]
        query_adata.obs[prob_cols] = probs.values
        
    query_adata.obsm['X_scpred_pca'] = X_projected

    print("--- ScPred Prediction Complete ---")
    return query_adata