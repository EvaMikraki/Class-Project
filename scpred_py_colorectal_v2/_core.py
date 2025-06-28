# scpred_py/_core.py


import anndata as ad

import pandas as pd

import numpy as np


from ._utils import check_adata, get_common_genes

from ._preprocessing import get_pca, project_pca

from ._training import train_svm

from ._prediction import predict_cells



class ScPredModel:

    def __init__(self):

        self.pca_model_ = None

        self.mean_ = None  # Replaces scaler_

        self.std_ = None   # Replaces scaler_

        self.informative_pcs_ = None

        self.classifier_ = None

        self.reference_genes_ = None

        self.reference_adata_ = None


    # In scpred_py/_core.py, inside the ScPredModel class

    def train(self, ref_adata, cell_type_key, kernel='linear', n_components=30):
        check_adata(ref_adata)
        if cell_type_key not in ref_adata.obs:
            raise ValueError(f"'{cell_type_key}' not found in ref_adata.obs.")

        print("--- Starting ScPred Training ---")
        self.reference_adata_ = ref_adata.copy()
        self.reference_genes_ = self.reference_adata_.var_names.tolist()
        
        self.pca_model_, self.mean_, self.std_, X_pca = get_pca(self.reference_adata_, n_components)
        
        # NOTE: Using all PCs for now as a baseline.
        # We can integrate select_informative_pcs here later if needed.
        print("--- Using all calculated PCs as features ---")
        self.informative_pcs_ = np.arange(X_pca.shape[1])
        X_pca_selected = X_pca
        
        labels = self.reference_adata_.obs[cell_type_key]
        
        # --- MODIFIED: Pass the kernel argument to train_svm ---
        self.classifier_ = train_svm(X_pca_selected, labels, kernel=kernel)
        
        print("--- ScPred Training Complete ---")


    def predict(self, query_adata, threshold=0.0):

        if self.pca_model_ is None or self.classifier_ is None or self.mean_ is None:

            raise RuntimeError("Model must be trained before prediction.")


        check_adata(query_adata)

       

        print("--- Starting ScPred Prediction ---")

       

        common_genes = get_common_genes(self.reference_adata_, query_adata)

        query_adata_sub = query_adata[:, common_genes].copy()

       

        # Pass the mean and std for projection

        X_projected = project_pca(query_adata_sub, self.pca_model_, self.mean_, self.std_)

       

        X_projected_selected = X_projected[:, self.informative_pcs_]


        labels_array, probs_df = predict_cells(self.classifier_, X_projected_selected, threshold=threshold)


        query_adata.obs['scpred_prediction'] = pd.Series(

            labels_array, index=query_adata.obs.index).astype('category')

       

        if probs_df is not None:

            probs_df.index = query_adata.obs.index

            prob_cols = [f"scpred_prob_{c}" for c in self.classifier_.estimator.classes_]

            query_adata.obs[prob_cols] = probs_df

           

        query_adata.obsm['X_scpred_pca'] = X_projected_selected


        print("--- ScPred Prediction Complete ---")

        return query_adata
