import anndata as ad
import pandas as pd
import scanpy as sc
from . import _utils, _preprocessing, _training, _prediction
# In _core.py
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ScPredModel:
    def __init__(self):
        self.pca_model_ = None
        self.scaler_ = None # <-- ADD THIS
        self.classifier_ = None
        self.reference_genes_ = None
        # self.reference_adata_ is no longer needed

    def train(self, ref_adata, cell_type_key, n_components=30):
        _utils.check_adata(ref_adata)
        
        # 1. Store reference genes
        self.reference_genes_ = ref_adata.var_names.tolist()

        # 2. Scale data and FIT the scaler
        X_data = ref_adata.X.toarray() if hasattr(ref_adata.X, 'toarray') else ref_adata.X
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
        if self.pca_model_ is None or self.classifier_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not fully trained.")

        _utils.check_adata(query_adata)
        
        # 1. Align query data to reference genes
        # Ensure query genes are in the same order as the original reference
        missing_genes = set(self.reference_genes_) - set(query_adata.var_names)
        if len(missing_genes) > 0:
            print(f"Warning: {len(missing_genes)} reference genes not found in query. Filling with 0s.")
        
        # Create a new AnnData with all reference genes, filling missing with 0
        aligned_query = ad.AnnData(
            X=pd.DataFrame(0, index=query_adata.obs_names, columns=self.reference_genes_),
            obs=query_adata.obs
        )
        common_genes = list(set(self.reference_genes_) & set(query_adata.var_names))
        aligned_query[:, common_genes].X = query_adata[:, common_genes].X

        # 2. Scale query data using the SAVED scaler
        X_query_data = aligned_query.X.toarray() if hasattr(aligned_query.X, 'toarray') else aligned_query.X
        X_query_scaled = self.scaler_.transform(X_query_data) # Use ONLY transform

        # 3. Project query data using the SAVED PCA model
        X_projected = self.pca_model_.transform(X_query_scaled) # Use ONLY transform

        # 4. Predict
        labels, probs = _prediction.predict_cells(X_projected, self.classifier_)

        # 5. Add results to original query_adata
        query_adata.obs['scpred_prediction'] = pd.Categorical(labels, categories=self.classifier_.classes_)
        # ... (rest of the code for adding probs) ...
        
        query_adata.obsm['X_scpred_pca'] = X_projected
        return query_adata