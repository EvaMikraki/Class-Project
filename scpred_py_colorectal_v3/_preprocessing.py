# scpred_py/_preprocessing.py


import scanpy as sc

from sklearn.decomposition import PCA

import numpy as np

from scipy.stats import mannwhitneyu

from statsmodels.stats.multitest import multipletests


# standard_preprocess function remains the same (without any sc.pp.scale)


def get_pca(adata, n_components=30):

    """

    Performs manual scaling with clipping (like scanpy) and PCA.

    Returns the fitted PCA model and the scaling parameters (mean, std).

    """

    print(f"Performing manual scaling (with clipping) and PCA with {n_components} components...")

   

    X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X


    # 1. Manually calculate mean and std from reference data

    mean = X_data.mean(axis=0)

    std = X_data.std(axis=0)

    # Replace std=0 with 1 to avoid division by zero

    std[std == 0] = 1


    # 2. Scale data and clip to max_value=10

    X_scaled = (X_data - mean) / std

    np.clip(X_scaled, a_min=None, a_max=10, out=X_scaled) # In-place clipping

   

    # 3. Perform PCA on the scaled and clipped data

    pca = PCA(n_components=n_components, random_state=42)

    X_pca = pca.fit_transform(X_scaled)

   

    print("Scaling and PCA finished.")

    # Return the scaling parameters instead of the scaler object

    return pca, mean, std, X_pca



def project_pca(adata, pca_model, mean, std):

    """

    Projects query data onto an existing PCA space using saved mean/std values.

    """

    print("Manually scaling query data and projecting onto existing PCA space...")

   

    X_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X


    # 1. Scale query data using the *same* mean/std from the reference

    X_scaled = (X_data - mean) / std

    np.clip(X_scaled, a_min=None, a_max=10, out=X_scaled)

   

    # 2. Project scaled data onto the existing PCA space

    X_projected = pca_model.transform(X_scaled)

   

    print("Projection finished.")

    return X_projected


# select_informative_pcs function remains the same


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