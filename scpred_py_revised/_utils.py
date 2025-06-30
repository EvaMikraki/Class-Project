# scpred_py/_utils.py

import scanpy as sc
import anndata as ad
import pandas as pd

def check_adata(adata):
    """Checks if the input is an AnnData object."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input must be an AnnData object.")
    print("AnnData object check passed.")

# Note: get_common_genes is no longer directly used in _core.py's prediction
# due to the more robust align_genes_to_reference function in _preprocessing.py.
# You can keep it if you have other uses, or remove it if it's truly unused.
# def get_common_genes(ref_adata, query_adata):
#     # ... (existing code) ...