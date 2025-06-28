import scanpy as sc
import anndata as ad

def check_adata(adata):
    """Checks if the input is an AnnData object."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input must be an AnnData object.")
    print("AnnData object check passed.")
