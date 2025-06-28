# scpred_py/_utils.py:
import scanpy as sc

import anndata as ad


def check_adata(adata):

    """Checks if the input is an AnnData object."""

    if not isinstance(adata, ad.AnnData):

        raise TypeError("Input must be an AnnData object.")

    print("AnnData object check passed.")


def get_common_genes(ref_adata, query_adata):

    """Finds common genes between reference and query datasets."""

    check_adata(ref_adata)

    check_adata(query_adata)


    ref_genes = ref_adata.var_names

    query_genes = query_adata.var_names


    common_genes = list(set(ref_genes) & set(query_genes))


    if len(common_genes) == 0:

        raise ValueError("No common genes found between reference and query data.")


    print(f"Found {len(common_genes)} common genes.")

    return common_genes 