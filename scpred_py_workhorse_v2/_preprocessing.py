# scpred_py/_preprocessing.py
# This file is intentionally minimal, as core preprocessing steps (scaling, PCA)
# are now handled directly within ScPredModel in _core.py.
# Other preprocessing (normalization, log1p, HVG) are handled in the notebook.

import scanpy as sc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Kept for completeness if you decide to re-add.

# No functions are explicitly defined here, as they are handled elsewhere
# You could leave this file empty, or add very generic functions that are
# truly independent of the ScPredModel's internal workflow.
# For instance, a function for general filtering if not done in notebook:
# def basic_filtering(adata):
#     sc.pp.filter_cells(adata, min_genes=200)
#     sc.pp.filter_genes(adata, min_cells=3)
#     return adata
