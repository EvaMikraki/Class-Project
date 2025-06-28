# scripts/load_emtab5061.py

import scanpy as sc
import pandas as pd
import os

DATA_PATH = "/root/Class-Project/data/E-MTAB-5061/pancreas_refseq_rpkms_counts_3514sc.txt.gz"
OUTPUT_FILE = "/root/Class-Project/data/E-MTAB-5061/emtab5061_adata.h5ad"

print(f"Loading dataset from {DATA_PATH}...")

# Load tab-delimited expression data, skipping the comment row and treating the first column as gene names
df = pd.read_csv(DATA_PATH, sep="\t", comment="#", header=0)

# First column is gene names
gene_names = df.iloc[:, 0].values
data = df.iloc[:, 1:].transpose()

# Construct AnnData object
adata = sc.AnnData(X=data.values)
adata.var_names = gene_names
adata.obs_names = data.index

print(f"Shape of AnnData object: {adata.shape}")

# Save to disk
adata.write(OUTPUT_FILE)
print(f"AnnData object saved to {OUTPUT_FILE}")
