import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad
import sys

DATA_DIR = "/root/Class-Project/data/GSE84133"

# Expecting donor file name passed as argument
if len(sys.argv) != 2:
    print("Usage: python load_gse84133.py <donor_filename>")
    sys.exit(1)

file = sys.argv[1]
donor_id = file.split("_")[0]
path = os.path.join(DATA_DIR, file)
print(f"🔄 Loading {file}...")

df = pd.read_csv(path, index_col=0)
print(f"📐 Shape: {df.shape}")

metadata = df[['barcode', 'assigned_cluster', 'pk']].copy()
metadata['donor'] = donor_id
metadata.index = df.index

counts = df.drop(columns=['barcode', 'assigned_cluster', 'pk'])
counts = counts.astype(np.float32)
X = sp.csr_matrix(counts.values)

adata = ad.AnnData(X=X, obs=metadata)
adata.var_names = counts.columns
adata.obs_names = metadata.index

output_path = os.path.join(DATA_DIR, f"gse84133_{donor_id}.h5ad")
adata.write(output_path)
print(f"✅ Saved {donor_id} to {output_path}")

# Run each donor file separately:
# python scripts/load_gse84133.py GSM2230757_human1_umifm_counts.csv.gz
# python scripts/load_gse84133.py GSM2230758_human2_umifm_counts.csv.gz
# python scripts/load_gse84133.py GSM2230759_human3_umifm_counts.csv.gz
# python scripts/load_gse84133.py GSM2230760_human4_umifm_counts.csv.gz
