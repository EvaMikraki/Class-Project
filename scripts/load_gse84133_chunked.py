import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import anndata as ad

DATA_DIR = "/root/Class-Project/data/GSE84133"
FILENAME = "GSM2230759_human3_umifm_counts.csv.gz"
FILEPATH = os.path.join(DATA_DIR, FILENAME)

print(f"🔄 Chunk-loading {FILENAME}...")

chunks = []
metadata_chunks = []

reader = pd.read_csv(FILEPATH, index_col=0, chunksize=500)
for i, chunk in enumerate(reader):
    print(f"🧩 Processing chunk {i + 1}...")

    meta = chunk[['barcode', 'assigned_cluster', 'pk']].copy()
    meta['donor'] = "GSM2230759"
    meta.index = chunk.index

    counts = chunk.drop(columns=['barcode', 'assigned_cluster', 'pk']).astype(np.float32)
    X = sp.csr_matrix(counts.values)

    adata = ad.AnnData(X=X, obs=meta)
    adata.var_names = counts.columns
    adata.obs_names = meta.index

    chunks.append(adata)

print("🔗 Concatenating all chunks...")
adata_all = ad.concat(chunks, join="outer")

output_file = os.path.join(DATA_DIR, "gse84133_GSM2230759.h5ad")
adata_all.write(output_file)
print(f"✅ Saved to {output_file}")
