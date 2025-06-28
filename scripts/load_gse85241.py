import pandas as pd
import numpy as np
import scanpy as sc
import os
from scipy import sparse

# Paths
DATA_DIR = "/root/Class-Project/data/GSE85241"
OUTPUT_FILE = os.path.join(DATA_DIR, "gse85241.h5ad")

# Files
EXPR_FILE = os.path.join(DATA_DIR, "GSE85241_cellsystems_dataset_4donors_updated.csv.gz")
BARCODE_FILE = os.path.join(DATA_DIR, "GSE85241_cel-seq_barcodes.csv.gz")

print("🔄 Loading expression matrix...")
expr = pd.read_csv(EXPR_FILE, index_col=0)
print(f"📐 Expression shape: {expr.shape}")

# Check if orientation is genes x cells — transpose if needed
if expr.shape[0] > expr.shape[1]:
    print("🔁 Transposing expression matrix (genes x cells → cells x genes)...")
    expr = expr.T

print("🔄 Loading barcode metadata...")
barcodes = pd.read_csv(BARCODE_FILE)

# Sanity check: matching dimensions
if expr.shape[0] != barcodes.shape[0]:
    raise ValueError(f"❌ Mismatch between cell count in expression ({expr.shape[0]}) and barcode metadata ({barcodes.shape[0]})")

# Use barcodes as obs
expr.index = barcodes["cell_id"]
barcodes.set_index("cell_id", inplace=True)

print("🧪 Creating sparse AnnData object...")
X = sparse.csr_matrix(expr.values.astype(np.float32))
adata = sc.AnnData(X=X, obs=barcodes, var=pd.DataFrame(index=expr.columns))

print(f"✅ AnnData created: {adata.shape} (cells, genes)")
print("💾 Saving to .h5ad...")
adata.write(OUTPUT_FILE)
print(f"✅ Saved to {OUTPUT_FILE}")
