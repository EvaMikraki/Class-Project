import pandas as pd
import scanpy as sc
import os

# Path to the dataset (update if your directory structure changes)
DATA_PATH = "/root/Class-Project/data/GSE81861/GSE81861_CRC_tumor_epithelial_cells_COUNT.csv.gz"

print(f"Loading dataset from {DATA_PATH}...")

try:
    # Load the dataset
    df = pd.read_csv(DATA_PATH, compression='gzip', index_col=0)

    print("✅ Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", df.columns[:5])  # Print first 5 sample names
    print(df.head())

    # Confirm orientation: genes should be rows, cells should be columns
    # If the reverse, uncomment this line to transpose
    # df = df.T

    # Construct AnnData object
    adata = sc.AnnData(df.T)  # .T because scanpy expects cells as rows, genes as columns
    print(f"✅ AnnData object created: {adata.shape} (cells, genes)")

except Exception as e:
    print("❌ Failed to load dataset:", e)

# Optional: Save the AnnData object
OUTPUT_PATH = "/root/Class-Project/data/GSE81861/gse81861_tumor_epithelial_cells.h5ad"
try:
    adata.write(OUTPUT_PATH)
    print(f"💾 AnnData object saved to: {OUTPUT_PATH}")
except Exception as e:
    print("❌ Failed to save AnnData object:", e)
