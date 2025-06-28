import pandas as pd
import numpy as np # Already in your script
import scanpy as sc # Already in your script

METADATA_PATH = "/root/Class-Project/data/GSE81608/GSE81608_series_matrix.txt.gz"

# This is an educated guess based on typical GEO series matrix files.
# You might need to adjust skiprows or comment character based on actual inspection.
try:
    # Try reading the file, skipping lines starting with '!' and assuming tab separation
    # You might need to adjust 'skiprows' based on where the actual data table begins.
    # Reading only a few rows initially to avoid memory issues during inspection.
    meta_df_test = pd.read_csv(METADATA_PATH, sep='\t', comment='!', nrows=200)
    print("Metadata file loaded for inspection.")
    print(meta_df_test.head())
    print(meta_df_test.columns)

    # A common pattern is that sample names are the column headers, and metadata fields are rows.
    # You might need to transpose this DataFrame to get cells as rows and metadata fields as columns.
    meta_df_transposed = meta_df_test.T
    print("\nTransposed Metadata (head):\n", meta_df_transposed.head())
    print("\nTransposed Metadata (columns):\n", meta_df_transposed.columns)

    # Look for a column that contains cell type information.
    # Common GEO metadata fields include "characteristics_ch1", "cell type", etc.
    # You'll need to identify this specific column after inspection.

except Exception as e:
    print(f"Error inspecting metadata file: {e}")