import pandas as pd
import scanpy as sc
import numpy as np
import os
import re

DATA_PATH = "/root/Class-Project/data/GSE81608/GSE81608_human_islets_rpkm.txt.gz"
SAVE_PATH = "/root/Class-Project/data/GSE81608/gse81608.h5ad"
METADATA_PATH = "/root/Class-Project/data/GSE81608/GSE81608_series_matrix.txt.gz"

print(f"Loading dataset from {DATA_PATH}...\n")

try:
    # --- Load Expression Data ---
    df = pd.read_csv(DATA_PATH, sep="\t", dtype=np.float32)
    print("✅ Expression dataset loaded successfully into DataFrame.")
    print(f"DataFrame Shape: {df.shape}")

    if "gene.id" in df.columns:
        df.set_index("gene.id", inplace=True)
    else:
        print("⚠️ 'gene.id' column not found. Assuming the first column is the gene ID and setting it as index.")
        df.set_index(df.columns[0], inplace=True)

    adata = sc.AnnData(df.T) # Transpose to get cells as rows
    print("✅ AnnData object created from expression data.")
    print(f"AnnData shape (cells x genes): {adata.shape}")

    adata.var_names_make_unique()
    print("✅ Gene names (adata.var_names) made unique.")

    # --- Load Cell Metadata ---
    print(f"\nLoading cell metadata from {METADATA_PATH}...")

    meta_df_raw = pd.read_csv(
        METADATA_PATH,
        sep='\t',
        skiprows=27,
        comment=None
    )

    meta_df_transposed = meta_df_raw.T

    # Set the first row as columns and remove it from data
    meta_df_transposed.columns = meta_df_transposed.iloc[0]
    meta_df = meta_df_transposed[1:]

    # --- NEW: Make column names unique before further processing ---
    # This prevents the 'DataFrame' object has no attribute 'str' error
    cols = pd.Series(meta_df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(cols[cols == dup].shape[0])]
    meta_df.columns = cols

    # Clean up column names by removing leading '!'
    meta_df.columns = meta_df.columns.str.lstrip('!')

    print("\nMetadata DataFrame Head (after initial load, transpose, and column renaming):\n", meta_df.head())
    print("\nMetadata DataFrame Columns (unique):\n", meta_df.columns.tolist())


    # --- Identify and Extract Cell Type Information ---
    cell_type_col = None
    cell_type_values = pd.Series(dtype='object') # Initialize an empty series for cell types

    # Potential keywords to identify cell type column in pancreatic islet data
    cell_type_keywords = ['alpha cell', 'beta cell', 'delta cell', 'gamma cell', 'acinar cell', 'ductal cell', 'islet cell']

    # Iterate through all columns, looking for 'characteristics_ch1' that contain relevant cell type terms
    for col in meta_df.columns:
        if 'characteristics_ch1' in col.lower():
            # Convert the column to string type for searching
            col_as_str = meta_df[col].astype(str).str.lower()
            
            # Check if any of the cell_type_keywords are present in this column
            if any(keyword in val for val in col_as_str.unique() for keyword in cell_type_keywords):
                print(f"Found potential cell type column based on keywords: '{col}'")
                
                # Now, let's try to extract the cell type.
                # The format might be 'cell type: X' or just 'X' if it's the only characteristic.
                # We will try to extract whatever follows 'cell type: ' or just take the whole string if it's simpler.
                
                extracted_types = col_as_str.apply(
                    lambda x: re.search(r'cell type: (.*)', x).group(1).strip()
                    if re.search(r'cell type: (.*)', x) else x.strip() # If 'cell type: ' not found, take the whole string
                )
                
                # Filter out any potential non-cell type values (like just "nan" or "none")
                extracted_types = extracted_types[~extracted_types.isin(['nan', 'none', 'null', '', 'not applicable'])]
                extracted_types = extracted_types.dropna() # Drop any remaining NaNs

                if not extracted_types.empty and len(extracted_types.unique()) > 1: # Ensure more than one unique cell type
                    cell_type_values = extracted_types
                    cell_type_col = col # Keep track of the column that provided the data
                    break # Stop after finding the first suitable column

    if not cell_type_values.empty:
        print(f"\n✅ Identified and extracted cell types from column: '{cell_type_col}'")
        print("\nExtracted cell types preview:\n", cell_type_values.value_counts().head())

        # ... (rest of the mapping code for cell_type_values to adata.obs remains the same)
        # --- Mapping Cell IDs (Sample_X to GSMXXXXXX to cell_type) ---
        # 1. Create a mapping from 'Pancreatic islet cell sample X' (from Sample_title) to GSM ID (meta_df index)
        # Clean 'Sample_title' (remove quotes and "Pancreatic islet cell ")
        meta_df['clean_title'] = meta_df['Sample_title'].str.strip('"').str.replace('Pancreatic islet cell sample ', 'Sample_')

        # Create a Series with GSM IDs as values and clean_title as index
        title_to_gsm_mapping = pd.Series(meta_df.index, index=meta_df['clean_title'])

        # Map adata.obs_names to their corresponding GSM ID
        mapped_gsm_ids = adata.obs_names.map(title_to_gsm_mapping)
        
        # Use the mapped GSM IDs to get the cell types from the cell_type_values Series
        adata.obs['cell_type'] = mapped_gsm_ids.map(cell_type_values)

        # Final check for mapping success
        if adata.obs['cell_type'].isnull().any():
            print("❌ Warning: Some cells did not get a cell_type after mapping. Check mapping logic!")
            print("Cells with missing cell types (first 5):\n", adata.obs[adata.obs['cell_type'].isnull()].head())
        else:
            print("✅ Cell types successfully mapped and added to adata.obs.")
            print("\nAnnData observations (cells) head with 'cell_type':\n", adata.obs.head())
            print("\nValue counts for 'cell_type':\n", adata.obs['cell_type'].value_counts())

    else:
        print("❌ Could not find a suitable cell type column in metadata. Metadata not added.")

    print("\nAnnData observations (cells) final head:\n", adata.obs.head())
    print("\nAnnData variables (genes) final head:\n", adata.var.head())

    adata.write(SAVE_PATH)
    print(f"\n💾 Saved AnnData to: {SAVE_PATH}")

except Exception as e:
    print(f"\n❌ Error during script execution: {e}")
    import traceback
    traceback.print_exc()