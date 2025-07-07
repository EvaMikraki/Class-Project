# scpred-py: A Python-Native Single-Cell Type Classification Tool

## Project Overview

`scpred-py` is a Python package for supervised cell-type classification from single-cell RNA-sequencing (scRNA-seq) data. Inspired by the `scPred` R package by Alquicira-Hernandez et al. (2019), this project aims to provide a robust, modular and user-friendly tool within the Python ecosystem. It implements a core classification pipeline encompassing data preprocessing, dimensionality reduction (PCA) and Support Vector Machine (SVM) classification, with a critical feature for confidence-based prediction thresholding.

This tool was developed as an ML class project, addressing challenges encountered during attempts to re-implement and extend the original R version, leading to a focus on building a consistently performing Python-native solution.

## Features

* **Flexible Preprocessing:** Supports both integrated internal preprocessing (filtering, normalization, log-transformation, HVG selection) and the option to use pre-processed data.
* **Robust Gene Alignment:** Ensures consistent feature space between reference and query datasets.
* **PCA-based Dimensionality Reduction:** Projects data into a lower-dimensional space.
* **Support Vector Machine Classification:** Utilizes a One-vs-Rest SVM with configurable kernel and regularization.
* **Confidence Thresholding:** Allows users to set a probability threshold to flag low-confidence predictions as 'unassigned', enhancing reliability.
* **Comprehensive Evaluation:** Integrates utilities for performance metric calculation and extensive plotting.

## Project Structure

The repository structure on GitHub is as follows:

```
.
├── models/
│   ├── scpred_model_paul15.pkl   # Saved model for Paul15
│   └── scpred_model_pbmc3k.pkl   # Saved model for pbmc3k
├── notebooks/
│   ├── 01_pbmc3k_hyperparameter_exploration.ipynb  # Notebook for hyperparameter exploration on pbmc3k
│   ├── 02_paul15_hyperparameter_exploration.ipynb  # Notebook for hyperparameter exploration on Paul15
│   ├── 03_pbmc3k_analysis.ipynb        # Notebook for analysis and plotting on pbmc3k
│   └── 04_paul15_analysis.ipynb        # Notebook for analysis and plotting on Paul15
├── scpred_py/                  # The core scpred-py package
│   ├── __init__.py
│   ├── _analysis_utils.py      # Utility functions for evaluation and reporting
│   ├── _core.py                # Core ScPredModel class (train, predict methods)
│   ├── _prediction.py          # Prediction-related helper functions
│   ├── _preprocessing.py       # Data preprocessing and transformation functions
│   ├── _training.py            # Model training helper functions
│   └── _utils.py               # General utility functions (e.g., AnnData checks)
├── .gitignore
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

**Note on excluded directories:** Directories such as `.git_backup/`, `.venv/` and `data/` are excluded from the GitHub repository via the `.gitignore` configuration. The `data/` directory is intended as an optional location for users to save `.h5ad` files if desired, though raw data is loaded directly from `scanpy.datasets` in the notebooks and preprocessed reference data is stored internally within the `ScPredModel` instances.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  **Create and activate a virtual environment:**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    # On Windows:
    # .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary libraries, including `scanpy`, `anndata`, `scikit-learn`, `numpy`, `pandas`, `matplotlib` and `seaborn`.

## Usage and Reproducibility

The project's analysis and results can be reproduced by running the Jupyter notebooks located in the `notebooks/` directory.

1.  **Start Jupyter Lab/Notebook:**
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```
2.  **Navigate to `notebooks/`:** Open the `notebooks/` directory in your Jupyter interface.

### Notebooks Overview:

The notebooks are designed to be self-contained for clarity and reproducibility. Each notebook includes its own data loading, preprocessing and model training steps, ensuring consistency regardless of the order in which notebooks are run.

* **`01_pbmc3k_hyperparameter_exploration.ipynb`**: This notebook facilitates the exploration of hyperparameter combinations for the pbmc3k dataset. It utilizes the `run_scpred_experiment` function to systematically train models across 32 different parameter settings. The primary purpose of this notebook is to observe how the model behaves under various configurations and to inform the manual selection of empirically chosen hyperparameters, which are then applied in the analysis notebooks. The models trained in this notebook are *not* saved.
* **`02_paul15_hyperparameter_exploration.ipynb`**: Similar to the pbmc3k notebook, this one performs hyperparameter exploration for the Paul15 dataset, running 32 experiments to assess model behavior across different parameter settings. The models trained here are also *not* saved, serving solely for empirical analysis to guide hyperparameter selection.
* **`03_pbmc3k_analysis.ipynb`**: This notebook contains the complete pipeline for the pbmc3k dataset. It loads the raw data, performs preprocessing, trains the `scpred-py` model with the *empirically selected hyperparameters* and then applies the model to the query data. It automates the evaluation and plotting processes using the `run_scenario` and `plot_results_comprehensive` functions, presenting the classification results for both no-threshold (0.0) and selected threshold (0.8) scenarios. The `ScPredModel` for pbmc3k is saved as `scpred_model_pbmc3k.pkl` in the `models/` directory upon completion.
* **`04_paul15_analysis.ipynb`**: This notebook provides the complete pipeline for the Paul15 dataset. It loads data, preprocesses, trains the `scpred-py` model with its *empirically selected hyperparameters* and evaluates predictions for no-threshold (0.0) and selected threshold (0.6) scenarios. It also leverages the `run_scenario` and `plot_results_comprehensive` functions for automated evaluation and visualization. The `ScPredModel` for Paul15 is saved as `scpred_model_paul15.pkl` in the `models/` directory upon completion.

**Experimenting with Thresholds:** The `run_scenario` function, used in the `03_pbmc3k_analysis.ipynb` and `04_paul15_analysis.ipynb` notebooks, is designed to facilitate easy experimentation with different prediction confidence thresholds. Users can modify the `threshold` parameter within the `run_scenario` calls to observe its impact on classification results and visualizations in an automated manner.

## Core Package (`scpred_py/`)

The `scpred_py/` directory contains the source code for the `scpred-py` package.
* `_core.py`: Defines the `ScPredModel` class, which orchestrates the training and prediction workflows. It includes the `perform_preprocessing` flag for flexible data handling.
* `_preprocessing.py`: Contains functions for initial data preprocessing (filtering, normalization, log-transformation), highly variable gene selection, data scaling, PCA fitting and gene alignment.
* `_training.py`: Implements the SVM classifier training logic.
* `_prediction.py`: Handles prediction, probability estimation and confidence thresholding.
* `_analysis_utils.py`: Provides utilities for evaluating model performance and generating classification reports.

## License

Proprietary for academic use only.