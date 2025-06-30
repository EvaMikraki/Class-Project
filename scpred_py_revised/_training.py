# scpred_py/_training.py

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold # Kept commented out as in original
import numpy as np
import pandas as pd # Ensure pandas is imported if labels are Series

def train_svm(X_pca, labels, kernel='linear', c=1.0, random_state=42):
    """
    Trains a One-vs-Rest SVM classifier on PCA components with configurable kernel and C.

    Args:
        X_pca (np.ndarray): PCA-transformed data (cells x PCs).
        labels (pd.Series or np.ndarray): Cell type labels for each cell.
        kernel (str): Specifies the kernel type to be used in the algorithm.
                      Must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'.
        c (float): Regularization parameter. The strength of the regularization is
                   inversely proportional to C. Must be strictly positive.
        random_state (int): Seed for the random number generator for reproducibility.

    Returns:
        sklearn.multiclass.OneVsRestClassifier: The trained classifier.
    """
    print(f"Training One-vs-Rest SVM classifier with kernel='{kernel}', C={c}...")

    # Initialize the SVM with specified kernel and C parameter
    svm = SVC(kernel=kernel, probability=True, C=c, random_state=random_state)

    # Use One-vs-Rest strategy for multi-class problems
    clf = OneVsRestClassifier(svm, n_jobs=-1) # Use all available cores

    # Train the final model on all data
    clf.fit(X_pca, labels)
    print("Training finished.")

    return clf
