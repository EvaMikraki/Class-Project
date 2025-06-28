from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def train_svm(X_pca, labels):
    """
    Trains a One-vs-Rest SVM classifier on PCA components.

    Args:
        X_pca (np.ndarray): PCA-transformed data (cells x PCs).
        labels (pd.Series or np.ndarray): Cell type labels for each cell.

    Returns:
        sklearn.multiclass.OneVsRestClassifier: The trained classifier.
    """
    print("Training One-vs-Rest SVM classifier...")

    # A simple linear SVM - scPred often uses this.
    # We use probability=True for prediction probabilities later.
    # C=1 is a default, scPred does hyperparameter tuning - A key area for expansion!
    svm = SVC(kernel='linear', probability=True, C=1.0, random_state=42)

    # Use One-vs-Rest strategy for multi-class problems
    clf = OneVsRestClassifier(svm, n_jobs=-1) # Use all available cores

    # --- Optional: Add Cross-Validation (Good Practice) ---
    # cv = StratifiedKFold(n_splits=5)
    # scores = cross_val_score(clf, X_pca, labels, cv=cv, scoring='accuracy')
    # print(f"Cross-validation accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    # --------------------------------------------------------

    # Train the final model on all data
    clf.fit(X_pca, labels)
    print("Training finished.")

    return clf