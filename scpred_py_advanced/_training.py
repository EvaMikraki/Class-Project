# scpred_py/_training.py

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

def train_svm(X_pca, labels):
    """
    Trains a One-vs-Rest SVM classifier with an RBF kernel and uses
    GridSearchCV for hyperparameter tuning.

    Args:
        X_pca (np.ndarray): PCA-transformed data (cells x PCs).
        labels (pd.Series or np.ndarray): Cell type labels for each cell.

    Returns:
        sklearn.multiclass.OneVsRestClassifier: The trained and tuned classifier.
    """
    print("Training RBF SVM classifier with hyperparameter tuning...")

    # Define the parameter grid for C and gamma
    # This is a small grid for speed. For a real run, you might expand it.
    # e.g., 'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]
    param_grid = {
        'estimator__C': [1, 10, 100],
        'estimator__gamma': ['scale', 'auto']
    }

    # Initialize the RBF SVM
    # We use probability=True for prediction probabilities later.
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    
    # Wrap it in OneVsRestClassifier for multi-class problems
    ovr_clf = OneVsRestClassifier(svm, n_jobs=-1)

    # Set up cross-validation for the grid search
    cv = StratifiedKFold(n_splits=3) # Use 3-fold CV for speed

    # Set up GridSearchCV
    # It will search over param_grid, using 3-fold CV, for the best estimator.
    # Note the 'estimator__' prefix for parameters, which tells GridSearchCV
    # to pass these parameters to the estimator *inside* OneVsRestClassifier.
    clf = GridSearchCV(estimator=ovr_clf, 
                       param_grid=param_grid, 
                       cv=cv, 
                       scoring='accuracy', 
                       n_jobs=-1,  # Use all available cores for search
                       verbose=1)  # Set to 2 or 3 for more detailed output

    # Train the final model on all data
    # GridSearchCV automatically refits the best model on the entire dataset
    clf.fit(X_pca, labels)

    print(f"Best parameters found: {clf.best_params_}")
    print(f"Best cross-validation score: {clf.best_score_:.4f}")
    print("Training finished.")

    # The returned 'clf' is the fully tuned and fitted model
    return clf