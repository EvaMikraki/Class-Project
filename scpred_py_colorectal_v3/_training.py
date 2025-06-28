# scpred_py/_training.py

from sklearn.svm import SVC  # Changed from LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def train_svm(X_pca, labels, kernel='linear'):
    """
    Trains a One-vs-Rest SVM, uses GridSearchCV for tuning,
    and then calibrates the model to produce probabilities.

    Args:
        X_pca (np.ndarray): PCA-transformed data (cells x PCs).
        labels (pd.Series or np.ndarray): Cell type labels for each cell.
        kernel (str): The SVM kernel to use ('linear' or 'rbf').

    Returns:
        sklearn.calibration.CalibratedClassifierCV: The trained, tuned, and
                                                    calibrated classifier.
    """
    print(f"Training and calibrating SVM with '{kernel}' kernel and hyperparameter tuning...")

    # --- NEW: Define parameter grid based on the chosen kernel ---
    if kernel == 'linear':
        param_grid = {'estimator__C': [0.01, 0.1, 1, 10, 100]}
    elif kernel == 'rbf':
        param_grid = {'estimator__C': [0.1, 1, 10],
                      'estimator__gamma': ['scale', 'auto']}
    else:
        raise ValueError("Unsupported kernel. Choose 'linear' or 'rbf'.")

    # Use the general SVC classifier which supports multiple kernels
    svm = SVC(kernel=kernel, random_state=42, probability=True, max_iter=-1) # probability=True is needed later
    ovr_clf = OneVsRestClassifier(svm, n_jobs=-1)

    # Set up GridSearchCV to find the best parameters
    cv = StratifiedKFold(n_splits=3)
    grid_search = GridSearchCV(estimator=ovr_clf,
                               param_grid=param_grid,
                               cv=cv,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=1)
    
    # Fit the grid search to find the best model
    grid_search.fit(X_pca, labels)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best internal cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best estimator found by the grid search.
    best_ovr_clf = grid_search.best_estimator_
    
    # Now, calibrate this best estimator to refine probabilities.
    # The 'prefit' option is crucial as it uses the already trained model.
    calibrated_clf = CalibratedClassifierCV(best_ovr_clf, cv="prefit", method='isotonic')
    calibrated_clf.fit(X_pca, labels)

    print("Training and calibration finished.")
    return calibrated_clf