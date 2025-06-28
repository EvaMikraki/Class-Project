# scpred_py/_training.py


from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.calibration import CalibratedClassifierCV

import numpy as np


def train_svm(X_pca, labels):

    """

    Trains a One-vs-Rest Linear SVM, uses GridSearchCV for tuning,

    and then calibrates the model to produce probabilities.


    Args:

        X_pca (np.ndarray): PCA-transformed data (cells x PCs).

        labels (pd.Series or np.ndarray): Cell type labels for each cell.


    Returns:

        sklearn.calibration.CalibratedClassifierCV: The trained, tuned, and

                                                    calibrated classifier.

    """

    print("Training and calibrating LINEAR SVM with hyperparameter tuning...")


    # Define the parameter grid for the C parameter of the underlying SVM

    param_grid = {'estimator__C': [0.01, 0.1, 1, 10]}


    # Set up the base estimator and the OneVsRest wrapper

    svm = LinearSVC(random_state=42, dual=False, max_iter=2000)

    ovr_clf = OneVsRestClassifier(svm, n_jobs=-1)


    # Set up GridSearchCV to find the best C for the OneVsRest(LinearSVC)

    cv = StratifiedKFold(n_splits=3)

    grid_search = GridSearchCV(estimator=ovr_clf,

                               param_grid=param_grid,

                               cv=cv,

                               scoring='accuracy',

                               n_jobs=-1,

                               verbose=1)

   

    # Fit the grid search to find the best model

    grid_search.fit(X_pca, labels)


    print(f"Best C parameter found: {grid_search.best_params_}")

    print(f"Best internal cross-validation score: {grid_search.best_score_:.4f}")

   

    # Get the best estimator found by the grid search.

    # This is a OneVsRestClassifier with the optimal LinearSVC.

    best_ovr_clf = grid_search.best_estimator_

   

    # Now, calibrate this best estimator to enable predict_proba.

    # The 'prefit' option is crucial as it uses the already trained model.

    calibrated_clf = CalibratedClassifierCV(best_ovr_clf, cv="prefit", method='isotonic')

    calibrated_clf.fit(X_pca, labels)


    print("Training and calibration finished.")

    return calibrated_clf