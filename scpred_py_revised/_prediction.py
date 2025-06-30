# scpred_py/_prediction.py

import pandas as pd
import numpy as np

def predict_cells(X_projected_pca, classifier):
    """
    Predicts cell types using the trained classifier.

    Args:
        X_projected_pca (np.ndarray): Projected PCA data for query cells.
        classifier (sklearn.base.BaseEstimator): The trained classifier.

    Returns:
        tuple: (np.ndarray, pd.DataFrame) - Predicted labels (as numpy array) and prediction
               probabilities.
    """
    print("Predicting cell types...")
    predicted_labels = classifier.predict(X_projected_pca)
    
    try:
        predicted_probs = classifier.predict_proba(X_projected_pca)
        # Create a DataFrame for probabilities with class names
        prob_df = pd.DataFrame(predicted_probs, columns=classifier.classes_)
    except AttributeError:
        print("Classifier does not support predict_proba. Returning only labels.")
        prob_df = None

    print("Prediction finished.")
    return predicted_labels, prob_df