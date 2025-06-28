# scpred_py/_prediction.py

import pandas as pd
import numpy as np

def predict_cells(classifier, X_projected_pca, threshold=0.0):
    """
    Predicts cell types using the trained classifier and applies a
    probability threshold.

    Args:
        classifier (sklearn.base.BaseEstimator): The trained classifier.
        X_projected_pca (np.ndarray): Projected PCA data for query cells.
        threshold (float): The minimum probability for a prediction to be
                           considered valid.

    Returns:
        tuple: (pd.Series, pd.DataFrame) - Final labels (with "unassigned")
               and prediction probabilities.
    """
    print("Predicting cell types...")
    
    # The CalibratedClassifierCV now supports predict_proba
    predicted_probs = classifier.predict_proba(X_projected_pca)
    
    # Get the highest probability for each cell
    max_probs = np.max(predicted_probs, axis=1)
    
    # Get the initial predicted labels (the class with the highest prob)
    initial_labels = classifier.classes_[np.argmax(predicted_probs, axis=1)]
    
    # Apply the threshold
    final_labels = pd.Series(initial_labels)
    final_labels[max_probs < threshold] = "unassigned"
    
    # Create a DataFrame for probabilities with class names
    prob_df = pd.DataFrame(predicted_probs, columns=classifier.classes_)
    
    print(f"Applied threshold of {threshold}. Found {(final_labels == 'unassigned').sum()} unassigned cells.")
    print("Prediction finished.")
    return final_labels, prob_df