import pandas as pd
import numpy as np

def predict_cells(X_projected_pca, classifier, threshold=0.0):
    """
    Predicts cell types using the trained classifier, with an option for probability thresholding.

    Args:
        X_projected_pca (np.ndarray): Projected PCA data for query cells.
        classifier (sklearn.base.BaseEstimator): The trained classifier.
        threshold (float): Minimum probability for a prediction.
                           Predictions below this will be flagged as "unassigned".

    Returns:
        tuple: (np.ndarray, pd.DataFrame) - Predicted labels (as numpy array, possibly with "unassigned")
                                           and prediction probabilities (DataFrame).
    """
    print("Predicting cell types...")
    
    # Get raw predictions and probabilities
    predicted_labels_raw = classifier.predict(X_projected_pca)
    
    prob_df = None
    try:
        predicted_probs_array = classifier.predict_proba(X_projected_pca)
        # Create a DataFrame for probabilities with class names
        prob_df = pd.DataFrame(predicted_probs_array, columns=classifier.classes_)
    except AttributeError:
        print("Classifier does not support predict_proba. Probability thresholding will not be applied.")
        # If no probabilities, thresholding cannot be applied.
        threshold = 0.0 # Effectively disable thresholding

    # Apply probability thresholding
    final_predicted_labels = predicted_labels_raw.copy().astype(object) # Use object dtype to allow mixed types (str and original labels)
    
    if threshold > 0.0 and prob_df is not None:
        # For each cell, find the maximum probability and its corresponding predicted class
        max_probs = prob_df.max(axis=1)
        
        # Identify cells where max probability is below threshold
        low_confidence_mask = (max_probs < threshold)
        
        # Set predictions to "unassigned" for low-confidence cells
        final_predicted_labels[low_confidence_mask] = "unassigned"
        print(f"Applied probability thresholding: {np.sum(low_confidence_mask)} cells set to 'unassigned' (threshold={threshold}).")
    elif threshold > 0.0 and prob_df is None:
        print("Warning: Probability thresholding requested, but classifier does not support `predict_proba`. Thresholding skipped.")

    print("Prediction finished.")
    return final_predicted_labels, prob_df
