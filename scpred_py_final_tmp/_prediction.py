import pandas as pd
import numpy as np

def predict_cells(X_projected_pca, classifier, threshold=0.0):
    """
    Predicts cell types using the trained classifier and applies a probability threshold.

    Args:
        X_projected_pca (np.ndarray): Projected PCA data for query cells.
        classifier (sklearn.base.BaseEstimator): The trained classifier.
        threshold (float): Minimum probability for a prediction.
                           Predictions below this are set to 'unassigned'.

    Returns:
        tuple: (pd.Series, pd.DataFrame) - Predicted labels and prediction
               probabilities.
    """
    print("Predicting cell types...")
    
    # Get probabilities first
    try:
        predicted_probs_array = classifier.predict_proba(X_projected_pca)
        prob_df = pd.DataFrame(predicted_probs_array, columns=classifier.classes_)
    except AttributeError:
        print("Classifier does not support predict_proba. Cannot apply threshold. Returning only labels.")
        predicted_labels = classifier.predict(X_projected_pca)
        return pd.Series(predicted_labels), None

    # Determine the predicted label based on the highest probability
    # If the threshold is 0.0, this is equivalent to simple argmax
    predicted_labels = prob_df.idxmax(axis=1)
    
    # Get the maximum probability for each cell
    max_probs = prob_df.max(axis=1)

    # Apply thresholding: if max_prob < threshold, set to 'unassigned'
    unassigned_count = (max_probs < threshold).sum()
    if unassigned_count > 0:
        predicted_labels.loc[max_probs < threshold] = 'unassigned'
        print(f"Applied probability thresholding: {unassigned_count} cells set to 'unassigned' (threshold={threshold}).")
    else:
        print("No cells fell below the prediction threshold.")

    print("Prediction finished.")
    return predicted_labels.astype('category'), prob_df # Ensure labels are categorical
