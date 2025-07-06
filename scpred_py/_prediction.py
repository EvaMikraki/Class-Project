# _prediction.py:

import pandas as pd

def predict_cells(X_projected_pca, classifier, query_obs_names, threshold=0.0): # Added query_obs_names
    """
    Predicts cell types using the trained classifier and applies a probability threshold.

    Args:
        X_projected_pca (np.ndarray): Projected PCA data for query cells.
        classifier (sklearn.base.BaseEstimator): The trained classifier.
        query_obs_names (pd.Index): The observation names (cell IDs) corresponding to X_projected_pca.
                                    This is crucial for correct indexing of output Series/DataFrames.
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
        # Use query_obs_names here to correctly index the probabilities DataFrame
        prob_df = pd.DataFrame(predicted_probs_array, columns=classifier.classes_, index=query_obs_names)
    except AttributeError:
        print("Classifier does not support predict_proba. Cannot apply threshold. Returning only labels.")
        # Ensure predicted_labels are indexed correctly even if no probabilities are available
        predicted_labels = classifier.predict(X_projected_pca)
        return pd.Series(predicted_labels, index=query_obs_names), None

    # Determine the predicted label based on the highest probability
    # predicted_labels will inherit the index from prob_df, which is now query_obs_names
    predicted_labels = prob_df.idxmax(axis=1) 
    
    # Get the maximum probability for each cell
    max_probs = prob_df.max(axis=1)

    # Apply thresholding: if max_prob < threshold, set to 'unassigned'
    unassigned_count = (max_probs < threshold).sum()
    if unassigned_count > 0:
        # Use .loc with the boolean mask to modify labels_series in place
        predicted_labels.loc[max_probs < threshold] = 'unassigned'
        print(f"Applied probability thresholding: {unassigned_count} cells set to 'unassigned' (threshold={threshold}).")
    else:
        print("No cells fell below the prediction threshold.")

    print("Prediction finished.")
    # Ensure labels are categorical and return both the Series and the DataFrame
    return predicted_labels.astype('category'), prob_df