# scpred_py/_prediction.py


import pandas as pd

import numpy as np


def predict_cells(classifier, X_projected_pca, threshold=0.0):

    """

    Predicts cell types using the trained classifier and applies a

    probability threshold.

    """

    print("Predicting cell types...")

   

    predicted_probs = classifier.predict_proba(X_projected_pca)

    max_probs = np.max(predicted_probs, axis=1)

   

    # Get class labels from the base estimator of the calibrator

    class_labels = classifier.estimator.classes_

    initial_labels = class_labels[np.argmax(predicted_probs, axis=1)]

   

    # Modify the numpy array directly. Use object dtype for strings.

    final_labels = initial_labels.astype(object)

    final_labels[max_probs < threshold] = "unassigned"

   

    # Create the probabilities DataFrame, but we will set the index later

    prob_df = pd.DataFrame(predicted_probs, columns=class_labels)

   

    print(f"Applied threshold of {threshold}. Found {(final_labels == 'unassigned').sum()} unassigned cells.")

    print("Prediction finished.")

   

    # Return the raw numpy array and the probs dataframe

    return final_labels, prob_df
