import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import mannwhitneyu

class scPred:
    """
    A Python implementation of the scPred algorithm for single-cell classification.

    This class follows the methodology described in:
    Alquicira-Hernandez, J., et al. (2019). scPred: accurate supervised method for cell-type
    classification from single-cell RNA-seq data. Genome Biology, 20(264).

    The workflow includes:
    1.  Data preprocessing (CPM and log2 transformation).
    2.  Scaling and dimensionality reduction using Principal Component Analysis (PCA) on a training dataset.
    3.  Selection of informative principal components (PCs) based on the Wilcoxon rank-sum test.
    4.  Training a one-vs-rest Support Vector Machine (SVM) classifier on the informative PCs.
    5.  Projection of new data onto the trained PC space and prediction of cell types.
    """

    def __init__(self, probability_threshold=0.9):
        """
        Initializes the scPred model.

        Args:
            probability_threshold (float): The minimum probability required to assign a cell
                                           to a class. Cells with a max probability below
                                           this threshold will be labeled 'Unassigned'.
        """
        self.scaler = None
        self.pca = None
        self.classifier = None
        self.informative_pcs = None
        self.class_labels = None
        self.probability_threshold = probability_threshold

    def _cpm_transform(self, data):
        """Transforms count data to Counts Per Million (CPM)."""
        # Ensure data is float for division
        data = data.astype(float)
        total_counts = np.sum(data, axis=1, keepdims=True)
        # Avoid division by zero for cells with no counts
        total_counts[total_counts == 0] = 1
        return (data / total_counts) * 1_000_000

    def _log2_transform(self, data):
        """Applies log2(x + 1) transformation."""
        return np.log2(data + 1)

    def train(self, X_train, y_train, p_value_threshold=0.05, variance_threshold=0.0001, perform_hpt=False):
        """
        Trains the scPred model on a reference dataset.

        Args:
            X_train (np.ndarray): The training gene expression matrix (cells x genes).
                                  Assumed to be raw counts.
            y_train (np.ndarray or list): The cell type labels for the training data.
            p_value_threshold (float): The p-value cutoff for selecting informative PCs.
            variance_threshold (float): The minimum variance a PC must explain to be considered.
            perform_hpt (bool): If True, performs hyperparameter tuning for the SVM.
                                This can be time-consuming.
        """
        print("Starting training process...")

        # --- 1. Preprocessing ---
        print("Step 1: Preprocessing data (CPM, Log2)...")
        X_cpm = self._cpm_transform(X_train)
        X_log = self._log2_transform(X_cpm)
        
        # --- 2. Scaling and PCA ---
        print("Step 2: Scaling data and performing PCA...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_log)

        # The paper uses SVD, but PCA from scikit-learn is a convenient implementation
        # that centers the data and provides the principal components (scores).
        self.pca = PCA()
        pc_scores = self.pca.fit_transform(X_scaled)

        # --- 3. Feature Selection (Informative PCs) ---
        print("Step 3: Selecting informative PCs...")
        
        # Filter PCs by explained variance
        explained_variance = self.pca.explained_variance_ratio_
        significant_variance_indices = np.where(explained_variance > variance_threshold)[0]
        pc_scores_filtered = pc_scores[:, significant_variance_indices]
        
        self.class_labels = np.unique(y_train)
        informative_pc_union = set()

        # Perform Wilcoxon rank-sum test for each class in a one-vs-rest manner
        for cell_type in self.class_labels:
            print(f"  - Finding informative PCs for class: {cell_type}")
            in_class_mask = (y_train == cell_type)
            
            p_values = []
            for i in range(pc_scores_filtered.shape[1]):
                pc_col = pc_scores_filtered[:, i]
                group1 = pc_col[in_class_mask]
                group2 = pc_col[~in_class_mask]
                
                # The Wilcoxon test requires non-identical samples
                if len(np.unique(group1)) > 1 and len(np.unique(group2)) > 1:
                    stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    p_values.append(p_val)
                else:
                    p_values.append(1.0) # Assign a non-significant p-value

            # Note: The paper mentions Benjamini-Hochberg correction. For simplicity,
            # this is omitted here but could be added using statsmodels.
            significant_pcs_for_class = np.where(np.array(p_values) < p_value_threshold)[0]
            informative_pc_union.update(significant_variance_indices[significant_pcs_for_class])

        self.informative_pcs = sorted(list(informative_pc_union))
        
        if not self.informative_pcs:
            raise ValueError("No informative PCs found. Try relaxing the p-value or variance thresholds.")
            
        print(f"Found {len(self.informative_pcs)} unique informative PCs across all classes.")
        
        X_train_final = pc_scores[:, self.informative_pcs]
        
        # --- 4. Model Training ---
        print("Step 4: Training the SVM classifier...")
        
        # Using a one-vs-rest strategy with an SVM, as described in the paper and proposal
        svm = SVC(probability=True, kernel='rbf')
        
        if perform_hpt:
            print("  - Performing hyperparameter tuning (GridSearchCV)...")
            # This is a basic grid. For real data, a wider search may be needed.
            param_grid = {'estimator__C': [0.1, 1, 10], 'estimator__gamma': ['scale', 'auto']}
            self.classifier = GridSearchCV(OneVsRestClassifier(svm), param_grid, cv=3)
        else:
            # Use default parameters if not tuning
            self.classifier = OneVsRestClassifier(SVC(probability=True, kernel='rbf'))
            
        self.classifier.fit(X_train_final, y_train)
        
        print("Training complete.")

    def predict(self, X_test):
        """
        Predicts cell types for a new dataset.

        Args:
            X_test (np.ndarray): The query gene expression matrix (cells x genes).
                                 Assumed to be raw counts.

        Returns:
            pd.DataFrame: A DataFrame with predicted labels, max probabilities, and assigned class.
        """
        if self.scaler is None or self.pca is None or self.classifier is None:
            raise RuntimeError("The model has not been trained yet. Call train() first.")

        print("Starting prediction process...")
        
        # --- 1. Preprocessing (using stored training parameters) ---
        print("Step 1: Preprocessing query data...")
        X_cpm = self._cpm_transform(X_test)
        X_log = self._log2_transform(X_cpm)
        
        # --- 2. Scaling and Projection ---
        print("Step 2: Scaling and projecting data onto trained PC space...")
        X_scaled = self.scaler.transform(X_log)
        pc_scores_test = self.pca.transform(X_scaled)
        
        # --- 3. Prediction ---
        print("Step 3: Predicting class probabilities...")
        X_test_final = pc_scores_test[:, self.informative_pcs]
        probabilities = self.classifier.predict_proba(X_test_final)
        
        # --- 4. Assigning Labels ---
        print("Step 4: Assigning final labels...")
        max_probs = np.max(probabilities, axis=1)
        pred_indices = np.argmax(probabilities, axis=1)
        
        # Map indices to class labels
        predicted_labels = self.class_labels[pred_indices]
        
        # Apply rejection threshold
        final_labels = np.where(max_probs >= self.probability_threshold, predicted_labels, "Unassigned")
        
        # Create a results dataframe
        results_df = pd.DataFrame({
            'Predicted_Label': final_labels,
            'Max_Probability': max_probs,
            'Assigned_Class': predicted_labels # The class with the highest probability before thresholding
        })
        
        # Add probability columns for each class
        for i, label in enumerate(self.class_labels):
            results_df[f'Prob_{label}'] = probabilities[:, i]
            
        print("Prediction complete.")
        return results_df


if __name__ == '__main__':
    # --- Example Usage with Dummy Data ---
    # This example demonstrates the workflow. Replace this with your actual data loading.
    
    print("--- scPred Python Implementation Demo ---")
    
    # 1. Generate dummy training and testing data
    # In a real scenario, you would load your AnnData or CSV files here.
    n_train_cells, n_test_cells, n_genes = 500, 100, 1000
    
    # Training data
    X_train_dummy = np.random.randint(0, 100, size=(n_train_cells, n_genes))
    y_train_dummy = np.random.choice(['T-cell', 'B-cell', 'Macrophage'], size=n_train_cells)
    
    # Testing data
    X_test_dummy = np.random.randint(0, 100, size=(n_test_cells, n_genes))
    
    print(f"\nGenerated dummy data: {n_train_cells} training cells, {n_test_cells} test cells, {n_genes} genes.")
    print(f"Training classes: {np.unique(y_train_dummy)}")
    
    # 2. Initialize and train the model
    # Set perform_hpt=True to run hyperparameter tuning (slower).
    model = scPred()
    model.train(X_train_dummy, y_train_dummy, perform_hpt=False)
    
    # 3. Predict on new data
    predictions = model.predict(X_test_dummy)
    
    # 4. Display results
    print("\n--- Prediction Results ---")
    print(predictions.head())
    
    print("\nDistribution of predicted labels:")
    print(predictions['Predicted_Label'].value_counts())
    
    print("\nDemo finished.")