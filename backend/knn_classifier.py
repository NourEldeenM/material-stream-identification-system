"""
KNN Classifier for Material Stream Identification into 7 classes.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class MaterialKNNClassifier:
    """
    KNN-based classifier for material stream identification.

    Attributes:
        n_neighbors (int): Number of neighbors to use
        weights (str): Weight function ('uniform' or 'distance')
        metric (str): Distance metric to use ('Euclidean', 'Manhattan', 'Minkowski')
        scaler (StandardScaler): Feature scaler for normalization
        knn (KNeighborsClassifier): The trained KNN model
        classes (dict): Mapping of class IDs to names
        rejection_threshold (float): Confidence threshold for unknown class
    """

    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean',
                 rejection_threshold=0.6):
        """
        Initialize the KNN classifier.

        Args:
            n_neighbors (int): Number of neighbors to consider
            weights (str): 'uniform' or 'distance' - how to weight neighbors
            metric (str): Distance metric ('Euclidean', 'Manhattan', 'Minkowski')
            rejection_threshold (float): Minimum probability to accept classification, else class 6 (unknown)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.rejection_threshold = rejection_threshold

        self.scaler = StandardScaler()

        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1
        )

        self.classes = {
            0: "Glass",
            1: "Paper",
            2: "Cardboard",
            3: "Plastic",
            4: "Metal",
            5: "Trash",
            6: "Unknown"
        }

    def train(self, X_train, y_train, X_test=None, y_test=None):
        """
        Train the KNN classifier on extracted features with hyperparameter tuning.

        Args:
            X_train (np.ndarray): Training features, shape (n_samples, n_features)
            y_train (np.ndarray): Training labels, shape (n_samples,)
            X_test (np.ndarray): Testing features
            y_test (np.ndarray): Testing labels

        Returns:
            dict: Training metrics and results
        """
        print(f"Training KNN Classifier")
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

        # Scale features (fit on training data only)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Perform hyperparameter tuning
        print("Performing hyperparameter tuning")
        best_params = self._grid_search(X_train_scaled, y_train)
        print(f"Best parameters found: {best_params}")

        train_pred = self.knn.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)

        results = {
            'train_accuracy': train_acc,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric
        }

        print(f"Training Accuracy: {train_acc:.4f}")

        # Evaluate on testing set
        if X_test is not None and y_test is not None:
            test_acc = self.evaluate(X_test, y_test)
            results['test_accuracy'] = test_acc
            print(f"Testing Accuracy: {test_acc:.4f}")

        return results

    def _grid_search(self, X_train, y_train):
        """
        Perform grid search for hyperparameter tuning.

        Args:
            X_train (np.ndarray): Scaled training features
            y_train (np.ndarray): Training labels

        Returns:
            dict: Best parameters found
        """
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        grid_search = GridSearchCV(
            KNeighborsClassifier(n_jobs=-1),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Update classifier with best parameters
        self.n_neighbors = grid_search.best_params_['n_neighbors']
        self.weights = grid_search.best_params_['weights']
        self.metric = grid_search.best_params_['metric']

        self.knn = grid_search.best_estimator_

        return grid_search.best_params_

    def predict(self, X):
        """
        Predict class labels for samples with rejection mechanism.

        Args:
            X (np.ndarray): Features to classify, shape (n_samples, n_features)
            return_confidence (bool): Whether to return confidence scores

        Returns:
            np.ndarray or tuple: Predicted labels
        """

        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)

        # Get probability predictions
        probabilities = self.knn.predict_proba(X_scaled)

        # Get the maximum probability for each sample
        max_probs = np.max(probabilities, axis=1)

        # Predict classes
        predictions = self.knn.predict(X_scaled)

        # Apply rejection threshold - classify as Unknown (class 6) if confidence too low
        predictions = np.where(max_probs < self.rejection_threshold, 6, predictions)

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on test data.

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True labels

        Returns:
            float: accuracy
        """
        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def save_model(self, filepath):
        """
        Save the trained model to disk.

        Args:
            filepath (str): Path to save the model
        """

        model_data = {
            'knn': self.knn,
            'scaler': self.scaler,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric,
            'rejection_threshold': self.rejection_threshold,
            'classes': self.classes,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.knn = model_data['knn']
        self.scaler = model_data['scaler']
        self.n_neighbors = model_data['n_neighbors']
        self.weights = model_data['weights']
        self.metric = model_data['metric']
        self.rejection_threshold = model_data['rejection_threshold']
        self.classes = model_data['classes']

        print(f"Model loaded from: {filepath}")
        print(f"Configuration: k={self.n_neighbors}, weights={self.weights}, metric={self.metric}")


def load_extracted_features(features_path):
    """
    Load the extracted features from disk.

    Args:
        features_path (str): Path to the saved features file

    Returns:
        tuple: (X, y) - features and labels
    """
    with open(features_path, 'rb') as f:
        data = pickle.load(f)

    X = data['features']
    y = data['labels']

    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each")

    return X, y


def main():
    print("KNN CLASSIFIER TRAINING")
    print("-" * 60)

    # Load extracted features
    features_path = 'features/extracted_features.pkl'

    print(f"\nLoading features from: {features_path}")
    X, y = load_extracted_features(features_path)

    # Split data into train and testing sets
    print("\nSplitting data into 80% train and 20% testing")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Initialize KNN classifier
    print("Initializing KNN Classifier")
    print("-" * 60)

    classifier = MaterialKNNClassifier(
        n_neighbors=3,
        weights='distance',
        metric='manhattan',
        rejection_threshold=0.0
    )

    # Train the model with hyperparameter tuning
    print("\nTraining classifier")
    classifier.train(
        X_train, y_train,
        X_test, y_test
    )

    print("-" * 60)
    print("EVALUATION ON TESTING SET")
    print("-" * 60)
    accuracy = classifier.evaluate(X_test, y_test)
    print(f"Testing Accuracy: {accuracy:.4f}")

    # Save the trained model
    print("\n" + "-" * 60)
    print("SAVING MODEL")
    print("-" * 60)

    model_path = 'models/knn_final.pkl'
    classifier.save_model(model_path)

    print(f"\n Training complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Testing Accuracy: {accuracy:.4f}")
    print(f"  Configuration: k={classifier.n_neighbors}, weights={classifier.weights}, metric={classifier.metric}")
    print("-" * 60)


if __name__ == "__main__":
    main()
