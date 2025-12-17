"""
SVM Classifier for Material Stream Identification into 7 classes.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class MaterialSVMClassifier:
    """
    SVM-based classifier for material stream identification.
    
    Attributes:
        kernel (str): SVM kernel type ('linear', 'poly', 'rbf', 'sigmoid')
        C (float): Regularization parameter
        gamma (str or float): Kernel coefficient
        scaler (StandardScaler): Feature scaler for normalization
        svm (SVC): The trained SVM model
        classes (dict): Mapping of class IDs to names
        rejection_threshold (float): Confidence threshold for unknown class
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', rejection_threshold=0.6):
        """
        Initialize the SVM classifier.
        
        Args:
            kernel (str): SVM kernel type
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
            rejection_threshold (float): Minimum probability to accept classification, else class 6 (unknown)
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.rejection_threshold = rejection_threshold
        
        self.scaler = StandardScaler()
        
        self.svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            decision_function_shape="ovr",
            random_state=42
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
    
    def train(self, X_train, y_train, X_test=None, y_test=None, use_grid_search=False):
        """
        Train the SVM classifier on extracted features with optional hyperparameter tuning.
        
        Args:
            X_train (np.ndarray): Training features, shape (n_samples, n_features)
            y_train (np.ndarray): Training labels, shape (n_samples,)
            X_test (np.ndarray): Testing features
            y_test (np.ndarray): Testing labels
            use_grid_search (bool): Whether to perform grid search (slower but may find better params)
            
        Returns:
            dict: Training metrics and results
        """
        print(f"Training SVM Classifier")
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        
        # Scale features (fit on training data only)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Perform hyperparameter tuning if requested
        if use_grid_search:
            print("Performing hyperparameter tuning")
            best_params = self._grid_search(X_train_scaled, y_train)
            print(f"Best parameters found: {best_params}")
        else:
            print("Training with initial parameters")
            self.svm.fit(X_train_scaled, y_train)
        
        train_pred = self.svm.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        results = {
            'train_accuracy': train_acc,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
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
            'kernel': ['rbf'],
            'C': [1, 10],
            'gamma': ['scale']
        }
        
        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update classifier with best parameters
        self.kernel = grid_search.best_params_['kernel']
        self.C = grid_search.best_params_['C']
        self.gamma = grid_search.best_params_['gamma']
        
        self.svm = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def predict(self, X):
        """
        Predict class labels for samples with rejection mechanism.
        
        Args:
            X (np.ndarray): Features to classify, shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted labels
        """
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Get probability predictions
        probabilities = self.svm.predict_proba(X_scaled)
        
        # Get the maximum probability for each sample
        max_probs = np.max(probabilities, axis=1)
        
        # Predict classes
        predictions = self.svm.predict(X_scaled)
        
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
            'svm': self.svm,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
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
        
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.kernel = model_data['kernel']
        self.C = model_data['C']
        self.gamma = model_data['gamma']
        self.rejection_threshold = model_data['rejection_threshold']
        self.classes = model_data['classes']
        
        print(f"Model loaded from: {filepath}")
        print(f"Configuration: kernel={self.kernel}, C={self.C}, gamma={self.gamma}")


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
    print("SVM CLASSIFIER TRAINING")
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
        random_state=51,
        stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Initialize SVM classifier
    print("\nInitializing SVM Classifier")
    print("-" * 60)
    
    classifier = MaterialSVMClassifier(
        kernel='rbf',
        C=3,
        gamma='scale',
        rejection_threshold=0.6
    )

    print("\nTraining classifier")
    classifier.train(
        X_train, y_train,
        X_test, y_test,
        use_grid_search=False
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
    
    model_path = 'models/svm_final.pkl'
    classifier.save_model(model_path)
    
    print(f"\n Training complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Testing Accuracy: {accuracy:.4f}")
    print(f"  Configuration: kernel={classifier.kernel}, C={classifier.C}, gamma={classifier.gamma}")
    print("-" * 60)


if __name__ == "__main__":
    main()
