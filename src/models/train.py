"""
Model training module
Handles model creation, training, and evaluation
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any
import joblib


class SentimentModel:
    """Sentiment analysis model wrapper"""
    
    def __init__(self, model_params: Dict[str, Any]):
        """
        Initialize model
        
        Args:
            model_params: Dictionary of model hyperparameters
        """
        self.model_params = model_params
        self.model = None
        self.is_trained = False
    
    def build_model(self):
        """Build the logistic regression model"""
        print("Building Logistic Regression model...")
        self.model = LogisticRegression(**self.model_params)
        print(f"Model parameters: {self.model_params}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.model is None:
            self.build_model()
        
        print(f"Training on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, label_names=None) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            label_names: Names of classes for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        if label_names is None:
            label_names = ['negative', 'neutral', 'positive']
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        results = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test, y_pred, target_names=label_names, output_dict=True
            )
        }
        
        return results
    
    def save_model(self, model_path: str):
        """
        Save trained model
        
        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load trained model
        
        Args:
            model_path: Path to model file
        """
        self.model = joblib.load(model_path)
        self.is_trained = True
        print(f"Model loaded from {model_path}")
