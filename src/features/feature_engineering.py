"""
Feature engineering module
Handles text vectorization and feature combination
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from typing import Tuple, Optional
import joblib


class FeatureEngineer:
    """Create features for sentiment analysis"""
    
    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0
    ):
        """
        Initialize FeatureEngineer
        
        Args:
            max_features: Maximum number of features for CountVectorizer
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = None
        self.label_encoder = None
        self.numeric_features = ['text_length', 'word_count', 'symbol_count']
    
    def fit_vectorizer(self, texts: pd.Series) -> 'FeatureEngineer':
        """
        Fit CountVectorizer on training texts
        
        Args:
            texts: Training texts
            
        Returns:
            self
        """
        print("Fitting CountVectorizer...")
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df
        )
        self.vectorizer.fit(texts)
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        return self
    
    def transform_text(self, texts: pd.Series):
        """
        Transform texts to bag-of-words features
        
        Args:
            texts: Input texts
            
        Returns:
            Sparse matrix of text features
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        return self.vectorizer.transform(texts)
    
    def fit_label_encoder(self, labels: pd.Series) -> 'FeatureEngineer':
        """
        Fit label encoder on training labels
        
        Args:
            labels: Training labels
            
        Returns:
            self
        """
        print("Fitting LabelEncoder...")
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        print(f"Classes: {list(self.label_encoder.classes_)}")
        return self
    
    def transform_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Transform labels to numeric format
        
        Args:
            labels: Input labels
            
        Returns:
            Encoded labels
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call fit_label_encoder first.")
        return self.label_encoder.transform(labels)
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original format
        
        Args:
            encoded_labels: Encoded labels
            
        Returns:
            Original labels
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted.")
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def create_features(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        fit: bool = False
    ):
        """
        Create complete feature matrix
        
        Args:
            df: Input dataframe with text and numeric features
            text_column: Name of text column
            fit: Whether to fit vectorizer (True for training data)
            
        Returns:
            Sparse feature matrix
        """
        # Transform text to BOW
        if fit:
            self.fit_vectorizer(df[text_column])
        
        text_features = self.transform_text(df[text_column])
        
        # Get numeric features
        numeric_features = df[self.numeric_features].values
        
        # Combine features
        X = hstack((text_features, numeric_features))
        
        # Convert to CSR format for efficiency
        X = X.tocsr()
        
        print(f"Feature matrix shape: {X.shape}")
        return X
    
    def save_artifacts(self, vectorizer_path: str, label_encoder_path: str):
        """
        Save vectorizer and label encoder
        
        Args:
            vectorizer_path: Path to save vectorizer
            label_encoder_path: Path to save label encoder
        """
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, vectorizer_path)
            print(f"Vectorizer saved to {vectorizer_path}")
        
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, label_encoder_path)
            print(f"Label encoder saved to {label_encoder_path}")
    
    def load_artifacts(self, vectorizer_path: str, label_encoder_path: str):
        """
        Load vectorizer and label encoder
        
        Args:
            vectorizer_path: Path to vectorizer
            label_encoder_path: Path to label encoder
        """
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Vectorizer loaded from {vectorizer_path}")
        
        self.label_encoder = joblib.load(label_encoder_path)
        print(f"Label encoder loaded from {label_encoder_path}")
