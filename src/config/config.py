"""
Configuration file for sentiment analysis project
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

class Config:
    """Configuration class for the sentiment analysis pipeline"""
    
    # Project root directory
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    # Data paths
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models_output"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # Hugging Face dataset
    HF_DATASET = "Sp1786/multiclass-sentiment-analysis-dataset"
    TRAIN_SPLIT = "train"
    VALIDATION_SPLIT = "validation"
    TEST_SPLIT = "test"
    
    # Model artifacts
    MODEL_NAME = "logistic_regression_countvec_model.pkl"
    VECTORIZER_NAME = "count_vectorizer.pkl"
    LABEL_ENCODER_NAME = "label_encoder.pkl"
    
    MODEL_PATH = MODELS_DIR / MODEL_NAME
    VECTORIZER_PATH = MODELS_DIR / VECTORIZER_NAME
    LABEL_ENCODER_PATH = MODELS_DIR / LABEL_ENCODER_NAME
    
    # Feature engineering parameters
    MAX_FEATURES = None  # Use all features from CountVectorizer
    NGRAM_RANGE = (1, 1)  # Unigrams only
    MIN_DF = 1
    MAX_DF = 1.0
    
    # Text preprocessing
    REMOVE_STOPWORDS = True
    LOWERCASE = True
    REMOVE_PUNCTUATION = True
    
    # Model hyperparameters (from hyperparameter tuning)
    MODEL_TYPE = "LogisticRegression"
    C = 1
    PENALTY = "l2"
    SOLVER = "liblinear"
    MAX_ITER = 2000
    RANDOM_STATE = 42
    CLASS_WEIGHT = None
    
    # Training parameters
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    CV_FOLDS = 3
    
    # Numeric features to include
    NUMERIC_FEATURES = ['text_length', 'word_count', 'symbol_count']
    
    # Sentiment labels
    SENTIMENT_LABELS = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }
    
    # Logging
    LOG_LEVEL = "INFO"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_params(cls):
        """Return model hyperparameters as a dictionary"""
        return {
            'C': cls.C,
            'penalty': cls.PENALTY,
            'solver': cls.SOLVER,
            'max_iter': cls.MAX_ITER,
            'random_state': cls.RANDOM_STATE,
            'class_weight': cls.CLASS_WEIGHT
        }
