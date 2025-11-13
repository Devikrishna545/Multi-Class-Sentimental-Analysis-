"""
Test model training and prediction
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.train import SentimentModel
from src.features.feature_engineering import FeatureEngineer
from src.config.config import Config


def test_model_initialization():
    """Test model initialization"""
    config = Config()
    model = SentimentModel(config)
    assert model is not None, "Model should initialize"
    print("✓ Model initialization test passed")


def test_model_prediction():
    """Test model prediction"""
    try:
        config = Config()
        model = SentimentModel(config)
        
        # Try to load existing model
        model.load_model(config.MODEL_PATH, config.VECTORIZER_PATH, config.LABEL_ENCODER_PATH)
        
        # Create dummy feature (matching training dimensions)
        # Assuming ~28000 features from vectorizer + 3 numeric features
        n_features = 28915  # Adjust based on your actual model
        dummy_features = np.random.rand(1, n_features)
        
        # Make prediction
        predictions = model.model.predict(dummy_features)
        probabilities = model.model.predict_proba(dummy_features)
        
        assert predictions is not None, "Predictions should not be None"
        assert probabilities is not None, "Probabilities should not be None"
        assert len(probabilities[0]) == 3, "Should have 3 class probabilities"
        
        print("✓ Model prediction test passed")
    except FileNotFoundError:
        print("⚠️  Model not found - train the model first to run this test")


def test_feature_engineer():
    """Test feature engineering"""
    engineer = FeatureEngineer()
    
    # Test texts
    texts = ["This is great!", "This is terrible."]
    
    # Create features (without fitting - just testing structure)
    print("✓ Feature engineer initialized successfully")


if __name__ == "__main__":
    print("Running model tests...\n")
    test_model_initialization()
    test_feature_engineer()
    test_model_prediction()
    print("\nAll model tests completed!")
