"""
Test data loading and preprocessing
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader
from src.data.preprocessing import TextPreprocessor


def test_data_loader():
    """Test data loading"""
    loader = DataLoader()
    train_df = loader.load_train_data()
    assert len(train_df) > 0, "Training data should not be empty"
    assert 'text' in train_df.columns, "Text column should exist"
    assert 'sentiment' in train_df.columns, "Sentiment column should exist"
    print("✓ Data loader test passed")


def test_preprocessor():
    """Test text preprocessing"""
    preprocessor = TextPreprocessor()
    
    # Test single text
    text = "This is a GREAT product! I love it."
    cleaned = preprocessor.clean_text(text)
    assert isinstance(cleaned, str), "Cleaned text should be string"
    assert len(cleaned) > 0, "Cleaned text should not be empty"
    
    # Test symbol counting
    symbol_count = preprocessor.count_symbols(text)
    assert symbol_count > 0, "Should detect symbols"
    
    print("✓ Preprocessor test passed")


if __name__ == "__main__":
    print("Running data tests...")
    test_data_loader()
    test_preprocessor()
    print("\nAll data tests passed!")
