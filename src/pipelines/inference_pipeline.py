"""
Inference pipeline
Make predictions on new text data
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Union, Tuple
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config import Config
from src.data.preprocessing import TextPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train import SentimentModel


class InferencePipeline:
    """Pipeline for making predictions on new text"""
    
    def __init__(self, config: Config = None):
        """
        Initialize inference pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize components
        self.preprocessor = TextPreprocessor(
            remove_stopwords=self.config.REMOVE_STOPWORDS,
            lowercase=self.config.LOWERCASE
        )
        self.feature_engineer = FeatureEngineer()
        self.model = SentimentModel({})
        
        # Load trained artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained model and feature engineering artifacts"""
        print("Loading trained artifacts...")
        
        # Load feature engineering artifacts
        self.feature_engineer.load_artifacts(
            str(self.config.VECTORIZER_PATH),
            str(self.config.LABEL_ENCODER_PATH)
        )
        
        # Load trained model
        self.model.load_model(str(self.config.MODEL_PATH))
        
        print("All artifacts loaded successfully!")
    
    def preprocess_texts(self, texts: Union[str, List[str]]) -> pd.DataFrame:
        """
        Preprocess input texts
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            DataFrame with processed texts and features
        """
        # Convert to list if single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Create DataFrame
        df = pd.DataFrame({'text': texts, 'text_original': texts})
        
        # Add text features (before cleaning)
        df = self.preprocessor.add_text_features(df, 'text_original')
        
        # Clean text
        df['text'] = self.preprocessor.clean_texts(df['text'])
        
        return df
    
    def predict(
        self,
        texts: Union[str, List[str]],
        return_proba: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts
        
        Args:
            texts: Single text or list of texts
            return_proba: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        # Preprocess texts
        df = self.preprocess_texts(texts)
        
        # Create features
        X = self.feature_engineer.create_features(df, text_column='text', fit=False)
        
        # Make predictions
        predictions_encoded = self.model.predict(X)
        predictions = self.feature_engineer.inverse_transform_labels(predictions_encoded)
        
        # Get probabilities
        probabilities = None
        if return_proba:
            probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_with_details(self, texts: Union[str, List[str]]) -> List[dict]:
        """
        Make predictions with detailed output
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of prediction dictionaries
        """
        # Convert to list if single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Get predictions and probabilities
        predictions, probabilities = self.predict(texts, return_proba=True)
        
        # Preprocess for cleaned text
        df = self.preprocess_texts(texts)
        
        # Create detailed results
        results = []
        for i, text in enumerate(texts):
            result = {
                'original_text': text,
                'cleaned_text': df['text'].iloc[i],
                'predicted_sentiment': predictions[i],
                'confidence': float(np.max(probabilities[i])),
                'probabilities': {
                    'negative': float(probabilities[i][0]),
                    'neutral': float(probabilities[i][1]),
                    'positive': float(probabilities[i][2])
                },
                'features': {
                    'text_length': int(df['text_length'].iloc[i]),
                    'word_count': int(df['word_count'].iloc[i]),
                    'symbol_count': int(df['symbol_count'].iloc[i])
                }
            }
            results.append(result)
        
        return results
    
    def predict_and_display(self, texts: Union[str, List[str]]):
        """
        Make predictions and display results
        
        Args:
            texts: Single text or list of texts
        """
        results = self.predict_with_details(texts)
        
        print("\n" + "="*70)
        print(" "*20 + "SENTIMENT PREDICTION RESULTS")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Prediction {i} ---")
            print(f"Original: '{result['original_text']}'")
            print(f"Cleaned:  '{result['cleaned_text']}'")
            print(f"\nPredicted Sentiment: {result['predicted_sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            print(f"\nProbability Distribution:")
            print(f"  Negative: {result['probabilities']['negative']:.4f}")
            print(f"  Neutral:  {result['probabilities']['neutral']:.4f}")
            print(f"  Positive: {result['probabilities']['positive']:.4f}")
            print(f"\nText Features:")
            print(f"  Length: {result['features']['text_length']}")
            print(f"  Words:  {result['features']['word_count']}")
            print(f"  Symbols: {result['features']['symbol_count']}")
        
        print("\n" + "="*70)
        
        return results


if __name__ == "__main__":
    # Example usage
    pipeline = InferencePipeline()
    
    # Sample texts
    sample_texts = [
        "The product was excellent and exceeded my expectations!",
        "This movie is terrible, I hate it.",
        "The service was okay, nothing special."
    ]
    
    # Make predictions with display
    results = pipeline.predict_and_display(sample_texts)
