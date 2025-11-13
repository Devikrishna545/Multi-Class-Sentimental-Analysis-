"""
Training pipeline
End-to-end pipeline for training the sentiment analysis model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessing import TextPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train import SentimentModel


class TrainingPipeline:
    """Complete training pipeline for sentiment analysis"""
    
    def __init__(self, config: Config = None):
        """
        Initialize training pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.config.create_directories()
        
        # Initialize components
        self.data_loader = DataLoader(self.config.HF_DATASET)
        self.preprocessor = TextPreprocessor(
            remove_stopwords=self.config.REMOVE_STOPWORDS,
            lowercase=self.config.LOWERCASE
        )
        self.feature_engineer = FeatureEngineer(
            max_features=self.config.MAX_FEATURES,
            ngram_range=self.config.NGRAM_RANGE,
            min_df=self.config.MIN_DF,
            max_df=self.config.MAX_DF
        )
        self.model = SentimentModel(self.config.get_model_params())
        
        # Data placeholders
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def load_data(self):
        """Load and clean data"""
        print("\n" + "="*50)
        print("STEP 1: LOADING DATA")
        print("="*50)
        
        self.train_df = self.data_loader.load_train_data()
        self.test_df = self.data_loader.load_test_data()
        
        # Clean data
        self.train_df = self.data_loader.clean_dataframe(self.train_df)
        self.test_df = self.data_loader.clean_dataframe(self.test_df)
        
        print(f"Training samples: {len(self.train_df)}")
        print(f"Test samples: {len(self.test_df)}")
    
    def preprocess_data(self):
        """Preprocess text and add features"""
        print("\n" + "="*50)
        print("STEP 2: PREPROCESSING DATA")
        print("="*50)
        
        # Add text features (before cleaning)
        self.train_df = self.preprocessor.add_text_features(self.train_df)
        self.test_df = self.preprocessor.add_text_features(self.test_df)
        
        # Clean text
        print("Cleaning training texts...")
        self.train_df['text'] = self.preprocessor.clean_texts(self.train_df['text'])
        
        print("Cleaning test texts...")
        self.test_df['text'] = self.preprocessor.clean_texts(self.test_df['text'])
    
    def engineer_features(self):
        """Create features from text and numeric data"""
        print("\n" + "="*50)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*50)
        
        # Fit and transform on training data
        self.X_train = self.feature_engineer.create_features(
            self.train_df,
            text_column='text',
            fit=True
        )
        
        # Transform test data (no fitting)
        self.X_test = self.feature_engineer.create_features(
            self.test_df,
            text_column='text',
            fit=False
        )
        
        # Encode labels
        self.feature_engineer.fit_label_encoder(self.train_df['sentiment'])
        self.y_train = self.feature_engineer.transform_labels(self.train_df['sentiment'])
        self.y_test = self.feature_engineer.transform_labels(self.test_df['sentiment'])
        
        print(f"Training features: {self.X_train.shape}")
        print(f"Test features: {self.X_test.shape}")
    
    def train_model(self):
        """Train the sentiment model"""
        print("\n" + "="*50)
        print("STEP 4: TRAINING MODEL")
        print("="*50)
        
        self.model.train(self.X_train, self.y_train)
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "="*50)
        print("STEP 5: EVALUATING MODEL")
        print("="*50)
        
        results = self.model.evaluate(
            self.X_test,
            self.y_test,
            label_names=list(self.config.SENTIMENT_LABELS.values())
        )
        
        return results
    
    def save_artifacts(self):
        """Save model and feature engineering artifacts"""
        print("\n" + "="*50)
        print("STEP 6: SAVING ARTIFACTS")
        print("="*50)
        
        # Save model
        self.model.save_model(str(self.config.MODEL_PATH))
        
        # Save feature engineering artifacts
        self.feature_engineer.save_artifacts(
            str(self.config.VECTORIZER_PATH),
            str(self.config.LABEL_ENCODER_PATH)
        )
        
        print("\nAll artifacts saved successfully!")
    
    def run(self):
        """Run the complete training pipeline"""
        print("\n" + "="*70)
        print(" "*20 + "SENTIMENT ANALYSIS TRAINING PIPELINE")
        print("="*70)
        
        # Run all steps
        self.load_data()
        self.preprocess_data()
        self.engineer_features()
        self.train_model()
        results = self.evaluate_model()
        self.save_artifacts()
        
        print("\n" + "="*70)
        print(" "*20 + "PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nFinal Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        return results


if __name__ == "__main__":
    # Run the training pipeline
    pipeline = TrainingPipeline()
    results = pipeline.run()
