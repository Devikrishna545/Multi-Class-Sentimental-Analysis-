"""
Evaluation pipeline
Comprehensive model evaluation and analysis
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.config import Config
from src.data.data_loader import DataLoader
from src.data.preprocessing import TextPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train import SentimentModel


class EvaluationPipeline:
    """Comprehensive evaluation pipeline"""
    
    def __init__(self, config: Config = None):
        """
        Initialize evaluation pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize components
        self.data_loader = DataLoader(self.config.HF_DATASET)
        self.preprocessor = TextPreprocessor(
            remove_stopwords=self.config.REMOVE_STOPWORDS,
            lowercase=self.config.LOWERCASE
        )
        self.feature_engineer = FeatureEngineer()
        self.model = SentimentModel({})
        
        # Load artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained model and artifacts"""
        print("Loading trained artifacts...")
        
        self.feature_engineer.load_artifacts(
            str(self.config.VECTORIZER_PATH),
            str(self.config.LABEL_ENCODER_PATH)
        )
        
        self.model.load_model(str(self.config.MODEL_PATH))
        
        print("Artifacts loaded successfully!")
    
    def evaluate_on_test_set(self) -> Dict[str, Any]:
        """Evaluate model on test dataset"""
        print("\n" + "="*50)
        print("EVALUATING ON TEST SET")
        print("="*50)
        
        # Load test data
        test_df = self.data_loader.load_test_data()
        test_df = self.data_loader.clean_dataframe(test_df)
        
        # Preprocess
        test_df = self.preprocessor.add_text_features(test_df)
        test_df['text'] = self.preprocessor.clean_texts(test_df['text'])
        
        # Create features
        X_test = self.feature_engineer.create_features(
            test_df,
            text_column='text',
            fit=False
        )
        
        # Encode labels
        y_test = self.feature_engineer.transform_labels(test_df['sentiment'])
        
        # Evaluate
        results = self.model.evaluate(
            X_test,
            y_test,
            label_names=list(self.config.SENTIMENT_LABELS.values())
        )
        
        return results
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            save_path: Optional path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.config.SENTIMENT_LABELS.values()),
            yticklabels=list(self.config.SENTIMENT_LABELS.values()),
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def analyze_errors(self, test_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """
        Analyze misclassified examples
        
        Args:
            test_df: Test dataframe
            predictions: Model predictions
            
        Returns:
            DataFrame of misclassified examples
        """
        # Get true labels
        true_labels = self.feature_engineer.transform_labels(test_df['sentiment'])
        
        # Find misclassified
        misclassified_mask = true_labels != predictions
        misclassified_df = test_df[misclassified_mask].copy()
        misclassified_df['predicted'] = self.feature_engineer.inverse_transform_labels(
            predictions[misclassified_mask]
        )
        
        print(f"\nMisclassified examples: {len(misclassified_df)}/{len(test_df)}")
        print(f"Error rate: {len(misclassified_df)/len(test_df)*100:.2f}%")
        
        return misclassified_df
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with visualizations"""
        print("\n" + "="*70)
        print(" "*15 + "COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Evaluate on test set
        results = self.evaluate_on_test_set()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(results['confusion_matrix'])
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED!")
        print("="*70)
        
        return results


if __name__ == "__main__":
    # Run comprehensive evaluation
    pipeline = EvaluationPipeline()
    results = pipeline.run_comprehensive_evaluation()
