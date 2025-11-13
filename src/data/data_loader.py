"""
Data loading module for sentiment analysis
Handles loading data from Hugging Face datasets
"""

import pandas as pd
from typing import Tuple
import os
from dotenv import load_dotenv
from huggingface_hub import login
import ssl
import certifi

class DataLoader:
    """Load and prepare datasets from Hugging Face"""
    
    def __init__(self, dataset_name: str = "Sp1786/multiclass-sentiment-analysis-dataset"):
        """
        Initialize DataLoader
        
        Args:
            dataset_name: Name of the Hugging Face dataset
        """
        self.dataset_name = dataset_name
        self.splits = {
            'train': 'train_df.csv',
            'validation': 'val_df.csv',
            'test': 'test_df.csv'
        }
        # Disable SSL verification for corporate networks
        self._disable_ssl_verification()
        self._login_to_hf()
    
    def _disable_ssl_verification(self):
        """Disable SSL verification to bypass certificate issues"""
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        print("SSL verification disabled for Hugging Face connection")
    
    def _login_to_hf(self):
        """Login to Hugging Face if token is available"""
        load_dotenv()
        hf_token = os.getenv('secret_token_hugface')
        if hf_token:
            login(hf_token)
            print("Successfully logged in to Hugging Face!")
        else:
            print("Warning: HF token not found. Public datasets only.")
    
    def load_train_data(self) -> pd.DataFrame:
        """Load training dataset"""
        print(f"Loading training data from {self.dataset_name}...")
        df = pd.read_csv(f"hf://datasets/{self.dataset_name}/{self.splits['train']}")
        print(f"Loaded {len(df)} training samples")
        return df
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset"""
        print(f"Loading test data from {self.dataset_name}...")
        df = pd.read_csv(f"hf://datasets/{self.dataset_name}/{self.splits['test']}")
        print(f"Loaded {len(df)} test samples")
        return df
    
    def load_validation_data(self) -> pd.DataFrame:
        """Load validation dataset"""
        print(f"Loading validation data from {self.dataset_name}...")
        df = pd.read_csv(f"hf://datasets/{self.dataset_name}/{self.splits['validation']}")
        print(f"Loaded {len(df)} validation samples")
        return df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = self.load_train_data()
        val_df = self.load_validation_data()
        test_df = self.load_test_data()
        return train_df, val_df, test_df
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning of dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Remove duplicates and null values
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Drop 'id' column if it exists
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        
        print(f"After cleaning: {len(df)} samples remaining")
        return df
