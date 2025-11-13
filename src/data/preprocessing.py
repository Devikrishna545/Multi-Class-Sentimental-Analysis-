"""
Text preprocessing module
Handles text cleaning and transformation
"""

import string
import re
from typing import List, Union
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import nltk

class TextPreprocessor:
    """Preprocess text data for sentiment analysis"""
    
    def __init__(self, remove_stopwords: bool = True, lowercase: bool = True):
        """
        Initialize TextPreprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.tokenizer = WordPunctTokenizer()
        
        # Download NLTK data if needed
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str) or len(text) == 0:
            return ''
        
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Remove stopwords and punctuation
        if self.remove_stopwords:
            filtered_tokens = [
                word for word in tokens 
                if word.lower() not in self.stop_words and word not in string.punctuation
            ]
        else:
            filtered_tokens = tokens
        
        # Join tokens
        cleaned_text = ' '.join(filtered_tokens)
        
        # Remove any remaining non-alphanumeric characters (except whitespace)
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        
        # Convert to lowercase if specified
        if self.lowercase:
            cleaned_text = cleaned_text.lower()
        
        return cleaned_text
    
    def clean_texts(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """
        Clean multiple texts
        
        Args:
            texts: List or Series of texts
            
        Returns:
            List of cleaned texts
        """
        if isinstance(texts, pd.Series):
            return texts.apply(self.clean_text).tolist()
        return [self.clean_text(text) for text in texts]
    
    def count_symbols(self, text: str) -> int:
        """
        Count special symbols/punctuation in text
        
        Args:
            text: Input text
            
        Returns:
            Number of symbols
        """
        pattern = r'[^\w\s]'
        matches = re.findall(pattern, text)
        return len(matches)
    
    def add_text_features(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Add text-based features to dataframe
        
        Args:
            df: Input dataframe
            text_column: Name of the text column
            
        Returns:
            Dataframe with added features
        """
        df = df.copy()
        
        # Text length (before cleaning)
        df['text_length'] = df[text_column].apply(len)
        
        # Word count (before cleaning)
        df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
        
        # Symbol count
        df['symbol_count'] = df[text_column].apply(self.count_symbols)
        
        print(f"Added features: text_length, word_count, symbol_count")
        return df
