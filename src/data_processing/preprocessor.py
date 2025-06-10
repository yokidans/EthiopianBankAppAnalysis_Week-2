# src/data_processing/preprocessor.py
import pandas as pd
from datetime import datetime
from pathlib import Path
import spacy
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_sm")

class ReviewPreprocessor:
    def __init__(self):
        self.text_columns = ['review']
        
    def _clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        # Remove special characters and extra whitespace
        text = ' '.join(text.split())
        return text.strip()

    def preprocess_text(self, text):
        """Advanced text cleaning with spaCy"""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        doc = nlp(text)
        tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_stop 
            and not token.is_punct
            and len(token.text) > 2
        ]
        return " ".join(tokens)

    def preprocess(self, input_data=None):
        """Full preprocessing pipeline
        Args:
            input_data: Either a DataFrame or a file path string
        """
        try:
            # Handle input
            if isinstance(input_data, pd.DataFrame):
                data = input_data.copy()
            elif isinstance(input_data, (str, Path)):
                data = pd.read_csv(input_data)
            elif input_data is None:
                input_path = settings.RAW_DATA_DIR / 'bank_reviews.csv'
                data = pd.read_csv(input_path)
            else:
                raise ValueError("Input must be either a DataFrame or file path")
            
            logger.info(f"Initial data shape: {data.shape}")
            
            if data.empty:
                logger.error("Empty DataFrame received")
                return pd.DataFrame()
            
            # Basic cleaning
            data = data.dropna(subset=['review'])
            data['review'] = data['review'].fillna('').apply(self._clean_text)
            data = data.drop_duplicates(subset=['review', 'bank'])
            
            # Handle numeric fields
            data['rating'] = data['rating'].fillna(0).astype(int)
            
            # Process dates
            try:
                data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
                data['year_month'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m')
            except Exception as e:
                logger.warning(f"Date processing error: {e}")
                data['year_month'] = 'unknown'
            
            # Advanced text preprocessing
            data['processed_text'] = data['review'].apply(self.preprocess_text)
            
            # Save processed data if input was a file path or default
            if not isinstance(input_data, pd.DataFrame):
                output_path = settings.PROCESSED_DATA_DIR / 'processed_reviews.parquet'
                data.to_parquet(output_path)
                logger.info(f"Processed data saved to {output_path}")
            
            logger.info(f"Preprocessing complete. Output shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise