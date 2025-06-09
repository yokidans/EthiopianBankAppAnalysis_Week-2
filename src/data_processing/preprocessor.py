import pandas as pd
from datetime import datetime
import spacy
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_sm")

class ReviewPreprocessor:
    def __init__(self):
        self.text_columns = ['review']

    def preprocess_text(self, text):
        """Advanced text cleaning with spaCy"""
        doc = nlp(text)
        tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_stop 
            and not token.is_punct
            and len(token.text) > 2
        ]
        return " ".join(tokens)

    def preprocess(self, input_file=None):
        """Full preprocessing pipeline"""
        print("Starting preprocessing...")
        try:
            input_path = input_file or settings.RAW_DATA_DIR / 'bank_reviews.csv'
            df = pd.read_csv(input_path)
            
            logger.info(f"Initial data shape: {df.shape}")
            
            # Clean data
            df = df.drop_duplicates(subset=['review', 'bank'])
            df['review'] = df['review'].fillna('').str.strip()
            df['rating'] = df['rating'].fillna(0).astype(int)
            
            # Process dates
            try:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                df['year_month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
            except Exception as e:
                logger.warning(f"Date processing error: {e}")
                df['year_month'] = 'unknown'
            
            # Text preprocessing
            df['processed_text'] = df['review'].apply(self.preprocess_text)
            
            # Save processed data
            output_path = settings.PROCESSED_DATA_DIR / 'processed_reviews.parquet'
            df.to_parquet(output_path)
            logger.info(f"Processed data saved to {output_path}")
            print(f"Preprocessing complete. Output shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise