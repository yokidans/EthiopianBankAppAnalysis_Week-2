from transformers import pipeline
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise

    def analyze(self, text):
        """Hybrid sentiment analysis with thresholding"""
        try:
            result = self.model(text)[0]
            pos_score = next(i['score'] for i in result if i['label'] == 'POSITIVE')
            neg_score = next(i['score'] for i in result if i['label'] == 'NEGATIVE')
            
            if pos_score > settings.SENTIMENT_THRESHOLDS['positive']:
                return 'POSITIVE', pos_score
            elif neg_score > (1 - settings.SENTIMENT_THRESHOLDS['neutral_lower']):
                return 'NEGATIVE', neg_score
            else:
                return 'NEUTRAL', 0.5
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for text: {e}")
            return 'NEUTRAL', 0.5