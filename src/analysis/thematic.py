import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import spacy
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_sm")

class ThematicAnalyzer:
    def __init__(self):
        self.theme_keywords = {
            'Account Access': ['login', 'password', 'account', 'access', 'verify'],
            'Transactions': ['transfer', 'payment', 'transaction', 'send', 'receive'],
            'User Experience': ['app', 'interface', 'design', 'experience', 'navigation'],
            'Customer Support': ['support', 'service', 'help', 'response', 'agent'],
            'Fees & Charges': ['fee', 'charge', 'cost', 'price', 'commission']
        }

    def analyze(self, df):
        """Perform thematic analysis on processed reviews"""
        logger.info("Starting thematic analysis...")
        
        bank_themes = {}
        
        for bank in settings.BANKS:
            logger.info(f"Analyzing themes for {bank}...")
            bank_reviews = df[df['bank'] == bank]
            
            if len(bank_reviews) == 0:
                logger.warning(f"No reviews found for {bank}")
                continue
                
            # Extract keywords using TF-IDF
            tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=50)
            tfidf_matrix = tfidf.fit_transform(bank_reviews['processed_text'])
            keywords = tfidf.get_feature_names_out()
            
            # Cluster keywords into themes
            themes = self._cluster_keywords(keywords)
            bank_themes[bank] = themes
            
            logger.info(f"Identified {len(themes)} themes for {bank}")
        
        return bank_themes

    def _cluster_keywords(self, keywords):
        """Group keywords into predefined themes"""
        themes = defaultdict(list)
        
        for keyword in keywords:
            keyword_doc = nlp(keyword)
            
            # Find best matching theme
            best_theme = None
            highest_sim = 0
            
            for theme, theme_words in self.theme_keywords.items():
                similarity = max(
                    keyword_doc.similarity(nlp(word))
                    for word in theme_words
                )
                
                if similarity > highest_sim and similarity > 0.6:
                    highest_sim = similarity
                    best_theme = theme
            
            if best_theme:
                themes[best_theme].append(keyword)
        
        return dict(themes)