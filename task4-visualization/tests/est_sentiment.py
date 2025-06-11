import unittest
from src.analysis.sentiment import SentimentAnalyzer

class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        label, score = self.analyzer.analyze("I love this bank! Great service!")
        self.assertEqual(label, "POSITIVE")
    
    def test_negative_sentiment(self):
        label, score = self.analyzer.analyze("Terrible experience with this bank")
        self.assertEqual(label, "NEGATIVE")