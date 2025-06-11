import pandas as pd
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from wordcloud import WordCloud
from typing import Tuple, Dict, Any, Optional, List
from src.data_processing.scraper import BankReviewScraper
from src.data_processing.preprocessor import ReviewPreprocessor
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.thematic import ThematicAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style='whitegrid')

class AnalysisPipeline:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.scraper = BankReviewScraper()
        self.preprocessor = ReviewPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.thematic_analyzer = ThematicAnalyzer()
        self._setup_fallback_analyzers()

    def _setup_fallback_analyzers(self):
        """Initialize fallback sentiment analyzers with lazy loading"""
        self._textblob_available = False
        self._vader_available = False
        
        try:
            from textblob import TextBlob
            self._textblob_available = True
            logger.info("TextBlob fallback analyzer available")
        except ImportError:
            logger.warning("TextBlob not available for fallback analysis")
        
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            self._vader_available = True
            logger.info("VADER fallback analyzer available")
        except ImportError:
            logger.warning("VADER not available for fallback analysis")

    def _validate_sentiment_results(self, scores: List[float], labels: Optional[List[str]], expected_length: int) -> bool:
        """Validate sentiment analysis results with detailed error messages"""
        if not isinstance(scores, (list, np.ndarray)):
            logger.error(f"Invalid scores type: {type(scores)}. Expected list or numpy array.")
            return False
            
        if len(scores) != expected_length:
            logger.error(f"Scores length mismatch: {len(scores)} != {expected_length}")
            return False
            
        if labels is not None:
            if not isinstance(labels, (list, np.ndarray)):
                logger.error(f"Invalid labels type: {type(labels)}. Expected list or numpy array.")
                return False
            if len(labels) != expected_length:
                logger.error(f"Labels length mismatch: {len(labels)} != {expected_length}")
                return False
                
        try:
            [float(x) for x in scores]  # Verify all scores are numeric
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid score values: {e}")
            return False

    def _analyze_with_fallbacks(self, texts: List[str]) -> Optional[Tuple[List[float], List[str]]]:
        """Perform sentiment analysis with multiple fallback mechanisms and detailed logging"""
        methods = [
            ('Primary analyzer', self._analyze_with_primary),
            ('TextBlob fallback', self._analyze_with_textblob),
            ('VADER fallback', self._analyze_with_vader),
            ('Keyword fallback', self._analyze_with_keywords)
        ]
        
        last_exception = None
        
        for name, method in methods:
            try:
                logger.info(f"Attempting sentiment analysis with {name}")
                result = method(texts)
                if result is not None:
                    scores, labels = result
                    if self._validate_sentiment_results(scores, labels, len(texts)):
                        logger.info(f"Successfully used {name} for sentiment analysis")
                        return result
                    else:
                        logger.warning(f"{name} returned invalid results")
                else:
                    logger.warning(f"{name} returned None")
            except Exception as e:
                last_exception = e
                logger.error(f"Sentiment method {name} failed: {str(e)}", exc_info=True)
        
        if last_exception:
            logger.error(f"All sentiment methods failed. Last error: {str(last_exception)}")
        else:
            logger.error("All sentiment methods returned invalid results")
        
        return None

    def _analyze_with_primary(self, texts: List[str]) -> Optional[Tuple[List[float], List[str]]]:
        """Primary sentiment analysis using configured analyzer with enhanced error handling"""
        if not hasattr(self.sentiment_analyzer, 'analyze'):
            logger.error("Primary sentiment analyzer missing 'analyze' method")
            return None
            
        try:
            result = self.sentiment_analyzer.analyze(texts)
            
            # Handle malformed string output
            if isinstance(result, str):
                logger.warning(f"Unexpected string output from analyzer: {result[:100]}...")
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    if any(label in result for label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']):
                        logger.warning("Creating dummy scores from label string")
                        n = len(texts)
                        if "POSITIVE" in result:
                            return ([1.0] * n, ['POSITIVE'] * n)
                        elif "NEGATIVE" in result:
                            return ([-1.0] * n, ['NEGATIVE'] * n)
                        else:
                            return ([0.0] * n, ['NEUTRAL'] * n)
                    return None
            
            # Handle different return formats
            if isinstance(result, pd.DataFrame):
                if {'sentiment_score', 'sentiment_label'}.issubset(result.columns):
                    return (result['sentiment_score'].tolist(), 
                           result['sentiment_label'].tolist())
                else:
                    logger.error("DataFrame missing required sentiment columns")
            elif isinstance(result, tuple) and len(result) == 2:
                return result
            elif isinstance(result, (list, np.ndarray)):
                return (result, None)
            
            logger.error(f"Unexpected result type from primary analyzer: {type(result)}")
            return None
            
        except Exception as e:
            logger.error(f"Primary analyzer failed: {str(e)}", exc_info=True)
            return None

    def _analyze_with_textblob(self, texts: List[str]) -> Optional[Tuple[List[float], List[str]]]:
        """Fallback analysis using TextBlob with enhanced validation"""
        if not self._textblob_available:
            return None
            
        try:
            from textblob import TextBlob
            
            scores = []
            labels = []
            for text in texts:
                try:
                    analysis = TextBlob(str(text))
                    score = analysis.sentiment.polarity
                    scores.append(score)
                    labels.append(
                        'POSITIVE' if score > 0.1 else 
                        'NEGATIVE' if score < -0.1 else 
                        'NEUTRAL'
                    )
                except Exception as e:
                    logger.warning(f"TextBlob failed on text: {str(text)[:100]}... Error: {str(e)}")
                    scores.append(0.0)
                    labels.append('NEUTRAL')
            
            return (scores, labels)
            
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {str(e)}", exc_info=True)
            return None

    def _analyze_with_vader(self, texts: List[str]) -> Optional[Tuple[List[float], List[str]]]:
        """Fallback analysis using VADER with enhanced validation"""
        if not self._vader_available:
            return None
            
        try:
            scores = []
            labels = []
            for text in texts:
                try:
                    vs = self._vader.polarity_scores(str(text))
                    scores.append(vs['compound'])
                    labels.append(
                        'POSITIVE' if vs['compound'] >= 0.05 else 
                        'NEGATIVE' if vs['compound'] <= -0.05 else 
                        'NEUTRAL'
                    )
                except Exception as e:
                    logger.warning(f"VADER failed on text: {str(text)[:100]}... Error: {str(e)}")
                    scores.append(0.0)
                    labels.append('NEUTRAL')
            
            return (scores, labels)
            
        except Exception as e:
            logger.error(f"VADER analysis failed: {str(e)}", exc_info=True)
            return None

    def _analyze_with_keywords(self, texts: List[str]) -> Optional[Tuple[List[float], List[str]]]:
        """Final fallback using keyword matching with enhanced validation"""
        positive_words = {
            'good', 'great', 'excellent', 'happy', 'love', 'awesome', 'fantastic',
            'wonderful', 'amazing', 'superb', 'perfect', 'nice', 'best'
        }
        negative_words = {
            'bad', 'poor', 'terrible', 'hate', 'awful', 'horrible', 'worst',
            'disappointing', 'unhappy', 'negative', 'angry', 'suck'
        }
        
        try:
            scores = []
            labels = []
            for text in texts:
                try:
                    text_lower = str(text).lower()
                    pos_count = sum(word in text_lower for word in positive_words)
                    neg_count = sum(word in text_lower for word in negative_words)
                    
                    score = (pos_count - neg_count) / max(1, (pos_count + neg_count))
                    scores.append(score)
                    labels.append(
                        'POSITIVE' if score > 0.3 else 
                        'NEGATIVE' if score < -0.3 else 
                        'NEUTRAL'
                    )
                except Exception as e:
                    logger.warning(f"Keyword analysis failed on text: {str(text)[:100]}... Error: {str(e)}")
                    scores.append(0.0)
                    labels.append('NEUTRAL')
            
            return (scores, labels)
            
        except Exception as e:
            logger.error(f"Keyword analysis failed: {str(e)}", exc_info=True)
            return None

    def _save_results(self, data: pd.DataFrame, filename: str) -> Path:
        """Helper method to save intermediate results with validation"""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
                
            output_path = self.data_dir / "processed" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not data.empty:
                data.to_csv(output_path, index=False)
                logger.info(f"Saved results to {output_path}")
            else:
                logger.warning("Attempted to save empty DataFrame")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}", exc_info=True)
            raise

    def scrape_or_load_data(self, scrape_new: bool = False) -> pd.DataFrame:
        """Either scrape new data or load existing dataset with enhanced validation"""
        try:
            raw_path = self.data_dir / "raw" / "bank_reviews.csv"
            
            if scrape_new or not raw_path.exists():
                logger.info("Scraping new data...")
                raw_data = self.scraper.scrape()
                
                if not isinstance(raw_data, pd.DataFrame):
                    raise ValueError("Scraper returned non-DataFrame result")
                if raw_data.empty:
                    raise ValueError("Scraper returned empty DataFrame")
                
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_data.to_csv(raw_path, index=False)
                logger.info(f"Saved new data to {raw_path}")
            else:
                logger.info(f"Loading existing data from {raw_path}")
                raw_data = pd.read_csv(raw_path)
                
                if raw_data.empty:
                    raise ValueError("Loaded empty DataFrame from file")
            
            logger.info(f"Loaded data with {len(raw_data)} records and columns: {list(raw_data.columns)}")
            return raw_data
            
        except Exception as e:
            logger.error(f"Failed to load/scrape data: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _preprocess_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure text data is properly formatted with enhanced validation"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            if 'review_text' not in df.columns:
                if 'review' in df.columns:
                    df = df.rename(columns={'review': 'review_text'})
                    logger.info("Renamed column 'review' to 'review_text'")
                else:
                    raise ValueError("No review text column found in DataFrame")
            
            # Clean text data
            df['review_text'] = (
                df['review_text']
                .astype(str)
                .str.strip()
                .replace(r'^\s*$', np.nan, regex=True)
            )
            
            # Validate after cleaning
            empty_count = df['review_text'].isna().sum()
            if empty_count > 0:
                logger.warning(f"Found {empty_count} empty/invalid text entries after cleaning")
            
            result = df.dropna(subset=['review_text'])
            
            if result.empty:
                raise ValueError("No valid text data remaining after cleaning")
                
            return result
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {str(e)}", exc_info=True)
            raise

    def _generate_sentiment_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive sentiment statistics with validation"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            required_columns = {'sentiment_label', 'sentiment_score'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"Missing required columns: {missing}")
            
            stats = {
                'counts': {
                    'positive': int((df['sentiment_label'] == 'POSITIVE').sum()),
                    'negative': int((df['sentiment_label'] == 'NEGATIVE').sum()),
                    'neutral': int((df['sentiment_label'] == 'NEUTRAL').sum()),
                    'total': len(df)
                },
                'scores': {
                    'mean': float(df['sentiment_score'].mean()),
                    'median': float(df['sentiment_score'].median()),
                    'std': float(df['sentiment_score'].std()),
                    'min': float(df['sentiment_score'].min()),
                    'max': float(df['sentiment_score'].max())
                },
                'distribution': {
                    'positive_pct': float((df['sentiment_label'] == 'POSITIVE').mean()),
                    'negative_pct': float((df['sentiment_label'] == 'NEGATIVE').mean()),
                    'neutral_pct': float((df['sentiment_label'] == 'NEUTRAL').mean())
                }
            }
            
            # Add score distribution bins
            bins = [-1, -0.5, -0.1, 0.1, 0.5, 1]
            score_dist = pd.cut(
                df['sentiment_score'],
                bins=bins,
                labels=['Strong Negative', 'Weak Negative', 'Neutral', 
                       'Weak Positive', 'Strong Positive']
            ).value_counts(normalize=True)
            stats['score_distribution'] = score_dist.to_dict()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to generate sentiment stats: {str(e)}", exc_info=True)
            return {}

    def synthesize_insights(self, df: pd.DataFrame, themes: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive insights from analysis results with robust error handling"""
        insights = {
            'summary_metrics': {},
            'sentiment_trends': {},
            'theme_distributions': {},
            'key_findings': []
        }
        
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise ValueError("Empty or invalid DataFrame provided")
                
            if not isinstance(themes, dict):
                raise ValueError("Themes must be a dictionary")
            
            # 1. Calculate summary metrics
            insights['summary_metrics'] = {
                'total_reviews': len(df),
                'average_rating': float(df['rating'].mean()) if 'rating' in df.columns else 0.0,
                'positive_sentiment': len(df[df['sentiment_label'] == 'POSITIVE']) / len(df),
                'negative_sentiment': len(df[df['sentiment_label'] == 'NEGATIVE']) / len(df),
                'banks_analyzed': df['bank'].nunique() if 'bank' in df.columns else 0
            }
            
            # 2. Analyze sentiment trends by bank
            if 'bank' in df.columns:
                for bank in df['bank'].unique():
                    try:
                        bank_df = df[df['bank'] == bank]
                        if not bank_df.empty:
                            insights['sentiment_trends'][bank] = {
                                'avg_sentiment': float(bank_df['sentiment_score'].mean()),
                                'positive_pct': len(bank_df[bank_df['sentiment_label'] == 'POSITIVE']) / len(bank_df),
                                'negative_pct': len(bank_df[bank_df['sentiment_label'] == 'NEGATIVE']) / len(bank_df)
                            }
                    except Exception as e:
                        logger.error(f"Failed to process bank {bank}: {str(e)}")
                        continue
            
            # 3. Analyze theme distributions
            for bank, bank_themes in themes.items():
                try:
                    if bank_themes and isinstance(bank_themes, dict):
                        if bank_themes:  # Check if themes exist for this bank
                            theme_items = list(bank_themes.items())
                            if theme_items:
                                most_common_theme = max(theme_items, key=lambda x: len(x[1]))[0]
                                insights['theme_distributions'][bank] = {
                                    'most_common_theme': most_common_theme,
                                    'total_themes': len(bank_themes),
                                    'theme_frequencies': {theme: len(keywords) for theme, keywords in bank_themes.items()}
                                }
                except Exception as e:
                    logger.error(f"Failed to process themes for bank {bank}: {str(e)}")
                    continue
            
            # 4. Generate key findings
            if insights['sentiment_trends']:
                try:
                    best_bank = max(insights['sentiment_trends'].items(), 
                                  key=lambda x: x[1]['avg_sentiment'])[0]
                    worst_bank = min(insights['sentiment_trends'].items(), 
                                    key=lambda x: x[1]['avg_sentiment'])[0]
                    
                    insights['key_findings'] = [
                        f"{best_bank} had the highest average sentiment score",
                        f"{worst_bank} had the lowest average sentiment score",
                        f"{len(df[df['sentiment_label'] == 'NEGATIVE'])} reviews contained negative sentiment"
                    ]
                    
                    # Add theme finding if available
                    if best_bank in insights['theme_distributions']:
                        insights['key_findings'].append(
                            f"The most common theme for {best_bank} was " +
                            f"'{insights['theme_distributions'][best_bank]['most_common_theme']}'"
                        )
                except Exception as e:
                    logger.error(f"Failed to generate key findings: {str(e)}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Insights generation failed: {str(e)}", exc_info=True)
            return insights

    def visualize_results(self, df: pd.DataFrame, themes: Dict[str, Any], insights: Dict[str, Any]) -> Optional[Path]:
        """Generate visualizations from analysis results with robust error handling"""
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning("No valid data available for visualization")
                return None
                
            viz_dir = self.data_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Sentiment Distribution Pie Chart
            try:
                plt.figure(figsize=(8, 6))
                df['sentiment_label'].value_counts().plot.pie(
                    autopct='%1.1f%%',
                    colors=['#4CAF50', '#F44336', '#FFC107'],
                    title='Overall Sentiment Distribution'
                )
                plt.savefig(viz_dir / f"sentiment_distribution_{timestamp}.png")
                plt.close()
            except Exception as e:
                logger.error(f"Failed to create sentiment pie chart: {str(e)}")
            
            # 2. Bank Comparison Bar Chart
            try:
                if 'bank' in df.columns and 'sentiment_score' in df.columns:
                    plt.figure(figsize=(10, 6))
                    sentiment_by_bank = df.groupby('bank')['sentiment_score'].mean().sort_values()
                    sentiment_by_bank.plot.barh(color='#2196F3')
                    plt.title('Average Sentiment Score by Bank')
                    plt.xlabel('Sentiment Score (Higher is Better)')
                    plt.tight_layout()
                    plt.savefig(viz_dir / f"bank_comparison_{timestamp}.png")
                    plt.close()
            except Exception as e:
                logger.error(f"Failed to create bank comparison chart: {str(e)}")
            
            # 3. Theme Word Clouds
            for bank, bank_themes in themes.items():
                try:
                    if bank_themes and isinstance(bank_themes, dict):
                        plt.figure(figsize=(10, 6))
                        theme_text = ' '.join([' '.join(keywords) for keywords in bank_themes.values()])
                        if theme_text.strip():
                            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(theme_text)
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.title(f'Key Themes for {bank}')
                            plt.axis('off')
                            plt.savefig(viz_dir / f"wordcloud_{bank.lower().replace(' ', '_')}_{timestamp}.png")
                            plt.close()
                except Exception as e:
                    logger.error(f"Failed to create word cloud for {bank}: {str(e)}")
            
            # 4. Rating Distribution
            try:
                if 'rating' in df.columns:
                    plt.figure(figsize=(8, 6))
                    sns.countplot(data=df, x='rating', palette='Blues_r', hue='rating', legend=False)
                    plt.title('Distribution of Star Ratings')
                    plt.xlabel('Star Rating')
                    plt.ylabel('Number of Reviews')
                    plt.tight_layout()
                    plt.savefig(viz_dir / f"rating_distribution_{timestamp}.png")
                    plt.close()
            except Exception as e:
                logger.error(f"Failed to create rating distribution: {str(e)}")
            
            logger.info(f"Visualizations saved to {viz_dir}")
            return viz_dir
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}", exc_info=True)
            return None

    def generate_report(self, df: pd.DataFrame, themes: Dict[str, Any], insights: Dict[str, Any], viz_dir: Optional[Path]) -> Optional[Path]:
        """Generate a comprehensive HTML report with robust error handling"""
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning("No valid data available for report generation")
                return None
                
            report_dir = Path(__file__).parent.parent / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"analysis_report_{timestamp}.html"
            
            # Get visualization paths
            viz_files = []
            if viz_dir and viz_dir.exists():
                viz_files = sorted(list(viz_dir.glob('*.png')))
            
            # Prepare theme content
            theme_content = []
            for bank, bank_themes in themes.items():
                if bank_themes and isinstance(bank_themes, dict):
                    theme_items = []
                    for theme, keywords in bank_themes.items():
                        if isinstance(keywords, (list, tuple)):
                            theme_items.append(f"""
                                <li><strong>{theme}:</strong> {', '.join(str(k) for k in keywords[:5])}{'...' if len(keywords) > 5 else ''}</li>
                            """)
                    if theme_items:
                        theme_content.append(f"""
                            <div>
                                <h3>{bank}</h3>
                                <ul>{"".join(theme_items)}</ul>
                            </div>
                        """)
            
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>Bank Review Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .container {{ width: 90%; margin: auto; }}
                    .section {{ margin-bottom: 30px; }}
                    .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                    img {{ max-width: 100%; border: 1px solid #ddd; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Bank Review Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}</p>
                    
                    <div class="section">
                        <h2>Summary Metrics</h2>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                            {"".join(f"<tr><td>{k}</td><td>{'{:.2f}'.format(v) if isinstance(v, float) else v}</td></tr>" 
                              for k, v in insights.get('summary_metrics', {}).items())}
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Key Findings</h2>
                        <ul>
                            {"".join(f"<li>{finding}</li>" for finding in insights.get('key_findings', []))}
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h2>Visualizations</h2>
                        <div class="grid">
                            {"".join(f'<div><h3>{f.stem.replace("_", " ").title()}</h3><img src="{f}" /></div>' 
                              for f in viz_files if f.exists())}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Detailed Theme Analysis</h2>
                        {"".join(theme_content)}
                    </div>
                </div>
            </body>
            </html>
            """
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Report generated at {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            return None

    def run_full_pipeline(self, scrape_new: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Optional[Path], Optional[Path]]:
        """Execute the complete analysis pipeline with robust error handling"""
        try:
            logger.info("=== Starting Pipeline Execution ===")
            
            # 1. Data acquisition
            raw_data = self.scrape_or_load_data(scrape_new)
            if raw_data.empty:
                logger.error("No data available for analysis")
                return pd.DataFrame(), {}, {}, None, None
            
            # 2. Data preprocessing
            logger.info("Starting preprocessing...")
            try:
                processed_data = self.preprocessor.preprocess(raw_data.copy())
                processed_data = self._preprocess_text_data(processed_data)
                
                # Validate required columns
                required_columns = {'review_text', 'rating', 'bank'}
                if not required_columns.issubset(processed_data.columns):
                    missing_cols = required_columns - set(processed_data.columns)
                    logger.error(f"Missing required columns: {missing_cols}")
                    return pd.DataFrame(), {}, {}, None, None
                    
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
                return pd.DataFrame(), {}, {}, None, None
            
            # 3. Sentiment analysis
            logger.info("Starting sentiment analysis...")
            try:
                valid_reviews = processed_data[processed_data['review_text'].str.strip().ne("")]
                if len(valid_reviews) == 0:
                    logger.error("No valid text data available for sentiment analysis")
                    return pd.DataFrame(), {}, {}, None, None
                
                review_texts = valid_reviews['review_text'].tolist()
                sentiment_results = self._analyze_with_fallbacks(review_texts)
                
                if sentiment_results is None:
                    logger.error("All sentiment analysis methods failed")
                    return pd.DataFrame(), {}, {}, None, None
                
                scores, labels = sentiment_results
                
                processed_data = valid_reviews.copy()
                processed_data['sentiment_score'] = scores
                processed_data['sentiment_label'] = labels if labels else [
                    'POSITIVE' if score > 0.1 else 
                    'NEGATIVE' if score < -0.1 else 
                    'NEUTRAL'
                    for score in scores
                ]
                
                sentiment_stats = self._generate_sentiment_stats(processed_data)
                logger.info(f"Sentiment analysis completed. Stats: {sentiment_stats}")
                
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
                return pd.DataFrame(), {}, {}, None, None
            
            # 4. Thematic analysis
            logger.info("Starting thematic analysis...")
            themes = {}
            if 'bank' in processed_data.columns:
                for bank in processed_data['bank'].unique():
                    try:
                        bank_reviews = processed_data[processed_data['bank'] == bank]['review_text'].tolist()
                        if bank_reviews:
                            bank_result = self.thematic_analyzer.analyze(bank_reviews)
                            if bank_result and isinstance(bank_result, dict):
                                themes[bank] = bank_result
                            else:
                                logger.warning(f"No valid themes returned for bank {bank}")
                        else:
                            logger.warning(f"No reviews available for bank {bank}")
                    except Exception as e:
                        logger.error(f"Thematic analysis failed for bank {bank}: {str(e)}")
                        continue
            
            logger.info(f"Thematic analysis completed for {len(themes)} banks")
            
            # 5. Generate insights
            logger.info("Generating insights...")
            insights = self.synthesize_insights(processed_data, themes)
            
            # 6. Create visualizations
            logger.info("Creating visualizations...")
            viz_dir = self.visualize_results(processed_data, themes, insights)
            
            # 7. Generate report
            logger.info("Generating report...")
            report_file = self.generate_report(processed_data, themes, insights, viz_dir)
            
            logger.info("=== Pipeline Completed Successfully ===")
            return processed_data, themes, insights, viz_dir, report_file
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return pd.DataFrame(), {}, {}, None, None

def main():
    """Main entry point for the pipeline with enhanced error handling"""
    print("=== Ethiopian Bank Review Analysis Pipeline ===")
    
    try:
        pipeline = AnalysisPipeline()
        df, themes, insights, viz_dir, report_file = pipeline.run_full_pipeline()
        
        if df.empty:
            print("\nERROR: No data available for analysis")
            return
        
        print("\n=== Key Insights ===")
        for finding in insights.get('key_findings', ["No insights generated"]):
            print(f"- {finding}")
        
        if viz_dir:
            print(f"\n=== Visualizations saved to: {viz_dir} ===")
        if report_file:
            print(f"\n=== Full report generated: {report_file} ===")
            
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # Install additional visualization dependencies if needed
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("Installing visualization dependencies...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "wordcloud", "matplotlib", "seaborn"])
    
    main()