import pandas as pd
import sys
import numpy as np  # Only if you actually need numpy elsewhere
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from wordcloud import WordCloud
from src.data_processing.scraper import BankReviewScraper
from src.data_processing.preprocessor import ReviewPreprocessor
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.thematic import ThematicAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style='whitegrid')  # Updated seaborn style initialization

class AnalysisPipeline:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.scraper = BankReviewScraper()
        self.preprocessor = ReviewPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.thematic_analyzer = ThematicAnalyzer()

    def _save_results(self, data, filename):
        """Helper method to save intermediate results"""
        output_path = self.data_dir / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
        return output_path

    def scrape_or_load_data(self, scrape_new=False):
        """Either scrape new data or load existing dataset"""
        try:
            raw_path = self.data_dir / "raw" / "bank_reviews.csv"
            
            if scrape_new or not raw_path.exists():
                logger.info("Scraping new data...")
                raw_data = self.scraper.scrape()
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_data.to_csv(raw_path, index=False)
            else:
                logger.info("Loading existing data...")
                raw_data = pd.read_csv(raw_path)
            
            logger.info(f"Loaded data with {len(raw_data)} records")
            return raw_data
            
        except Exception as e:
            logger.error(f"Failed to load/scrape data: {str(e)}")
            return pd.DataFrame()

    def synthesize_insights(self, df, themes):
        """Generate comprehensive insights from analysis results"""
        insights = {
            'summary_metrics': {},
            'sentiment_trends': {},
            'theme_distributions': {},
            'key_findings': []
        }
        
        if df.empty:
            logger.warning("Empty dataframe provided for insights")
            return insights
            
        try:
            # 1. Calculate summary metrics
            insights['summary_metrics'] = {
                'total_reviews': len(df),
                'average_rating': df['rating'].mean(),
                'positive_sentiment': len(df[df['sentiment_label'] == 'POSITIVE']) / len(df),
                'negative_sentiment': len(df[df['sentiment_label'] == 'NEGATIVE']) / len(df),
                'banks_analyzed': df['bank'].nunique()
            }
            
            # 2. Analyze sentiment trends by bank
            for bank in df['bank'].unique():
                bank_df = df[df['bank'] == bank]
                insights['sentiment_trends'][bank] = {
                    'avg_sentiment': bank_df['sentiment_score'].mean(),
                    'positive_pct': len(bank_df[bank_df['sentiment_label'] == 'POSITIVE']) / len(bank_df),
                    'negative_pct': len(bank_df[bank_df['sentiment_label'] == 'NEGATIVE']) / len(bank_df)
                }
            
            # 3. Analyze theme distributions
            for bank, bank_themes in themes.items():
                if bank_themes:  # Check if themes exist for this bank
                    insights['theme_distributions'][bank] = {
                        'most_common_theme': max(bank_themes.items(), key=lambda x: len(x[1]))[0],
                        'total_themes': len(bank_themes),
                        'theme_frequencies': {theme: len(keywords) for theme, keywords in bank_themes.items()}
                    }
            
            # 4. Generate key findings
            if insights['sentiment_trends']:
                best_bank = max(insights['sentiment_trends'].items(), 
                              key=lambda x: x[1]['avg_sentiment'])[0]
                worst_bank = min(insights['sentiment_trends'].items(), 
                                key=lambda x: x[1]['avg_sentiment'])[0]
                
                insights['key_findings'] = [
                    f"{best_bank} had the highest average sentiment score",
                    f"{worst_bank} had the lowest average sentiment score",
                    f"The most common theme across banks was '{insights['theme_distributions'][best_bank]['most_common_theme']}'",
                    f"{len(df[df['sentiment_label'] == 'NEGATIVE'])} reviews contained negative sentiment"
                ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Insights generation failed: {str(e)}")
            return insights

    def visualize_results(self, df, themes, insights):
        """Generate visualizations from analysis results"""
        try:
            if df.empty:
                logger.warning("No data available for visualization")
                return None
                
            viz_dir = Path(__file__).parent.parent / "data" / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Sentiment Distribution Pie Chart
            plt.figure(figsize=(8, 6))
            sentiment_counts = df['sentiment_label'].value_counts()
            sentiment_counts.plot.pie(
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336', '#FFC107'],
                title='Overall Sentiment Distribution',
                labels=['Positive', 'Negative', 'Neutral']
            )
            plt.savefig(viz_dir / f"sentiment_distribution_{timestamp}.png")
            plt.close()
            
            # 2. Bank Comparison Bar Chart
            plt.figure(figsize=(10, 6))
            sentiment_by_bank = df.groupby('bank')['sentiment_score'].mean().sort_values()
            sentiment_by_bank.plot.barh(color='#2196F3')
            plt.title('Average Sentiment Score by Bank')
            plt.xlabel('Sentiment Score (Higher is Better)')
            plt.tight_layout()
            plt.savefig(viz_dir / f"bank_comparison_{timestamp}.png")
            plt.close()
            
            # 3. Theme Word Clouds
            for bank, bank_themes in themes.items():
                if bank_themes:
                    plt.figure(figsize=(10, 6))
                    theme_text = ' '.join([' '.join(keywords) for keywords in bank_themes.values()])
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(theme_text)
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.title(f'Key Themes for {bank}')
                    plt.axis('off')
                    plt.savefig(viz_dir / f"wordcloud_{bank.lower().replace(' ', '_')}_{timestamp}.png")
                    plt.close()
            
            # 4. Rating Distribution - Fixed with proper hue assignment
            plt.figure(figsize=(8, 6))
            rating_labels = {
                1: "1 Star - Very Poor",
                2: "2 Stars - Poor",
                3: "3 Stars - Average",
                4: "4 Stars - Good",
                5: "5 Stars - Excellent"
            }
            
            # Create a new column with descriptive labels
            df['rating_label'] = df['rating'].map(rating_labels)
            
            sns.countplot(
                data=df,
                x='rating',
                hue='rating',  # Assign hue to fix the warning
                palette='Blues_r',
                legend=False
            )
            plt.title('Distribution of Star Ratings')
            plt.xlabel('Star Rating')
            plt.ylabel('Number of Reviews')
            
            # Add custom x-tick labels
            plt.xticks(ticks=range(5), labels=[rating_labels[i+1] for i in range(5)], rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"rating_distribution_{timestamp}.png")
            plt.close()
            
            logger.info(f"Visualizations saved to {viz_dir}")
            return viz_dir
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            return None

    def generate_report(self, df, themes, insights, viz_dir):
        """Generate a comprehensive HTML report"""
        try:
            if df.empty:
                logger.warning("No data available for report generation")
                return None
                
            report_dir = Path(__file__).parent.parent / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"analysis_report_{timestamp}.html"
            
            # Get visualization paths
            viz_files = list(viz_dir.glob('*.png')) if viz_dir else []
            
            # Create HTML report - Fixed string formatting
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
                            {"".join(
                                f"<tr><td>{k}</td><td>{v:.2f if isinstance(v, float) else v}</td></tr>"
                                for k, v in insights['summary_metrics'].items()
                            )}
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Key Findings</h2>
                        <ul>
                            {"".join(f"<li>{finding}</li>" for finding in insights['key_findings'])}
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h2>Visualizations</h2>
                        <div class="grid">
                            {"".join(
                                f'<div><h3>{f.stem.replace("_", " ").title()}</h3><img src="{f}" /></div>'
                                for f in viz_files
                            )}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Detailed Theme Analysis</h2>
                        {"".join(
                            f'<div><h3>{bank}</h3><ul>'
                            + "".join(
                                f'<li><strong>{theme}:</strong> {", ".join(keywords[:5])}...</li>'
                                for theme, keywords in bank_themes.items()
                            )
                            + '</ul></div>'
                            for bank, bank_themes in themes.items()
                        )}
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
            logger.error(f"Report generation failed: {str(e)}")
            return None
            
           
    def run_full_pipeline(self, scrape_new=False):
        """Execute the complete analysis pipeline with robust error handling"""
        # Initialize default return values
        processed_data = pd.DataFrame()
        themes = {}
        insights = {
            'summary_metrics': {},
            'sentiment_trends': {},
            'theme_distributions': {},
            'key_findings': []
        }
        viz_dir = None
        report_file = None

        try:
            logger.info("=== Starting Pipeline Execution ===")
            
            # 1. Data Acquisition
            logger.info("Loading data...")
            try:
                raw_data = self.scrape_or_load_data(scrape_new)
                if raw_data.empty:
                    logger.error("No data available for analysis")
                    return processed_data, themes, insights, viz_dir, report_file
                logger.info(f"Loaded {len(raw_data)} records")
            except Exception as e:
                logger.error(f"Data loading failed: {str(e)}")
                return processed_data, themes, insights, viz_dir, report_file

            # 2. Data Preprocessing
            logger.info("Preprocessing data...")
            try:
                processed_data = self.preprocessor.preprocess(raw_data.copy())
                
                # Handle column naming
                text_col = next((col for col in ['review_text', 'review', 'text'] 
                            if col in processed_data.columns), None)
                if not text_col:
                    logger.error("No text column found in data")
                    return processed_data, themes, insights, viz_dir, report_file
                if text_col != 'review_text':
                    processed_data = processed_data.rename(columns={text_col: 'review_text'})
                    
                # Validate required columns
                required_cols = {'review_text', 'rating', 'bank'}
                if not required_cols.issubset(processed_data.columns):
                    missing = required_cols - set(processed_data.columns)
                    logger.error(f"Missing required columns: {missing}")
                    return processed_data, themes, insights, viz_dir, report_file
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                return processed_data, themes, insights, viz_dir, report_file

            # 3. Sentiment Analysis (numpy-free implementation)
            logger.info("Performing sentiment analysis...")
            try:
                valid_reviews = processed_data[processed_data['review_text'].str.strip().ne("")]
                if len(valid_reviews) == 0:
                    logger.error("No valid text for sentiment analysis")
                    return processed_data, themes, insights, viz_dir, report_file

                review_texts = valid_reviews['review_text'].tolist()
                sentiment_results = self.sentiment_analyzer.analyze(review_texts)
                logger.debug(f"Sentiment results type: {type(sentiment_results)}")

                # Initialize results dataframe
                processed_data = valid_reviews.copy()

                # Handle all possible return formats
                if isinstance(sentiment_results, pd.DataFrame):
                    if {'sentiment_score', 'sentiment_label'}.issubset(sentiment_results.columns):
                        processed_data = sentiment_results
                    else:
                        logger.error("Missing sentiment columns in DataFrame")
                        return processed_data, themes, insights, viz_dir, report_file
                
                elif isinstance(sentiment_results, (list, tuple)):
                    if isinstance(sentiment_results, tuple) and len(sentiment_results) == 2:
                        # Handle (scores, labels) tuple
                        scores, labels = sentiment_results
                        
                        # Convert string scores if needed
                        if isinstance(scores, str):
                            try:
                                scores = [float(x) for x in scores.split(',')]
                            except ValueError:
                                logger.warning("Couldn't parse scores string, using neutral scores")
                                scores = [0.0] * len(valid_reviews)
                        
                        # Ensure labels is a list
                        if isinstance(labels, (int, float)):
                            sentiment = 'POSITIVE' if labels > 0 else 'NEGATIVE' if labels < 0 else 'NEUTRAL'
                            labels = [sentiment] * len(valid_reviews)
                        elif isinstance(labels, str):
                            labels = labels.split(',') if ',' in labels else [labels] * len(valid_reviews)
                        
                        if len(scores) == len(valid_reviews) and len(labels) == len(valid_reviews):
                            processed_data['sentiment_score'] = scores
                            processed_data['sentiment_label'] = labels
                        else:
                            logger.error("Length mismatch in sentiment results")
                            return processed_data, themes, insights, viz_dir, report_file
                    else:
                        # Handle single list of scores
                        if len(sentiment_results) == len(valid_reviews):
                            processed_data['sentiment_score'] = list(sentiment_results)
                            processed_data['sentiment_label'] = [
                                'POSITIVE' if score > 0 else 'NEGATIVE' if score < 0 else 'NEUTRAL'
                                for score in sentiment_results
                            ]
                        else:
                            logger.error("Length mismatch in sentiment scores")
                            return processed_data, themes, insights, viz_dir, report_file
                
                elif isinstance(sentiment_results, (int, float)):
                    # Handle single numeric value
                    processed_data['sentiment_score'] = float(sentiment_results)
                    processed_data['sentiment_label'] = (
                        'POSITIVE' if sentiment_results > 0 else
                        'NEGATIVE' if sentiment_results < 0 else
                        'NEUTRAL'
                    )
                else:
                    logger.error(f"Unsupported sentiment results type: {type(sentiment_results)}")
                    return processed_data, themes, insights, viz_dir, report_file

            except Exception as e:
                logger.error(f"Sentiment analysis failed: {str(e)}")
                return processed_data, themes, insights, viz_dir, report_file

            # 4. Thematic Analysis
            logger.info("Performing thematic analysis...")
            try:
                themes = self.thematic_analyzer.analyze(processed_data)
                if not themes:
                    logger.warning("No themes extracted from data")
            except Exception as e:
                logger.error(f"Thematic analysis failed: {str(e)}")
                themes = {}

            # 5. Generate Insights
            logger.info("Generating insights...")
            try:
                insights = self.synthesize_insights(processed_data, themes)
            except Exception as e:
                logger.error(f"Insights generation failed: {str(e)}")
                insights = {
                    'summary_metrics': {},
                    'sentiment_trends': {},
                    'theme_distributions': {},
                    'key_findings': []
                }

            # 6. Create Visualizations
            logger.info("Creating visualizations...")
            try:
                viz_dir = self.visualize_results(processed_data, themes, insights)
            except Exception as e:
                logger.error(f"Visualization failed: {str(e)}")
                viz_dir = None

            # 7. Generate Report
            logger.info("Generating report...")
            try:
                report_file = self.generate_report(processed_data, themes, insights, viz_dir)
            except Exception as e:
                logger.error(f"Report generation failed: {str(e)}")
                report_file = None

            logger.info("=== Pipeline Completed Successfully ===")
            return processed_data, themes, insights, viz_dir, report_file

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return processed_data, themes, insights, viz_dir, report_file

def main():
    """Main entry point for the pipeline"""
    print("=== Ethiopian Bank Review Analysis Pipeline ===")
    
    try:
        pipeline = AnalysisPipeline()
        df, themes, insights, viz_dir, report_file = pipeline.run_full_pipeline()
        
        if df.empty:
            print("\nERROR: No data available for analysis")
            return
        
        print("\n=== Key Insights ===")
        for finding in insights['key_findings']:
            print(f"- {finding}")
        
        if viz_dir:
            print(f"\n=== Visualizations saved to: {viz_dir} ===")
        if report_file:
            print(f"\n=== Full report generated: {report_file} ===")
            
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
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