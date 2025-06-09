import pandas as pd
import sys
import json
from pathlib import Path
from datetime import datetime
from src.data_processing.scraper import BankReviewScraper
from src.data_processing.preprocessor import ReviewPreprocessor
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.thematic import ThematicAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AnalysisPipeline:
    def __init__(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        self.scraper = BankReviewScraper()
        self.preprocessor = ReviewPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.theme_analyzer = ThematicAnalyzer()
        logger.info("All components initialized")

    def _save_results(self, df, themes):
        """Save analysis results to output directory"""
        try:
            output_dir = Path(__file__).parent.parent / "data" / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save processed data with sentiment analysis
            data_file = output_dir / f"processed_reviews_{timestamp}.parquet"
            df.to_parquet(data_file)
            logger.info(f"Saved processed data to {data_file}")
            
            # Save themes
            themes_file = output_dir / f"themes_{timestamp}.json"
            with open(themes_file, 'w') as f:
                json.dump(themes, f, indent=2)
            logger.info(f"Saved themes to {themes_file}")
            
            # Save human-readable report
            report_file = output_dir / f"analysis_report_{timestamp}.csv"
            report_cols = ['bank', 'review', 'rating', 'sentiment_label', 'sentiment_score']
            df[report_cols].to_csv(report_file, index=False)
            logger.info(f"Saved analysis report to {report_file}")
            
            return data_file, themes_file, report_file
            
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            return None, None, None

    def run_full_pipeline(self, scrape_new=False):
        """Execute the complete analysis pipeline"""
        try:
            logger.info("=== Starting Pipeline Execution ===")
            
            # 1. Data Collection
            logger.info("[1/4] Data Collection Phase")
            if scrape_new:
                logger.info("Running fresh scrape...")
                if not self.scraper.scrape_all_banks():
                    logger.error("Data collection failed")
                    return pd.DataFrame(), {}
            
            # 2. Data Preprocessing
            logger.info("[2/4] Data Preprocessing Phase")
            try:
                processed_data = self.preprocessor.preprocess()
                if processed_data.empty:
                    logger.error("No data available after preprocessing")
                    return pd.DataFrame(), {}
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                return pd.DataFrame(), {}
            
            # 3. Sentiment Analysis
            logger.info("[3/4] Sentiment Analysis Phase")
            try:
                sentiment_results = processed_data['processed_text'].apply(
                    lambda x: self.sentiment_analyzer.analyze(x)
                )
                processed_data = processed_data.assign(
                    sentiment_label=sentiment_results.apply(lambda x: x[0]),
                    sentiment_score=sentiment_results.apply(lambda x: x[1])
                )
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {str(e)}")
                return processed_data, {}
            
            # 4. Thematic Analysis
            logger.info("[4/4] Thematic Analysis Phase")
            try:
                themes = self.theme_analyzer.analyze(processed_data) or {}
                logger.info(f"Identified {len(themes)} thematic clusters")
            except Exception as e:
                logger.error(f"Thematic analysis failed: {str(e)}")
                themes = {}
            
            # Save all results
            data_file, themes_file, report_file = self._save_results(processed_data, themes)
            if not all([data_file, themes_file, report_file]):
                logger.error("Some output files failed to save")
            
            logger.info("=== Pipeline Completed ===")
            return processed_data, themes
            
        except Exception as e:
            logger.error(f"Pipeline crashed: {str(e)}")
            return pd.DataFrame(), {}

def main():
    """Main execution function with user feedback"""
    print("=== Ethiopian Bank Review Analysis Pipeline ===")
    print("This may take several minutes to complete...\n")
    
    try:
        pipeline = AnalysisPipeline()
        
        # Ask user if they want fresh data
        scrape_new = input("Scrape new data? (y/n): ").lower() == 'y'
        
        # Run pipeline
        df, themes = pipeline.run_full_pipeline(scrape_new=scrape_new)
        
        if df is not None and not df.empty:
            # Show sample results
            print("\n=== Sample Results ===")
            print(df[['bank', 'review', 'sentiment_label']].head(3))
            
            # Show themes
            if themes:
                print("\n=== Identified Themes ===")
                for bank, bank_themes in themes.items():
                    print(f"\nBank: {bank}")
                    for theme, keywords in bank_themes.items():
                        print(f"- {theme}: {', '.join(keywords[:3])}...")
            
            # Show where files were saved
            output_dir = Path(__file__).parent.parent / "data" / "outputs"
            print("\n=== Output Files ===")
            print(f"Results saved to: {output_dir}")
            for f in output_dir.glob('*'):
                if f.is_file():
                    print(f"- {f.name}")
        else:
            print("\nERROR: Pipeline failed to produce results")
            
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    main()