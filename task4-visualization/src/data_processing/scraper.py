from google_play_scraper import app, Sort, reviews_all
import pandas as pd
from datetime import datetime
import time
import socket
from urllib.error import URLError
from src.config import settings
from src.config.constants import BANK_APPS
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BankReviewScraper:
    def __init__(self):
        self.max_retries = 3
        self.timeout = 30

    def safe_get_reviews(self, app_id):
        """Handle network issues with retries"""
        for attempt in range(self.max_retries):
            try:
                socket.setdefaulttimeout(self.timeout)
                for country in ['et', 'us']:  # Try Ethiopia first, then US
                    try:
                        return reviews_all(
                            app_id,
                            lang='en',
                            country=country,
                            sort=Sort.NEWEST,
                            count=settings.SCRAPE_LIMIT
                        )
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1} with country {country} failed: {e}")
                        time.sleep(3)
            except (URLError, socket.timeout) as e:
                logger.warning(f"Network error on attempt {attempt+1}: {e}")
                time.sleep(10 * (attempt + 1))
        return []

    def process_reviews(self, raw_reviews, bank_name):
        """Process raw reviews into structured format"""
        processed = []
        for review in raw_reviews:
            try:
                date_str = review.get('at', '')
                try:
                    date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d').strftime('%Y-%m-%d')
                except:
                    date = datetime.now().strftime('%Y-%m-%d')
                
                content = str(review.get('content', '')).strip()
                content = ' '.join(content.split())  # Normalize whitespace
                
                processed.append({
                    'review': content,
                    'rating': int(review.get('score', 0)),
                    'date': date,
                    'bank': bank_name,
                    'source': 'Google Play',
                    'thumbs_up': review.get('thumbsUpCount', 0)
                })
            except Exception as e:
                logger.error(f"Skipping malformed review: {e}")
        return processed

    def scrape_all_banks(self):
        """Main scraping function for all banks"""
        print("Starting bank scraping...")
        all_reviews = []
        
        for bank_name, app_ids in BANK_APPS.items():
            logger.info(f"Processing {bank_name}...")
            
            for app_id in app_ids:
                reviews = self.safe_get_reviews(app_id)
                if reviews:
                    processed = self.process_reviews(reviews, bank_name)
                    if processed:
                        all_reviews.extend(processed)
                        logger.info(f"Collected {len(processed)} reviews for {bank_name}")
                        break  # Success with this package ID
                
                time.sleep(5)
            time.sleep(15)
        
        if all_reviews:
            print(f"Scraped {len(all_reviews)} total reviews")
            df = pd.DataFrame(all_reviews)
            df = df.drop_duplicates(subset=['review', 'bank'])
            df = df[df['review'].str.len() > 3]
            df = df[df['rating'] > 0]
            df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
            
            output_path = settings.RAW_DATA_DIR / 'bank_reviews.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} reviews to {output_path}")
            return df
        else:
            logger.error("No valid reviews collected")
            return None