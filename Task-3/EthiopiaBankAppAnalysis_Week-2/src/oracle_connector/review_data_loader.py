import cx_Oracle
import pandas as pd
import logging
from datetime import datetime
from db_connection_manager import OracleConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewDataLoader:
    """Handles loading of bank review data into Oracle database"""
    
    def __init__(self):
        self.connection = None
    
    def __enter__(self):
        self.connection = OracleConnectionManager.get_connection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            OracleConnectionManager.release_connection(self.connection)
    
    def load_banks(self, banks_data):
        """Load bank data into the database"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            # Prepare SQL for batch insert
            sql = """
            INSERT INTO banks (
                bank_name, bank_code, website_url, established_date, headquarters
            ) VALUES (:1, :2, :3, TO_DATE(:4, 'YYYY-MM-DD'), :5)
            """
            
            # Execute batch insert
            cursor.executemany(sql, banks_data)
            self.connection.commit()
            
            logger.info(f"Inserted {cursor.rowcount} banks into database")
            return True
            
        except cx_Oracle.DatabaseError as e:
            logger.error(f"Error loading banks: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def load_reviews(self, reviews_data):
        """Load review data into the database"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            # Prepare SQL for batch insert
            sql = """
            INSERT INTO reviews (
                bank_id, review_date, reviewer_name, review_title, 
                review_text, rating, sentiment_score, sentiment_magnitude,
                cleaned_text, topics
            ) VALUES (
                :1, TO_DATE(:2, 'YYYY-MM-DD'), :3, :4, :5, :6, :7, :8, :9, :10
            )
            """
            
            # Batch insert in chunks for better performance
            chunk_size = 1000
            total_inserted = 0
            
            for i in range(0, len(reviews_data), chunk_size):
                chunk = reviews_data[i:i + chunk_size]
                cursor.executemany(sql, chunk)
                self.connection.commit()
                total_inserted += len(chunk)
                logger.info(f"Inserted {len(chunk)} reviews (total: {total_inserted})")
            
            return True
            
        except cx_Oracle.DatabaseError as e:
            logger.error(f"Error loading reviews: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    @staticmethod
    def prepare_data(cleaned_data_path):
        """Prepare data for database insertion"""
        try:
            # Read cleaned data
            df = pd.read_csv(cleaned_data_path)
            
            # Prepare banks data
            banks_df = df[['bank_name', 'bank_code', 'website_url', 'established_date', 'headquarters']]
            banks_df = banks_df.drop_duplicates()
            banks_data = [tuple(x) for x in banks_df.to_records(index=False)]
            
            # Create bank_id mapping
            bank_id_map = {name: idx+1 for idx, name in enumerate(banks_df['bank_name'].unique())}
            
            # Prepare reviews data
            reviews_data = []
            for _, row in df.iterrows():
                reviews_data.append((
                    bank_id_map[row['bank_name']],  # bank_id
                    row['review_date'],
                    row.get('reviewer_name', None),
                    row.get('review_title', None),
                    row.get('review_text', None),
                    row.get('rating', None),
                    row.get('sentiment_score', None),
                    row.get('sentiment_magnitude', None),
                    row.get('cleaned_text', None),
                    row.get('topics', None)
                ))
            
            return banks_data, reviews_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

def main():
    """Main execution function"""
    try:
        # Initialize connection pool
        OracleConnectionManager.initialize_pool()
        
        # Prepare data (replace with your actual cleaned data path)
        cleaned_data_path = "data/cleaned_reviews.csv"
        
        # Create loader instance
        with ReviewDataLoader() as loader:
            # Prepare data
            banks_data, reviews_data = loader.prepare_data(cleaned_data_path)
            
            # Load data
            if loader.load_banks(banks_data):
                logger.info("Bank data loaded successfully")
            
            if loader.load_reviews(reviews_data):
                logger.info("Review data loaded successfully")
                
    except Exception as e:
        logger.error(f"Fatal error in data loading: {e}")
    finally:
        OracleConnectionManager.close_pool()

if __name__ == "__main__":
    main()