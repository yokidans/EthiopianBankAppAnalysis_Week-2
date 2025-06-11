import sys
import os
import logging
import cx_Oracle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('review_loader.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_dir = project_root / 'src'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

from src.oracle_connector.db_connection_manager import OracleConnectionManager

# Constants
CLEANED_DATA_PATH = os.path.join(project_root, 'data', 'cleaned_reviews.csv')
BATCH_SIZE = 500
DATE_FORMAT = 'YYYY-MM-DD'

def convert_to_date(date_value):
    """Convert various date formats to YYYY-MM-DD string"""
    if pd.isna(date_value) or date_value is None:
        return None
        
    # Handle numpy.datetime64
    if isinstance(date_value, np.datetime64):
        try:
            ts = pd.Timestamp(date_value)
            return ts.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Could not convert numpy datetime {date_value}: {e}")
            return None
            
    if isinstance(date_value, (datetime, pd.Timestamp)):
        return date_value.strftime('%Y-%m-%d')
        
    if isinstance(date_value, str):
        try:
            return datetime.strptime(date_value, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            pass
            
    if isinstance(date_value, (int, float, np.integer, np.floating)):
        try:
            if date_value > 59:
                date_value -= 1
            return (datetime(1899, 12, 30) + pd.Timedelta(days=date_value)).strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Could not convert numeric value {date_value} to date: {e}")
            return None
            
    logger.warning(f"Unrecognized date format: {type(date_value)} - {date_value}")
    return None

class ReviewDataLoader:
    """Handles loading of bank review data into Oracle database"""
    
    def __init__(self):
        self.connection = None
        self.bank_id_map = {}  # Store bank name to ID mapping
    
    def __enter__(self):
        self.connection = OracleConnectionManager.get_connection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            OracleConnectionManager.release_connection(self.connection)
    
    def load_banks(self, banks_data: List[Tuple]) -> bool:
        """Load bank data into the database"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            # First check if banks already exist
            cursor.execute("SELECT bank_name, bank_id FROM banks")
            existing_banks = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Prepare SQL for insert or update
            insert_sql = """
            INSERT INTO banks (
                bank_name, bank_code, website_url, established_date, headquarters
            ) VALUES (:1, :2, :3, TO_DATE(:4, 'YYYY-MM-DD'), :5)
            RETURNING bank_id INTO :6
            """
            
            update_sql = """
            UPDATE banks SET
                bank_code = :1,
                website_url = :2,
                established_date = TO_DATE(:3, 'YYYY-MM-DD'),
                headquarters = :4
            WHERE bank_name = :5
            """
            
            # Process each bank
            new_bank_ids = []
            for record in banks_data:
                bank_name = str(record[0])
                if bank_name in existing_banks:
                    # Update existing bank
                    cursor.execute(update_sql, (
                        str(record[1]),  # bank_code
                        str(record[2]),  # website_url
                        convert_to_date(record[3]),  # established_date
                        str(record[4]),  # headquarters
                        bank_name
                    ))
                    self.bank_id_map[bank_name] = existing_banks[bank_name]
                else:
                    # Insert new bank
                    bank_id_var = cursor.var(cx_Oracle.NUMBER)
                    cursor.execute(insert_sql, (
                        bank_name,
                        str(record[1]),  # bank_code
                        str(record[2]),  # website_url
                        convert_to_date(record[3]),  # established_date
                        str(record[4]),  # headquarters
                        bank_id_var
                    ))
                    new_id = bank_id_var.getvalue()[0]
                    self.bank_id_map[bank_name] = new_id
                    new_bank_ids.append(new_id)
            
            self.connection.commit()
            
            if new_bank_ids:
                logger.info(f"Inserted {len(new_bank_ids)} new banks")
            logger.info(f"Total banks in system: {len(self.bank_id_map)}")
            return True
            
        except cx_Oracle.DatabaseError as e:
            logger.error(f"Error loading banks: {e}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def load_reviews(self, reviews_data: List[Tuple]) -> bool:
        """Load review data into the database"""
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            # First verify all referenced banks exist
            valid_reviews = []
            missing_banks = set()
            
            for review in reviews_data:
                bank_name = review[0]
                if bank_name not in self.bank_id_map:
                    missing_banks.add(bank_name)
                    continue
                
                try:
                    date_value = convert_to_date(review[1])
                    if not date_value:
                        continue
                    
                    valid_reviews.append((
                        self.bank_id_map[bank_name],  # bank_id
                        date_value,                  # review_date
                        str(review[2]) if review[2] else None,  # reviewer_name
                        str(review[3]) if review[3] else None,  # review_title
                        str(review[4]),               # review_text
                        float(review[5]),             # rating
                        float(review[6]) if review[6] else 0,  # sentiment_score
                        float(review[7]) if review[7] else 0,  # sentiment_magnitude
                        str(review[8]) if review[8] else None,  # cleaned_text
                        str(review[9]) if review[9] else None   # topics
                    ))
                except Exception as e:
                    logger.error(f"Error formatting review for {bank_name}: {e}")
                    continue
            
            if missing_banks:
                logger.warning(f"Reviews skipped for non-existent banks: {missing_banks}")
            
            if not valid_reviews:
                logger.error("No valid reviews to insert")
                return False
            
            # Insert reviews
            sql = """
            INSERT INTO reviews (
                bank_id, review_date, reviewer_name, review_title, 
                review_text, rating, sentiment_score, sentiment_magnitude,
                cleaned_text, topics
            ) VALUES (
                :1, TO_DATE(:2, 'YYYY-MM-DD'), :3, :4, :5, :6, :7, :8, :9, :10
            )
            """
            
            total_inserted = 0
            for i in range(0, len(valid_reviews), BATCH_SIZE):
                batch = valid_reviews[i:i + BATCH_SIZE]
                cursor.executemany(sql, batch)
                self.connection.commit()
                total_inserted += len(batch)
                logger.info(f"Inserted batch of {len(batch)} reviews (total: {total_inserted})")
            
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
    def prepare_data(cleaned_data_path: str) -> Tuple[List[Tuple], List[Tuple]]:
        """Prepare data for database insertion with validation"""
        try:
            if not os.path.exists(cleaned_data_path):
                raise FileNotFoundError(f"CSV file not found at {cleaned_data_path}")
            
            logger.info(f"Loading data from {cleaned_data_path}")
            df = pd.read_csv(cleaned_data_path)
            
            # Convert numpy types and handle missing values
            df = df.replace({np.nan: None})
            for col in df.columns:
                if df[col].dtype == np.int64:
                    df[col] = df[col].astype(object)
            
            # Validate required columns
            required_columns = {
                'bank_name', 'bank_code', 'website_url', 'established_date', 'headquarters',
                'review_date', 'review_text', 'rating'
            }
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(f"CSV missing required columns: {missing}")
            
            # Convert date columns
            for date_col in ['established_date', 'review_date']:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Drop rows with invalid dates
            initial_count = len(df)
            df = df.dropna(subset=['established_date', 'review_date'])
            if len(df) < initial_count:
                logger.warning(f"Dropped {initial_count - len(df)} rows with invalid dates")
            
            # Prepare banks data
            banks_df = df[['bank_name', 'bank_code', 'website_url', 'established_date', 'headquarters']]
            banks_df = banks_df.drop_duplicates()
            
            # Prepare reviews data
            reviews_data = []
            for _, row in df.iterrows():
                reviews_data.append((
                    row['bank_name'],
                    row['review_date'],
                    row.get('reviewer_name'),
                    row.get('review_title'),
                    row['review_text'],
                    row['rating'],
                    row.get('sentiment_score'),
                    row.get('sentiment_magnitude'),
                    row.get('cleaned_text'),
                    row.get('topics')
                ))
            
            logger.info(f"Prepared {len(banks_df)} banks and {len(reviews_data)} reviews")
            return [tuple(x) for x in banks_df.to_records(index=False)], reviews_data
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
            raise

def test_connection() -> bool:
    """Test the database connection with simple query"""
    conn = None
    try:
        conn = OracleConnectionManager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT 'Connection successful' FROM dual")
        result = cursor.fetchone()
        
        logger.info(f"Database test successful: {result[0]}")
        return True
        
    except cx_Oracle.DatabaseError as e:
        error, = e.args
        logger.error(f"Connection test failed. Error Code: {error.code}\nMessage: {error.message}")
        return False
    finally:
        if conn:
            OracleConnectionManager.release_connection(conn)

def main() -> int:
    """Main execution function"""
    try:
        logger.info("Starting review data loading process")
        
        if not OracleConnectionManager.initialize_pool():
            raise RuntimeError("Failed to initialize connection pool")
        
        if not test_connection():
            raise RuntimeError("Database connection test failed")
        
        with ReviewDataLoader() as loader:
            banks_data, reviews_data = ReviewDataLoader.prepare_data(CLEANED_DATA_PATH)
            
            if not loader.load_banks(banks_data):
                raise RuntimeError("Failed to load bank data")
            
            if not loader.load_reviews(reviews_data):
                raise RuntimeError("Failed to load review data")
            
            logger.info(f"Successfully loaded {len(banks_data)} banks and {len(reviews_data)} reviews")
        
        return 0
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    finally:
        OracleConnectionManager.close_pool()
        logger.info("Connection pool closed")

if __name__ == "__main__":
    sys.exit(main())