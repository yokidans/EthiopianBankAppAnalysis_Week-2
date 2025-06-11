import sys
import logging
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from oracle_connector.db_connection_manager import OracleConnectionManager

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

def test_connection():
    """Test the database connection with simple query"""
    conn = None
    try:
        conn = OracleConnectionManager.get_connection()
        cursor = conn.cursor()
        
        # Execute a simple query to verify connection
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

def main():
    """Main execution function"""
    try:
        logger.info("Starting database connection test")
        
        # Initialize and test connection
        if not OracleConnectionManager.initialize_pool():
            raise RuntimeError("Failed to initialize connection pool")
        
        if test_connection():
            logger.info("All database tests completed successfully")
        else:
            logger.error("Database connection test failed")
            
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        return 1
    finally:
        OracleConnectionManager.close_pool()
    return 0

if __name__ == "__main__":
    sys.exit(main())