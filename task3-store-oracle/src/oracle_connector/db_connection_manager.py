import sys
import cx_Oracle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add src directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from config.database_config import DB_CONFIG

class OracleConnectionManager:
    """Manages Oracle database connections with pooling"""
    
    _pool = None
    
    @classmethod
    def initialize_pool(cls):
        """Initialize the connection pool with retry logic"""
        if cls._pool is None:
            try:
                # Initialize Oracle client if path is specified
                if 'ORACLE_CLIENT_PATH' in DB_CONFIG:
                    try:
                        cx_Oracle.init_oracle_client(lib_dir=DB_CONFIG['ORACLE_CLIENT_PATH'])
                    except cx_Oracle.ProgrammingError as e:
                        logger.warning(f"Oracle client already initialized: {e}")
                
                # Create connection pool
                cls._pool = cx_Oracle.SessionPool(
                    user=DB_CONFIG['user'],
                    password=DB_CONFIG['password'],
                    dsn=DB_CONFIG['dsn'],
                    min=DB_CONFIG.get('min', 1),
                    max=DB_CONFIG.get('max', 5),
                    increment=DB_CONFIG.get('increment', 1),
                    threaded=DB_CONFIG.get('threaded', True),
                    encoding=DB_CONFIG.get('encoding', 'UTF-8'),
                    getmode=cx_Oracle.SPOOL_ATTRVAL_WAIT
                )
                logger.info("Oracle connection pool initialized successfully")
                return True
                
            except cx_Oracle.DatabaseError as e:
                error, = e.args
                logger.error(f"Connection pool initialization failed. Error Code: {error.code}\nMessage: {error.message}")
                if error.code == 12154:  # TNS error
                    logger.error("Check your TNS names or connection string")
                elif error.code == 1017:  # Invalid credentials
                    logger.error("Check your username/password")
                return False
            except Exception as e:
                logger.error(f"Unexpected error during pool initialization: {str(e)}")
                return False
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool with error handling"""
        if cls._pool is None:
            if not cls.initialize_pool():
                raise RuntimeError("Cannot get connection - pool initialization failed")
        
        try:
            return cls._pool.acquire()
        except cx_Oracle.DatabaseError as e:
            error, = e.args
            logger.error(f"Failed to acquire connection. Error Code: {error.code}\nMessage: {error.message}")
            raise
    
    @classmethod
    def release_connection(cls, connection):
        """Release a connection back to the pool"""
        if cls._pool is not None and connection is not None:
            try:
                cls._pool.release(connection)
            except cx_Oracle.DatabaseError as e:
                error, = e.args
                logger.warning(f"Error releasing connection: {error.message}")
    
    @classmethod
    def close_pool(cls):
        """Close all connections in the pool"""
        if cls._pool is not None:
            try:
                cls._pool.close()
                cls._pool = None
                logger.info("Connection pool closed successfully")
            except cx_Oracle.DatabaseError as e:
                error, = e.args
                logger.error(f"Error closing pool: {error.message}")