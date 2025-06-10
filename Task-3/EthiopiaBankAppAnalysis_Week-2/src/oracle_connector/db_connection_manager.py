import cx_Oracle
from config.database_config import DB_CONFIG

class OracleConnectionManager:
    """Manages Oracle database connections and connection pooling"""
    
    _pool = None
    
    @classmethod
    def initialize_pool(cls):
        """Initialize the connection pool"""
        if cls._pool is None:
            try:
                # Initialize Oracle client if needed
                if hasattr(DB_CONFIG, 'ORACLE_CLIENT_PATH'):
                    cx_Oracle.init_oracle_client(lib_dir=DB_CONFIG['ORACLE_CLIENT_PATH'])
                
                cls._pool = cx_Oracle.SessionPool(
                    user=DB_CONFIG['user'],
                    password=DB_CONFIG['password'],
                    dsn=DB_CONFIG['dsn'],
                    min=DB_CONFIG['min'],
                    max=DB_CONFIG['max'],
                    increment=DB_CONFIG['increment'],
                    encoding=DB_CONFIG['encoding'],
                    threaded=DB_CONFIG['threaded']
                )
                print("Oracle connection pool initialized")
            except cx_Oracle.DatabaseError as e:
                print(f"Failed to initialize connection pool: {e}")
                raise
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool"""
        if cls._pool is None:
            cls.initialize_pool()
        return cls._pool.acquire()
    
    @classmethod
    def release_connection(cls, connection):
        """Release a connection back to the pool"""
        if cls._pool is not None and connection is not None:
            cls._pool.release(connection)
    
    @classmethod
    def close_pool(cls):
        """Close all connections in the pool"""
        if cls._pool is not None:
            cls._pool.close()
            cls._pool = None
            print("Oracle connection pool closed")