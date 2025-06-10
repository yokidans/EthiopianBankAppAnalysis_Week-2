# Database Configuration Template
# Rename to database_config.py and fill in your credentials
# DO NOT COMMIT THIS FILE WITH REAL CREDENTIALS

DB_CONFIG = {
    'user': 'bank_reviews',
    'password': 'your_password_here',
    'dsn': 'localhost/XEPDB1',  # Format: hostname/service_name
    'encoding': 'UTF-8',
    'min': 1,  # Minimum connections in pool
    'max': 5,  # Maximum connections in pool
    'increment': 1,  # Connection increment
    'threaded': True
}

# Oracle Client Configuration (if needed)
# ORACLE_CLIENT_PATH = "/path/to/instantclient"