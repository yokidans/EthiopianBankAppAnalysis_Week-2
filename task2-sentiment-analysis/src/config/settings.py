from pathlib import Path
from src.config.constants import BANK_APPS, SENTIMENT_THRESHOLDS

# Paths
PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"

# Analysis
BANKS = list(BANK_APPS.keys())
SCRAPE_LIMIT = 500  # Max reviews per bank