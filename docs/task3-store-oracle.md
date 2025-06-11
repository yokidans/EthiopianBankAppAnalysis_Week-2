# Bank Review Data Ingestion Pipeline

This project ingests, cleans, and stores bank review data in an Oracle database.

## Features
- Oracle database schema for bank reviews
- Python data loading scripts
- Complete documentation

## Setup
1. Install Oracle XE (see `docs/oracle_xe_setup_guide.md`)
2. Create database user (see schema scripts)
3. Install Python requirements: `pip install -r environment/requirements.txt`

## Usage
1. Configure database connection in `src/config/database_config_template.py`
2. Run data loader: `python src/oracle_connector/review_data_loader.py`