#!/bin/bash
source ../venv/bin/activate
python -m src.data_processing.scraper
deactivate