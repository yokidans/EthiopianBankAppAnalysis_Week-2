#!/bin/bash
source ../venv/bin/activate
python -m src.pipeline --scrape
deactivate