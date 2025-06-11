# Ethiopian Bank App Review Analysis
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
Project Overview
markdown
## 📌 Project Overview
A comprehensive NLP pipeline analyzing customer reviews from three major Ethiopian banking apps:
1. Commercial Bank of Ethiopia (CBE)
2. Bank of Abyssinia (BOA)
3. Dashen Bank

The project delivers actionable insights through four key phases:
Branch Structure Summary
markdown
## 🌿 Branch Architecture
| Branch | Purpose | Key Deliverables |
|--------|---------|------------------|
| `data-collection` | Raw data acquisition | Scraped reviews, cleaned datasets |
| `sentiment-analysis` | Emotional tone evaluation | Sentiment scores, classification models |
| `oracle-integration` | Data storage solution | Database schema, loading scripts |
| `visualization` | Insight presentation | Dashboards, analytical reports |

# Detailed Task Integration

## 🔍 Data Collection (Task 1)
**Objective**: Acquire and preprocess 1,200+ customer reviews (400+ per bank)

**Technical Implementation**:
- `google-play-scraper` Python package
- Automated retry mechanism (3 attempts)
- Outputs to `/data/raw/` and `/data/processed/`

**Preprocessing Pipeline**:
1. Deduplication
2. Date standardization (YYYY-MM-DD)
3. Text cleaning:
   - Whitespace normalization
   - Special character handling

**How to Run**:

    pip install -r requirements.txt
    python src/scrape_reviews.py
    python src/preprocess_reviews.py

## 😊 Sentiment Analysis (Task 2)
**Methodology**:
- Hybrid DistilBERT model with confidence thresholds
- Three-class classification: Positive/Negative/Neutral

**Key Features**:
- Sentiment distribution by bank
- Rating-sentiment correlation
- Thematic sentiment analysis

**Output Structure**:
### results/
### ├── sentiment_scores.csv
### ├── model_metrics.json
### └── analysis_report.csv

## 💾 Oracle Integration (Task 3)
**Database Schema**:
- Tables: `reviews`, `sentiments`, `themes`
- Relationships: One-to-many (review-to-analysis)

**ETL Process**:
1. CSV → Pandas DataFrame
2. Data validation
3. Batch loading (1000 rows/commit)

**Configuration**:
- Connection details in `config/db.ini`
- SQL scripts in `migration_scripts/`

## 📊 Visualization (Task 4)
**Dashboard Components**:
1. Sentiment chart by bank
2. Theme frequency treemap
3. Rating distribution over time

**Technical Stack**:
- Plotly for interactive visuals
- Automated HTML report generation

**Access**:
streamlit run dashboard/app.py
text

## Unified Project Structure

## 🗂 Project Structure
EthiopianBankAppAnalysis/
### ├── data/ # All data assets
### │ ├── raw/ # Raw scraped data (Task 1)
### │ ├── processed/ # Cleaned datasets (Task 1)
### │ └── results/ # Analysis outputs (Tasks 2-4)
### ├── src/
### │ ├── collection/ # Task 1 scripts
### │ ├── analysis/ # Task 2 scripts
### │ ├── database/ # Task 3 scripts
### │ └── visualization/ # Task 4 scripts
### ├── docs/ # Documentation
### ├── requirements.txt # Python dependencies
### └── README.md # This file

text
Consolidated How-To Guide
markdown
## 🚀 Getting Started

### Installation

    git clone https://github.com/yokidans/EthiopianBankAppAnalysis_Week-2.git
    cd EthiopianBankAppAnalysis_Week-2
    git checkout main
    pip install -r requirements.txt
Workflow
Data Collection → data-collection branch

Analysis → sentiment-analysis branch

Database Setup → oracle-integration branch

Visualization → visualization branch

## Troubleshooting Matrix

## ⚠️ Common Issues

| Problem | Solution | Affected Task |
|---------|----------|---------------|
| Scraping fails | Use VPN, verify package names | Task 1 |
| Sentiment misclassification | Adjust confidence thresholds | Task 2 |
| DB connection issues | Verify credentials in db.ini | Task 3 |
| Visualization loading slow | Reduce dataset sample size | Task 4 |
Success Metrics
markdown
## ✅ Validation Criteria

| Task | Success Metrics |
|------|-----------------|
| 1 | 400+ clean reviews/bank, no duplicates |
| 2 | 85%+ sentiment accuracy, comprehensive report |
| 3 | All data loaded, query response <2s |
| 4 | Interactive dashboards, auto-generated reports |
