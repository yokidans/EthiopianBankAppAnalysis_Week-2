<<<<<<< HEAD
# Ethiopian Bank Review Analysis Pipeline

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive NLP pipeline for analyzing customer reviews of Ethiopian banks from the Google Play Store, featuring sentiment analysis, thematic modeling, and automated reporting.

## 🚀 Key Features

### 🔍 Data Processing
- **Automated Review Collection**
  - Scrapes reviews from multiple bank apps
  - Handles network errors and regional restrictions
  - Automatic duplicate removal and data cleaning

- **Advanced Text Preprocessing**
  - Lemmatization and stopword removal
  - Custom text normalization
  - Date standardization and metadata enrichment

### 📊 Analysis Capabilities
- **Sentiment Analysis**
  - Hybrid approach using DistilBERT with confidence thresholds
  - Scores categorized as Positive/Negative/Neutral
  - Aggregate sentiment by bank and rating

- **Thematic Analysis**
  - TF-IDF keyword extraction
  - Semantic clustering using spaCy
  - 5 predefined theme categories:
    - Account Access
    - Transactions  
    - User Experience
    - Customer Support
    - Fees & Charges

### 📈 Visualization & Reporting
- **Automated Visualizations**
  
       python
       plt.savefig(viz_dir / "sentiment_distribution.png")  # Sample output

  

  
=======
# Ethiopian Bank App Review Analysis - Task 1

## 📌 Objective
Collect and preprocess customer reviews from three Ethiopian banking apps on Google Play Store to analyze customer satisfaction trends.

## 🛠️ Task 1: Data Collection & Preprocessing
### 📂 Folder Structure
EthiopiaBankAppAnalysis/
### ├── data/
### │ ├── raw/ # Raw scraped data
### │ └── processed/ # Cleaned CSV files
### ├── src/
### │ ├── scrape_reviews.py # Main scraping script
### │ └── preprocess_reviews.py # Data cleaning script
### ├── .gitignore
### ├── requirements.txt
### └── README.md 

### 🔍 Target Banks
1. Commercial Bank of Ethiopia (CBE)
2. Bank of Abyssinia (BOA)
3. Dashen Bank

### ⚙️ Technical Implementation
**Scraping Script Features:**
- Uses `google-play-scraper` Python package
- Handles network errors with retry mechanism (3 attempts)
- Collects 400+ reviews per bank (1,200+ total)
- Extracts:
  - Review text
  - Star rating (1-5)
  - Review date
  - App version
  - Thumbs-up count

**Preprocessing Steps:**
1. Deduplication (same review from same bank)
2. Date standardization (YYYY-MM-DD format)
3. Text cleaning:
   - Remove extra whitespace
   - Handle missing values
4. Quality filtering:
   - Remove reviews < 3 characters
   - Exclude 0-star ratings (potential fake reviews)

### 🚀 How to Run
 bash
 
    1. Install dependencies
        pip install -r requirements.txt
    2. Run scraper (saves to data/raw/)
        python src/scrape_reviews.py
    3. Preprocess data (saves to data/processed/)
        python src/preprocess_reviews.py

## 📊 Expected Output
- Raw Data (per bank):
- Processed Data:

## ⏳ Time Estimate
- Step	Duration
- Scraping	15-30 mins
- Preprocessing	< 1 min
##⚠️ Troubleshooting
### Scraping Fails:

- Try VPN if getting regional restrictions

- Verify package names in scrape_reviews.py

## ✅ Success Criteria
- 400+ reviews collected per bank

- Clean data in standardized CSV format

- No duplicate reviews

- All dates normalized
>>>>>>> 35a1d91 (Your descriptive commit message about Task-2)
