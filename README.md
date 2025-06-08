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

  

  
