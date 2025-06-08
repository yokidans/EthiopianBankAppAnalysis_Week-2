# Ethiopian Bank Review Analysis Pipeline

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
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

  ```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#f5f5f5', 'edgeLabelBackground':'#fff'}}}%%
graph TD
    A[Data Collection] --> B[Preprocessing]
    B --> C[Sentiment Analysis]
    C --> D[Thematic Analysis]
    D --> E[Insight Synthesis]
    E --> F[Visualization]
    F --> G[Report Generation]
    
    %% Styling
    classDef default fill:#f0f8ff,stroke:#4682b4,stroke-width:2px,color:#333;
    classDef process fill:#e6f3ff,stroke:#5b9bd5,stroke-width:2px;
    class A,B,C,D,E,F,G process;
    
    %% Optional click actions (for interactive docs)
    click A "https://example.com/data-collection" "Data Collection Docs"
    click G "https://example.com/reporting" "Report Generation Docs"
```

  
