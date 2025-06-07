# Ethiopian Bank App Review Analysis - Task 1

## 📌 Objective
Collect and preprocess customer reviews from three Ethiopian banking apps on Google Play Store to analyze customer satisfaction trends.

## 🛠️ Task 1: Data Collection & Preprocessing
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

📊 Expected Output
  #### Raw Data (per bank):
  #### reprocessing	< 1 min
  
## ⚠️ Troubleshooting
   Scraping Fails:

- Try VPN if getting regional restrictions

- Verify package names in scrape_reviews.py

- Authentication Errors:

## ✅ Success Criteria
- 400+ reviews collected per bank

- Clean data in standardized CSV format

- No duplicate reviews

- All dates normalized

