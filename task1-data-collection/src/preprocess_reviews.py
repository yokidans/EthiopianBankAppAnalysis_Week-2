import pandas as pd
from datetime import datetime

def preprocess_data(input_file='bank_reviews.csv', output_file='processed_bank_reviews.csv'):
    """
    Preprocess the scraped review data
    """
    # Load data
    df = pd.read_csv(input_file)
    
    print(f"Initial data shape: {df.shape}")
    
    # 1. Remove duplicates (again, just to be sure)
    df = df.drop_duplicates(subset=['review', 'bank'], keep='first')
    
    # 2. Handle missing data
    df['review'] = df['review'].fillna('')
    df['rating'] = df['rating'].fillna(0)
    
    # 3. Normalize dates (ensure they're in YYYY-MM-DD format)
    try:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error processing dates: {str(e)}")
        # Try to handle different date formats if needed
    
    # 4. Clean review text
    df['review'] = df['review'].str.strip()  # Remove leading/trailing whitespace
    
    # 5. Add a year-month column for easier time-based analysis
    try:
        df['year_month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    except:
        df['year_month'] = 'unknown'
    
    print("\nData after preprocessing:")
    print(f"- Total reviews: {len(df)}")
    print(f"- By bank:")
    print(df['bank'].value_counts())
    
    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"\nProcessed data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    preprocess_data()