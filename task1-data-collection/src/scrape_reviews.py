from google_play_scraper import app, Sort, reviews_all
import pandas as pd
from datetime import datetime
import time
import socket
from urllib.error import URLError

# Updated with verified package names (June 2024)
BANK_APPS = {
    'Commercial Bank of Ethiopia': ['com.combanketh.mobilebanking'],
    'Bank of Abyssinia': ['com.boa.boaMobileBanking'],
    'Dashen Bank': ['com.dashen.dashensuperapp'],
    'Awash Bank': ['com.awashbank.mobilebanking'],
    'Nib International Bank': ['com.nibinternationalbank.mobilebanking']
}

def safe_get_reviews(app_id, max_retries=3):
    """Handle network issues with retries and regional restrictions"""
    for attempt in range(max_retries):
        try:
            socket.setdefaulttimeout(30)
            # Try with Ethiopia first, then fallback to US
            for country in ['et', 'us']:
                try:
                    return reviews_all(
                        app_id,
                        lang='en',
                        country=country,
                        sort=Sort.NEWEST,
                        count=500  # Increased from 400
                    )
                except Exception as e:
                    print(f"Attempt {attempt + 1} with country {country} failed: {str(e)}")
                    time.sleep(3)
        except (URLError, socket.timeout) as e:
            print(f"Network error on attempt {attempt + 1}: {str(e)}")
            time.sleep(10 * (attempt + 1))  # Longer backoff
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            break
    return []

def process_reviews(raw_reviews, bank_name):
    """Enhanced review processing with better error handling"""
    processed = []
    for review in raw_reviews:
        try:
            # Handle various date formats
            date_str = review.get('at', '')
            if isinstance(date_str, str):
                try:
                    date = datetime.strptime(date_str.split('T')[0], '%Y-%m-%d').strftime('%Y-%m-%d')
                except:
                    date = datetime.now().strftime('%Y-%m-%d')
            else:
                date = datetime.now().strftime('%Y-%m-%d')
                
            # Clean review text
            content = str(review.get('content', '')).strip()
            content = ' '.join(content.split())  # Remove extra whitespace
                
            processed.append({
                'review': content,
                'rating': int(review.get('score', 0)),
                'date': date,
                'bank': bank_name,
                'source': 'Google Play Store',
                'app_version': review.get('reviewCreatedVersion', 'N/A'),
                'thumbs_up': review.get('thumbsUpCount', 0)
            })
        except Exception as e:
            print(f"Skipping malformed review: {str(e)}")
    return processed

def verify_app_exists(app_id):
    """Check if app exists before scraping"""
    try:
        app_info = app(app_id)
        return {
            'exists': True,
            'title': app_info['title'],
            'rating': app_info.get('score', 'N/A'),
            'installs': app_info.get('installs', 'N/A')
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }

def main():
    all_reviews = []
    failed_banks = []
    
    print("🏦 Ethiopian Bank Review Scraper v2.0")
    print("="*40)
    
    for bank_name, app_ids in BANK_APPS.items():
        print(f"\n⏳ Processing {bank_name}...")
        bank_success = False
        
        for app_id in app_ids:
            print(f"🔍 Checking package: {app_id}")
            
            # Verify app exists
            app_check = verify_app_exists(app_id)
            if not app_check['exists']:
                print(f"⚠️ Package not available: {app_id} | Error: {app_check.get('error', 'Unknown')}")
                continue
                
            print(f"✅ Found: {app_check['title']} | Rating: {app_check['rating']} | Installs: {app_check['installs']}")
            
            # Get reviews with retries
            reviews = safe_get_reviews(app_id)
            if reviews:
                processed = process_reviews(reviews, bank_name)
                if processed:
                    all_reviews.extend(processed)
                    print(f"✔ Collected {len(processed)} reviews (from {len(reviews)} raw reviews)")
                    bank_success = True
                    break  # Success with this package ID
                else:
                    print("⚠️ No valid reviews could be processed")
            else:
                print("⚠️ No reviews collected (possible regional restriction)")
                
            time.sleep(5)  # Short delay between attempts
        
        if not bank_success:
            failed_banks.append(bank_name)
        time.sleep(15)  # Longer delay between banks
    
    # Create and clean DataFrame
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        
        # Enhanced cleaning
        df = df.drop_duplicates(subset=['review', 'bank'], keep='first')
        df = df[df['review'].str.len() > 3]  # Remove very short reviews
        df = df[df['rating'] > 0]  # Remove 0-star ratings (often fake)
        
        # Add metadata
        df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'ethiopian_bank_reviews_{timestamp}.csv'
        df.to_csv(filename, index=False, encoding='utf-8')
        
        print("\n" + "="*40)
        print(f"💾 Successfully saved {len(df)} reviews to {filename}")
        print("="*40)
        print("\nSummary Statistics:")
        print(df.groupby('bank')['rating'].describe())
        
        if failed_banks:
            print(f"\n❌ Failed to collect reviews for: {', '.join(failed_banks)}")
    else:
        print("\n❌ No valid reviews collected for any bank")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")