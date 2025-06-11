import pandas as pd

# Load your original CSV
df = pd.read_csv("clean_reviews.csv")

# Map bank names to their respective details
bank_details = {
    "Commercial Bank of Ethiopia": {
        "bank_code": "CBE",
        "headquarters": "Addis Ababa",
        "website_url": "https://www.combanketh.et",
        "established_date": "1942"
    },
    "Bank of Abyssinia": {
        "bank_code": "BOA",
        "headquarters": "Addis Ababa",
        "website_url": "https://www.bankofabyssinia.com",
        "established_date": "1996"
    },
    "Dashen Bank": {
        "bank_code": "DASH",
        "headquarters": "Addis Ababa",
        "website_url": "https://www.dashenbanksc.com",
        "established_date": "1995"
    }
}

# Assign bank details based on the 'bank' column
df["bank_code"] = df["bank"].map(lambda x: bank_details.get(x, {}).get("bank_code", "UNKNOWN"))
df["headquarters"] = df["bank"].map(lambda x: bank_details.get(x, {}).get("headquarters", "Unknown"))
df["website_url"] = df["bank"].map(lambda x: bank_details.get(x, {}).get("website_url", "Unknown"))
df["established_date"] = df["bank"].map(lambda x: bank_details.get(x, {}).get("established_date", "Unknown"))

# Rename columns to match expected format (if needed)
df = df.rename(columns={
    "bank": "bank_name",
    "review": "review_text",
    "date": "review_date"
})

# Save the processed CSV
df.to_csv("cleaned_reviews.csv", index=False)

print("CSV preprocessing complete! Saved as 'cleaned_reviews.csv'.")