import pandas as pd
import os
import re

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

input_file = os.path.join(cleaned_path, "top1000_no_missing.csv")
output_file = os.path.join(cleaned_path, "top1000_cleaned_text.csv")

# Text cleaning function
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

print("Starting text cleaning process...")

chunk_number = 0
total_rows = 0
first_chunk = True

for chunk in pd.read_csv(input_file, chunksize=50000):

    chunk_number += 1
    total_rows += len(chunk)

    # Apply cleaning
    chunk["query_clean"] = chunk["query"].apply(clean_text)
    chunk["passage_clean"] = chunk["passage"].apply(clean_text)

    # Save to file
    if first_chunk:
        chunk.to_csv(output_file, index=False, mode="w")
        first_chunk = False
    else:
        chunk.to_csv(output_file, index=False, mode="a", header=False)

    print(f"Chunk {chunk_number} processed | Total rows processed: {total_rows}")

print("\nText cleaning completed successfully!")
print("Saved file:", output_file)