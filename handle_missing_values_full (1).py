import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

input_file = os.path.join(cleaned_path, "top1000_no_duplicates.csv")
output_file = os.path.join(cleaned_path, "top1000_no_missing.csv")

total_rows = 0
kept_rows = 0
removed_rows = 0
chunk_number = 0
first_chunk = True

for chunk in pd.read_csv(input_file, chunksize=50000):
    chunk_number += 1
    total_rows += len(chunk)

    # count rows before cleaning
    before_rows = len(chunk)

    # remove rows where query or passage is missing
    chunk = chunk.dropna(subset=["query", "passage"])

    # fill remaining missing values safely
    chunk["query"] = chunk["query"].fillna("")
    chunk["passage"] = chunk["passage"].fillna("")

    after_rows = len(chunk)
    removed_rows += (before_rows - after_rows)
    kept_rows += after_rows

    if first_chunk:
        chunk.to_csv(output_file, index=False, mode="w")
        first_chunk = False
    else:
        chunk.to_csv(output_file, index=False, mode="a", header=False)

    print(f"Chunk {chunk_number} processed | Total rows: {total_rows} | Kept: {kept_rows} | Removed missing rows: {removed_rows}")

print("\nMissing value handling complete.")
print("Input rows:", total_rows)
print("Rows kept:", kept_rows)
print("Rows removed:", removed_rows)
print("Saved file:", output_file)