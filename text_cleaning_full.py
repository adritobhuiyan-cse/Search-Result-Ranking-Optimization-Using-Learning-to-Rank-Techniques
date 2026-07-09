import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

input_file = os.path.join(cleaned_path, "top1000_no_missing.csv")
output_file = os.path.join(cleaned_path, "top1000_cleaned_text.csv")

first_chunk = True

for chunk in pd.read_csv(input_file, chunksize=50000):

    chunk["query_clean"] = chunk["query"].str.lower()
    chunk["passage_clean"] = chunk["passage"].str.lower()

    if first_chunk:
        chunk.to_csv(output_file, index=False)
        first_chunk = False
    else:
        chunk.to_csv(output_file, mode="a", header=False, index=False)

print("Text converted to lowercase.")