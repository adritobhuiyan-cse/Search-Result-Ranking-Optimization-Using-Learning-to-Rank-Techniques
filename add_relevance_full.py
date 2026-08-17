import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")
raw_path = os.path.join(BASE_DIR, "data", "raw")

input_file = os.path.join(cleaned_path, "top1000_cleaned_text.csv")
qrels_file = os.path.join(raw_path, "qrels.dev.tsv")
output_file = os.path.join(cleaned_path, "final_search_dataset.csv")

print("Loading qrels (relevance labels)...")

# Load qrels (small file → safe to load fully)
qrels = pd.read_csv(
    qrels_file,
    sep="\t",
    names=["qid", "unused", "pid", "relevance"]
)[["qid", "pid", "relevance"]]

print("Qrels loaded:", qrels.shape)

chunk_number = 0
total_rows = 0
first_chunk = True

print("\nStarting merge process...")

for chunk in pd.read_csv(input_file, chunksize=50000):

    chunk_number += 1
    total_rows += len(chunk)

    # Merge relevance
    chunk = chunk.merge(qrels, on=["qid", "pid"], how="left")

    # Fill missing relevance → 0
    chunk["relevance"] = chunk["relevance"].fillna(0).astype(int)

    # Save output
    if first_chunk:
        chunk.to_csv(output_file, index=False, mode="w")
        first_chunk = False
    else:
        chunk.to_csv(output_file, index=False, mode="a", header=False)

    print(f"Chunk {chunk_number} processed | Total rows: {total_rows}")

print("\nRelevance merging completed!")
print("Final dataset saved at:", output_file)