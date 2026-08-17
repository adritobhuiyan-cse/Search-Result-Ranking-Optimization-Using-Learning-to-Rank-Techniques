import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(BASE_DIR, "data", "raw")
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(cleaned_path, exist_ok=True)

input_file = os.path.join(raw_path, "top1000.dev")
output_file = os.path.join(cleaned_path, "top1000_no_duplicates.csv")

seen_pairs = set()
total_rows = 0
kept_rows = 0
removed_duplicates = 0
chunk_number = 0

# write header only once
first_chunk = True

for chunk in pd.read_csv(
    input_file,
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    chunksize=50000
):
    chunk_number += 1
    total_rows += len(chunk)

    unique_rows = []

    for _, row in chunk.iterrows():
        pair = (row["qid"], row["pid"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_rows.append(row)
        else:
            removed_duplicates += 1

    if unique_rows:
        clean_chunk = pd.DataFrame(unique_rows)

        if first_chunk:
            clean_chunk.to_csv(output_file, index=False, mode="w")
            first_chunk = False
        else:
            clean_chunk.to_csv(output_file, index=False, mode="a", header=False)

        kept_rows += len(clean_chunk)

    print(f"Chunk {chunk_number} processed | Total rows: {total_rows} | Kept: {kept_rows} | Removed duplicates: {removed_duplicates}")

print("\nDuplicate removal complete.")
print("Input rows:", total_rows)
print("Rows kept:", kept_rows)
print("Duplicates removed:", removed_duplicates)
print("Saved file:", output_file)