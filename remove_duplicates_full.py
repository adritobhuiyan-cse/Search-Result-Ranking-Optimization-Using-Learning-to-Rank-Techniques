import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_path = os.path.join(BASE_DIR, "data", "raw")

input_file = os.path.join(raw_path, "top1000.dev")

chunk_number = 0
total_rows = 0

for chunk in pd.read_csv(
    input_file,
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    chunksize=50000
):
    
    total_rows += len(chunk)

    print(f"Chunk {chunk_number} processed")
    print(f"Total rows so far: {total_rows}")

print("Dataset loaded successfully.")