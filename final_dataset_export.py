import pandas as pd
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

input_file = os.path.join(cleaned_path, "final_search_dataset.csv")
output_file = os.path.join(cleaned_path, "clean_search_dataset.csv")

print("Starting final dataset export...")

chunk_number = 0
total_rows = 0
first_chunk = True

for chunk in pd.read_csv(input_file, chunksize=50000):

    chunk_number += 1
    total_rows += len(chunk)

    # Select required columns
    final_chunk = chunk[[
        "qid",
        "pid",
        "query_clean",
        "passage_clean",
        "relevance"
    ]]

    # OPTIONAL: normalize click_count if exists
    if "click_count" in chunk.columns:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        final_chunk["click_count_norm"] = scaler.fit_transform(
            chunk[["click_count"]]
        )

    # Save output
    if first_chunk:
        final_chunk.to_csv(output_file, index=False, mode="w")
        first_chunk = False
    else:
        final_chunk.to_csv(output_file, index=False, mode="a", header=False)

    print(f"Chunk {chunk_number} processed | Total rows: {total_rows}")

print("\nFinal dataset export completed!")
print(f"Clean dataset saved: {total_rows} rows")
print("File:", output_file)