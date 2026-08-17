import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw")
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

# Load sample from top1000
top1000_sample = pd.read_csv(
    os.path.join(data_path, "top1000.dev"),
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    nrows=50000
)

sample_file = os.path.join(cleaned_path, "top1000_sample.csv")
top1000_sample.to_csv(sample_file, index=False)

print("Sample dataset saved successfully.")
print("File:", sample_file)
print("Shape:", top1000_sample.shape)