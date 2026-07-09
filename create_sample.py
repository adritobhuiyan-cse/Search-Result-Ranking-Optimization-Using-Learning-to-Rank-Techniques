import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw")

print("Loading s")


top1000_sample = pd.read_csv(
    os.path.join(data_path, "top1000.dev"),
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    nrows=10000
)

print("Dataset loaded successfully.")
print("Shape:", top1000_sample.shape)