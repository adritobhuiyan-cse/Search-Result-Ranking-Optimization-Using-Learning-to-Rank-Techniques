import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw")

print("Dataset path:", data_path)

# Load queries
queries = pd.read_csv(
    os.path.join(data_path, "queries.dev.tsv"),
    sep="\t",
    names=["qid", "query"]
)

# Load only small sample of collection
passages = pd.read_csv(
    os.path.join(data_path, "collection.tsv"),
    sep="\t",
    names=["pid", "passage"],
    nrows=1000
)

# Load only sample of top1000 because file is huge
top1000 = pd.read_csv(
    os.path.join(data_path, "top1000.dev"),
    sep="\t",
    names=["qid", "pid", "query", "passage"],
    nrows=5000
)

# Load qrels
qrels = pd.read_csv(
    os.path.join(data_path, "qrels.dev.tsv"),
    sep="\t",
    names=["qid", "unused", "pid", "relevance"]
)

print("\nDataset Loaded Successfully")
print("Queries shape:", queries.shape)
print("Passages sample shape:", passages.shape)
print("Top1000 sample shape:", top1000.shape)
print("Qrels shape:", qrels.shape)

print("\nQueries preview:")
print(queries.head())

print("\nPassages preview:")
print(passages.head())

print("\nTop1000 preview:")
print(top1000.head())

print("\nQrels preview:")
print(qrels.head())



# check datatypes

print("\nQueries data types:")
print(queries.dtypes)

print("\nPassages data types:")
print(passages.dtypes)

print("\nTop1000 data types:")
print(top1000.dtypes)

print("\nQrels data types:")
print(qrels.dtypes)


# check for missinf values

print("\nMissing values in queries:")
print(queries.isnull().sum())

print("\nMissing values in passages sample:")
print(passages.isnull().sum())

print("\nMissing values in top1000 sample:")
print(top1000.isnull().sum())

print("\nMissing values in qrels:")
print(qrels.isnull().sum())