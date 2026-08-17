import os

files = [
    "data/raw/queries.dev.tsv",
    "data/raw/collection.tsv",
    "data/raw/top1000.dev",
    "data/raw/qrels.dev.tsv"
]

for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"{file} -> {size} bytes")
    else:
        print(f"{file} -> NOT FOUND")