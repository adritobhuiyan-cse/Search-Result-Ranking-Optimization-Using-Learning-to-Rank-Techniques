"""
STEP 2 — Data Cleaning  (Shafi's Role)
Reads raw_msmarco.csv → clean_search_dataset.csv
Run: python step2_clean_data.py
"""
import re
import pandas as pd

IN_FILE  = "raw_msmarco.csv"
OUT_FILE = "clean_search_dataset.csv"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def main():
    print("="*60)
    print("STEP 2 — Data Cleaning")
    print("="*60)

    print("\n[1/4] Loading raw_msmarco.csv...")
    df = pd.read_csv(IN_FILE, dtype={'qid':int,'pid':int,'relevance':int,
                                      'query':str,'passage':str})
    print(f"      Rows: {len(df):,} | Queries: {df['qid'].nunique():,}")

    print("[2/4] Validating relevance labels...")
    for val, cnt in df['relevance'].value_counts().sort_index().items():
        print(f"      relevance={val}: {cnt:,} ({100*cnt/len(df):.1f}%)")
    assert df['relevance'].sum() > 0, "ERROR: No relevant rows — re-run step1!"

    print("[3/4] Cleaning text...")
    df = df.drop_duplicates(['qid','pid']).dropna(subset=['query','passage'])
    df['query_clean']   = df['query'].apply(clean_text)
    df['passage_clean'] = df['passage'].apply(clean_text)
    df = df[(df['query_clean'].str.len()>0) & (df['passage_clean'].str.len()>0)]

    print("[4/4] Saving...")
    df[['qid','pid','query_clean','passage_clean','relevance']].to_csv(OUT_FILE, index=False)

    print(f"\nDONE — {OUT_FILE}")
    print(f"  Rows: {len(df):,} | Relevant: {df['relevance'].sum():,}")
    print("Next: python step3_feature_engineering.py")

if __name__ == "__main__":
    main()
