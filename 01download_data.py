"""
STEP 1 — Download MS MARCO from Hugging Face (NO Azure, NO manual download)
Requires: pip install datasets

How it works:
  - Downloads microsoft/ms_marco (v1.1) validation split from Hugging Face
  - Each row has: query, passages (list of up to 10 passages), is_selected (relevance)
  - Explodes into one row per (query, passage) pair
  - is_selected == 1 means the user clicked it = relevant

Run: python step1_download_data.py
"""

import os
import pandas as pd

OUT_FILE = "raw_msmarco.csv"
MAX_QUERIES = 6000   # ~60K rows total. Increase to 10000 for more data.

def main():
    print("=" * 60)
    print("STEP 1 — Download MS MARCO from Hugging Face")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n[ERROR] 'datasets' not installed. Run: pip install datasets")
        return

    print("\n[1/3] Downloading microsoft/ms_marco v1.1 (validation split)...")
    print("      ~50 MB download. This may take 1-3 minutes...")

    # datasets 4.x removed trust_remote_code
    try:
        ds = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try: pip install --upgrade datasets huggingface_hub")
        print("  3. If HuggingFace is blocked, run: python step1_offline_fallback.py")
        return

    print(f"      Downloaded {len(ds):,} queries")

    # ── Explode ────────────────────────────────────────────────
    print(f"\n[2/3] Building (query, passage) pairs (first {MAX_QUERIES:,} queries)...")
    records = []
    for i, row in enumerate(ds):
        if i >= MAX_QUERIES:
            break
        qid      = int(row['query_id'])
        query    = str(row['query'])
        passages = row['passages']['passage_text']
        selected = row['passages']['is_selected']

        for offset, (passage, is_sel) in enumerate(zip(passages, selected)):
            records.append({
                'qid':       qid,
                'pid':       qid * 100 + offset,
                'query':     query,
                'passage':   str(passage),
                'relevance': int(is_sel),
            })

    df = pd.DataFrame(records)
    print(f"      Total rows: {len(df):,}")
    print(f"      Unique queries: {df['qid'].nunique():,}")

    rel_dist = df['relevance'].value_counts()
    print(f"\n      Relevance distribution:")
    for val in sorted(rel_dist.index):
        cnt = rel_dist[val]
        print(f"        relevance={val}: {cnt:,} rows ({100*cnt/len(df):.1f}%)")

    if df['relevance'].sum() == 0:
        print("\n  [ERROR] No relevant rows! Something went wrong.")
        return

    print(f"\n      OK: {df['relevance'].sum():,} relevant (query, passage) pairs")

    # ── Save ───────────────────────────────────────────────────
    print(f"\n[3/3] Saving to {OUT_FILE}...")
    df.to_csv(OUT_FILE, index=False)
    size_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"      Saved: {size_mb:.1f} MB")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"  File: {OUT_FILE}")
    print(f"  Rows: {len(df):,} | Queries: {df['qid'].nunique():,} | Relevant: {df['relevance'].sum():,}")
    print(f"\nNext step: python step2_clean_data.py")


if __name__ == "__main__":
    main()
