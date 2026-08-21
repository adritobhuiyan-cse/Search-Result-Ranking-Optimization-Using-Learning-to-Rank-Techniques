"""
STEP 3 — Feature Engineering  (Tahmid's Role)
Reads clean_search_dataset.csv → features_dataset.csv

Features (18 total):
  Text match:   query_len, doc_len, term_overlap, exact_match,
                keyword_density, query_avg_tf, passage_uniq_ratio
  Retrieval:    tfidf_cosine, bm25_score, idf_sum
  Positional:   term_overlap_bigram, first_occurrence_rank
  Proxy:        dwell_time_proxy, ctr
  NEW stronger: query_doc_len_ratio, jaccard_similarity,
                avg_idf_of_overlap_terms, tf_idf_sum_overlap

Run: python step3_feature_engineering.py
"""

import os, math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

IN_FILE  = "clean_search_dataset.csv"
OUT_FILE = "features_dataset.csv"


def avg_tf(text):
    tokens = text.split()
    if not tokens: return 0.0
    c = Counter(tokens)
    return sum(c.values()) / len(tokens)

def passage_uniq_ratio(text):
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def jaccard(q_terms, d_terms):
    if not q_terms or not d_terms: return 0.0
    inter = len(q_terms & d_terms)
    union = len(q_terms | d_terms)
    return inter / union if union else 0.0

def bigram_overlap(q_tokens, d_tokens):
    """Count shared bigrams between query and document."""
    def bigrams(tokens):
        return set(zip(tokens, tokens[1:]))
    qb = bigrams(q_tokens)
    db = bigrams(d_tokens)
    return len(qb & db)

def first_occurrence_rank(q_terms, d_tokens):
    """Position (0-indexed) of first query term in doc, normalised by doc len."""
    for i, t in enumerate(d_tokens):
        if t in q_terms:
            return i / max(len(d_tokens), 1)
    return 1.0  # no match → worst rank


def main():
    print("="*60)
    print("STEP 3 — Feature Engineering  (18 features)")
    print("="*60)

    print("\n[1/8] Loading data...")
    df = pd.read_csv(IN_FILE, dtype={'qid':int,'pid':int,'relevance':int,
                                      'query_clean':str,'passage_clean':str})
    df['query_clean']   = df['query_clean'].fillna('')
    df['passage_clean'] = df['passage_clean'].fillna('')
    print(f"      Rows: {len(df):,} | Queries: {df['qid'].nunique():,}")
    print(f"      Relevant: {df['relevance'].sum():,}")

    # Basic token features
    print("[2/8] Basic token features...")
    df['query_tokens']   = df['query_clean'].apply(str.split)
    df['passage_tokens'] = df['passage_clean'].apply(str.split)
    df['query_terms']    = df['query_tokens'].apply(set)
    df['passage_terms']  = df['passage_tokens'].apply(set)

    df['query_len']           = df['query_tokens'].apply(len)
    df['doc_len']             = df['passage_tokens'].apply(len)
    df['query_avg_tf']        = df['query_clean'].apply(avg_tf)
    df['passage_uniq_ratio']  = df['passage_clean'].apply(passage_uniq_ratio)
    df['query_doc_len_ratio'] = df['query_len'] / (df['doc_len'] + 1)

    df['term_overlap']    = df.apply(lambda r: len(r['query_terms'] & r['passage_terms']), axis=1)
    df['keyword_density'] = df.apply(
        lambda r: r['term_overlap'] / (r['doc_len'] + 1e-9), axis=1)
    df['exact_match']     = df.apply(
        lambda r: int(r['query_clean'] in r['passage_clean']), axis=1)
    df['jaccard_similarity'] = df.apply(
        lambda r: jaccard(r['query_terms'], r['passage_terms']), axis=1)
    df['term_overlap_bigram'] = df.apply(
        lambda r: bigram_overlap(r['query_tokens'], r['passage_tokens']), axis=1)
    df['first_occurrence_rank'] = df.apply(
        lambda r: first_occurrence_rank(r['query_terms'], r['passage_tokens']), axis=1)
    df['dwell_time_proxy'] = np.log1p(df['doc_len'])

    # TF-IDF 
    print("[3/8] Building TF-IDF on full corpus...")
    tfidf = TfidfVectorizer(max_features=60_000, sublinear_tf=True,
                            min_df=2, ngram_range=(1,1))
    doc_matrix   = tfidf.fit_transform(df['passage_clean'].tolist())
    query_matrix = tfidf.transform(df['query_clean'].tolist())
    vocab = tfidf.vocabulary_
    idf   = tfidf.idf_

    print("[4/8] TF-IDF cosine similarity (batched)...")
    from sklearn.metrics.pairwise import cosine_similarity
    cos_scores = np.zeros(len(df), dtype=np.float32)
    batch = 4000
    for s in range(0, len(df), batch):
        e = min(s+batch, len(df))
        cos_scores[s:e] = cosine_similarity(
            query_matrix[s:e], doc_matrix[s:e]).diagonal()
        if s % 20000 == 0:
            print(f"      {s:,}/{len(df):,}")
    df['tfidf_cosine'] = cos_scores

    print("[5/8] IDF-based features...")
    def idf_sum(q_tokens):
        return sum(idf[vocab[t]] for t in q_tokens if t in vocab)

    def avg_idf_overlap(q_terms, p_terms):
        shared = q_terms & p_terms
        if not shared: return 0.0
        return sum(idf[vocab[t]] for t in shared if t in vocab) / len(shared)

    def tfidf_sum_overlap(row):
        """Sum of TF-IDF weights of shared query terms in the document."""
        shared = row['query_terms'] & row['passage_terms']
        if not shared: return 0.0
        d_tokens = row['passage_tokens']
        d_len = len(d_tokens) + 1e-9
        total = 0.0
        for t in shared:
            if t not in vocab: continue
            tf = d_tokens.count(t) / d_len
            total += (1 + math.log(tf)) * idf[vocab[t]] if tf > 0 else 0
        return total

    df['idf_sum']                = df['query_tokens'].apply(idf_sum)
    df['avg_idf_of_overlap_terms'] = df.apply(
        lambda r: avg_idf_overlap(r['query_terms'], r['passage_terms']), axis=1)
    df['tf_idf_sum_overlap']     = df.apply(tfidf_sum_overlap, axis=1)

    #  BM25 per query group
    print("[6/8] BM25 per query group (this takes a few minutes)...")
    bm25_scores = np.zeros(len(df), dtype=np.float32)
    for qid_val, group_idx in df.groupby('qid').groups.items():
        grp = df.loc[group_idx]
        passages_tok = [p.split() for p in grp['passage_clean']]
        if not passages_tok: continue
        bm25 = BM25Okapi(passages_tok)
        q_tok = grp['query_clean'].iloc[0].split()
        scores = bm25.get_scores(q_tok)
        bm25_scores[group_idx] = scores
    df['bm25_score'] = bm25_scores

    #  CTR proxy 
    print("[7/8] CTR proxy...")
    df['ctr'] = df.groupby('qid')['relevance'].transform('mean')

    # Export 
    print("[8/8] Saving features_dataset.csv...")
    FEATURE_COLS = [
        'qid','pid','relevance',
        # text match
        'query_len','doc_len','query_avg_tf','passage_uniq_ratio',
        'query_doc_len_ratio','term_overlap','keyword_density',
        'exact_match','jaccard_similarity','term_overlap_bigram',
        'first_occurrence_rank',
        # retrieval
        'tfidf_cosine','bm25_score','idf_sum',
        'avg_idf_of_overlap_terms','tf_idf_sum_overlap',
        # proxy
        'dwell_time_proxy','ctr',
    ]
    out = df[FEATURE_COLS].dropna()
    out.to_csv(OUT_FILE, index=False)

    print(f"\nDONE — {OUT_FILE}")
    print(f"  Shape: {out.shape}")
    print(f"  Relevant: {(out['relevance']>0).sum():,}")
    print("\nFeature stats (non-ID columns):")
    feat_only = out.drop(columns=['qid','pid','relevance'])
    print(feat_only.describe().round(4).to_string())
    print("\nNext: python step4_train_model.py")


if __name__ == "__main__":
    main()
