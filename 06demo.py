"""
STEP 6 — Two-Stage Search Ranking Demo
  Stage 1: BM25 retrieves top-100 candidates from corpus
  Stage 2: LambdaMART re-ranks with all 18 features

Run: python step6_demo.py
"""

import json, pickle, re, math, warnings
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

warnings.filterwarnings("ignore")

MODEL_FILE  = "best_model.pkl"
META_FILE   = "model_metadata.json"
CORPUS_FILE = "clean_search_dataset.csv"
N_CANDS     = 100   # BM25 retrieves this many
TOP_K       = 10    # show this many after rerank


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def avg_tf(text):
    t = text.split()
    return sum(Counter(t).values())/len(t) if t else 0.0

def passage_uniq_ratio(text):
    t = text.split()
    return len(set(t))/len(t) if t else 0.0

def jaccard(q_terms, d_terms):
    if not q_terms or not d_terms: return 0.0
    i = len(q_terms & d_terms)
    u = len(q_terms | d_terms)
    return i/u if u else 0.0

def bigram_overlap(q_tok, d_tok):
    qb = set(zip(q_tok, q_tok[1:]))
    db = set(zip(d_tok, d_tok[1:]))
    return len(qb & db)

def first_occ(q_terms, d_tokens):
    for i, t in enumerate(d_tokens):
        if t in q_terms: return i / max(len(d_tokens), 1)
    return 1.0

def idf_sum_fn(q_tokens, vocab, idf):
    return sum(idf[vocab[t]] for t in q_tokens if t in vocab)

def avg_idf_overlap(q_terms, p_terms, vocab, idf):
    shared = q_terms & p_terms
    if not shared: return 0.0
    return sum(idf[vocab[t]] for t in shared if t in vocab) / len(shared)

def tfidf_sum_overlap(q_terms, p_terms, p_tokens, vocab, idf):
    shared = q_terms & p_terms
    if not shared: return 0.0
    d_len = len(p_tokens) + 1e-9
    total = 0.0
    for t in shared:
        if t not in vocab: continue
        tf = p_tokens.count(t) / d_len
        total += (1 + math.log(tf)) * idf[vocab[t]] if tf > 0 else 0
    return total


class SearchRankingSystem:
    def __init__(self):
        print("Loading model...")
        with open(MODEL_FILE, 'rb') as f:
            self.model = pickle.load(f)
        with open(META_FILE) as f:
            self.feature_cols = json.load(f)['feature_cols']

        print("Loading passage corpus...")
        df = pd.read_csv(CORPUS_FILE, dtype={'qid':int,'pid':int})
        self.passages = (df[['pid','passage_clean']]
                         .drop_duplicates('pid')
                         .reset_index(drop=True))
        self.texts = self.passages['passage_clean'].tolist()
        print(f"  Corpus: {len(self.passages):,} passages")

        # Build full-corpus TF-IDF — MUST match step3 exactly
        print("Building TF-IDF index (same params as step3)...")
        self.tfidf = TfidfVectorizer(max_features=60_000, sublinear_tf=True,
                                      min_df=2, ngram_range=(1,1))
        self.doc_matrix = self.tfidf.fit_transform(self.texts)
        self.vocab = self.tfidf.vocabulary_
        self.idf   = self.tfidf.idf_

        print("Building BM25 index...")
        self.bm25 = BM25Okapi([t.split() for t in self.texts])
        print("System ready!\n")

    def rank(self, query_text, top_k=TOP_K, n_cands=N_CANDS):
        query    = clean_text(query_text)
        q_tok    = query.split()
        q_terms  = set(q_tok)

        # Stage 1: BM25 retrieval
        all_bm25 = self.bm25.get_scores(q_tok)
        top_idx  = np.argpartition(all_bm25, -n_cands)[-n_cands:]
        top_idx  = top_idx[np.argsort(all_bm25[top_idx])[::-1]]

        cands = self.passages.iloc[top_idx].copy().reset_index(drop=True)
        cands['bm25_score'] = all_bm25[top_idx]

        # Stage 2: Compute all 18 features on candidates
        # TF-IDF cosine — use full-corpus matrix (same as training)
        q_vec = self.tfidf.transform([query])
        cand_mat = self.doc_matrix[top_idx]
        cands['tfidf_cosine'] = cosine_similarity(q_vec, cand_mat).flatten()

        # Precompute tokens for passage
        cands['p_tok']   = cands['passage_clean'].apply(str.split)
        cands['p_terms'] = cands['p_tok'].apply(set)

        # Scalar query features
        q_len              = len(q_tok)
        q_avg_tf           = avg_tf(query)
        idf_sum_val        = idf_sum_fn(q_tok, self.vocab, self.idf)
        ctr_val            = 0.0  # unknown at inference

        cands['query_len']            = q_len
        cands['query_avg_tf']         = q_avg_tf
        cands['idf_sum']              = idf_sum_val
        cands['ctr']                  = ctr_val

        # Per-passage features
        cands['doc_len']              = cands['p_tok'].apply(len)
        cands['passage_uniq_ratio']   = cands['passage_clean'].apply(passage_uniq_ratio)
        cands['query_doc_len_ratio']  = q_len / (cands['doc_len'] + 1)
        cands['term_overlap']         = cands['p_terms'].apply(lambda pt: len(q_terms & pt))
        cands['keyword_density']      = cands['term_overlap'] / (cands['doc_len'] + 1e-9)
        cands['exact_match']          = cands['passage_clean'].apply(lambda p: int(query in p))
        cands['jaccard_similarity']   = cands['p_terms'].apply(lambda pt: jaccard(q_terms, pt))
        cands['term_overlap_bigram']  = cands['p_tok'].apply(lambda pt: bigram_overlap(q_tok, pt))
        cands['first_occurrence_rank'] = cands['p_tok'].apply(lambda pt: first_occ(q_terms, pt))
        cands['dwell_time_proxy']     = np.log1p(cands['doc_len'])
        cands['avg_idf_of_overlap_terms'] = cands['p_terms'].apply(
            lambda pt: avg_idf_overlap(q_terms, pt, self.vocab, self.idf))
        cands['tf_idf_sum_overlap']   = cands.apply(
            lambda r: tfidf_sum_overlap(q_terms, r['p_terms'],
                                        r['p_tok'], self.vocab, self.idf), axis=1)

        # Re-rank
        X = cands[self.feature_cols].values
        cands['ltr_score'] = self.model.predict(X)

        return (cands.nlargest(top_k, 'ltr_score')
                [['pid','passage_clean','bm25_score','tfidf_cosine','ltr_score']]
                .reset_index(drop=True))


def main():
    print("="*60)
    print("STEP 6 — Two-Stage Search Ranking Demo")
    print("  Stage 1: BM25 retrieves top-100 candidates")
    print("  Stage 2: LambdaMART re-ranks with 18 features")
    print("="*60+"\n")

    sys = SearchRankingSystem()

    demo_queries = [
        "treatment for diabetes",
        "how does photosynthesis work",
        "history of the roman empire",
        "what is machine learning",
        "how to improve memory",
    ]

    print("="*60)
    print("Demo: 5 sample queries")
    print("="*60)

    for q in demo_queries:
        print(f"\nQuery: \"{q}\"")
        print("-"*56)
        results = sys.rank(q, top_k=5)
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            snippet = row['passage_clean'][:140].replace('\n',' ')
            print(f"  #{rank} [LTR={row['ltr_score']:+.4f} | "
                  f"BM25={row['bm25_score']:.2f} | cos={row['tfidf_cosine']:.3f}]")
            print(f"     {snippet}...")

    print("\n"+"="*60)
    print("Interactive Mode (type 'exit' to quit)")
    print("="*60)

    while True:
        try:
            q = input("\nEnter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ('exit','quit','q',''): break
        results = sys.rank(q, top_k=10)
        print(f"\nTop 10 for: \"{q}\"")
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            snippet = row['passage_clean'][:160].replace('\n',' ')
            print(f"\n  #{rank} [LTR={row['ltr_score']:+.4f} | BM25={row['bm25_score']:.2f}]")
            print(f"     {snippet}...")

    print("\nDone!")

if __name__ == "__main__":
    main()
