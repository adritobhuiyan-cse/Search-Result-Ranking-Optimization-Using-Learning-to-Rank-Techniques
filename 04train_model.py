"""
STEP 4 — LambdaMART Training  (Adrito's Role)
Reads features_dataset.csv → trained_ranking_model.pkl

Run: python step4_train_model.py
"""

import json, pickle, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

IN_FILE     = "features_dataset.csv"
MODEL_FILE  = "trained_ranking_model.pkl"
META_FILE   = "model_metadata.json"
RANKED_FILE = "ranked_results.csv"

FEATURE_COLS = [
    'query_len','doc_len','query_avg_tf','passage_uniq_ratio',
    'query_doc_len_ratio','term_overlap','keyword_density',
    'exact_match','jaccard_similarity','term_overlap_bigram',
    'first_occurrence_rank',
    'tfidf_cosine','bm25_score','idf_sum',
    'avg_idf_of_overlap_terms','tf_idf_sum_overlap',
    'dwell_time_proxy','ctr',
]


def group_sizes(groups_arr):
    sizes, last, cur = [], None, 0
    for g in groups_arr:
        if g != last:
            if last is not None: sizes.append(cur)
            cur, last = 1, g
        else:
            cur += 1
    if last is not None: sizes.append(cur)
    return sizes


def main():
    print("="*60)
    print("STEP 4 — LambdaMART Training")
    print("="*60)

    print("\n[1/5] Loading features...")
    df = pd.read_csv(IN_FILE, dtype={'qid':int,'pid':int,'relevance':int})
    print(f"      Rows: {len(df):,} | Queries: {df['qid'].nunique():,}")
    print(f"      Relevant: {(df['relevance']>0).sum():,}")

    if (df['relevance']>0).sum() == 0:
        print("ERROR: Zero relevant rows. Re-run step2 and step3.")
        return

    X      = df[FEATURE_COLS].values
    y      = df['relevance'].values
    groups = df['qid'].values

    print("[2/5] Train/test split by query group...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr = group_sizes(groups[tr_idx])
    g_te = group_sizes(groups[te_idx])

    print(f"      Train: {X_tr.shape[0]:,} rows / {len(g_tr):,} queries  "
          f"({y_tr.sum():,} relevant)")
    print(f"      Test:  {X_te.shape[0]:,} rows / {len(g_te):,} queries  "
          f"({y_te.sum():,} relevant)")

    print("\n[3/5] Training LambdaMART (watch NDCG@10 rise)...")
    model = xgb.XGBRanker(
        tree_method      = 'hist',
        objective        = 'rank:ndcg',
        eval_metric      = 'ndcg@10',
        eta              = 0.05,
        max_depth        = 6,
        min_child_weight = 0.1,
        gamma            = 1.0,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        n_estimators     = 400,
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(
        X_tr, y_tr, group=g_tr,
        eval_set=[(X_te, y_te)], eval_group=[g_te],
        verbose=50,
    )

    print("\n[4/5] Ranking test documents...")
    df_te = df.iloc[te_idx].copy()
    df_te['pred_score'] = model.predict(X_te)
    df_te['rank'] = (df_te.groupby('qid')['pred_score']
                         .rank(ascending=False, method='first').astype(int))
    df_te.sort_values(['qid','rank']).to_csv(RANKED_FILE, index=False)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    print("\n  Top-10 Feature Importances:")
    print(imp.sort_values(ascending=False).head(10).to_string())

    print("\n[5/5] Saving model...")
    with open(MODEL_FILE,'wb') as f: pickle.dump(model, f)
    with open(META_FILE,'w') as f:
        json.dump({'feature_cols': FEATURE_COLS}, f, indent=2)

    print(f"\nDONE")
    print(f"  {MODEL_FILE}  |  {META_FILE}  |  {RANKED_FILE}")
    print("Next: python step5_evaluate.py")


if __name__ == "__main__":
    main()
