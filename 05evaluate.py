"""
STEP 5 — Evaluation & Hyperparameter Tuning  (Adiba's Role)
Reads ranked_results.csv + features_dataset.csv
Outputs evaluation_report.csv, best_model.pkl, hyperparam_search_log.csv

Run: python step5_evaluate.py
"""

import pickle, random, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score as sklearn_ndcg

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

RANKED_FILE   = "ranked_results.csv"
FEATURES_FILE = "features_dataset.csv"
REPORT_FILE   = "evaluation_report.csv"
BEST_MODEL    = "best_model.pkl"

FEATURE_COLS = [
    'query_len','doc_len','query_avg_tf','passage_uniq_ratio',
    'query_doc_len_ratio','term_overlap','keyword_density',
    'exact_match','jaccard_similarity','term_overlap_bigram',
    'first_occurrence_rank',
    'tfidf_cosine','bm25_score','idf_sum',
    'avg_idf_of_overlap_terms','tf_idf_sum_overlap',
    'dwell_time_proxy','ctr',
]


# ── Metric helpers ──────────────────────────────────────────────────────────

def ndcg_at_k(rel, scores, k=10):
    if rel.max() == 0: return 0.0
    return float(sklearn_ndcg(rel.reshape(1,-1), scores.reshape(1,-1), k=k))

def average_precision(sorted_rel):
    ap, hits = 0.0, 0
    for i, r in enumerate(sorted_rel):
        if r > 0:
            hits += 1
            ap += hits / (i + 1)
    return ap / max(hits, 1)

def reciprocal_rank(sorted_rel):
    for i, r in enumerate(sorted_rel):
        if r > 0: return 1.0 / (i + 1)
    return 0.0

def precision_at_k(sorted_rel, k=10):
    return sum(1 for r in sorted_rel[:k] if r > 0) / k

def evaluate(df, k=10):
    ndcg_l, map_l, mrr_l, pk_l = [], [], [], []
    for _, grp in df.groupby('qid'):
        rel  = grp['relevance'].values
        pred = grp['pred_score'].values
        ndcg_l.append(ndcg_at_k(rel, pred, k))
        srel = grp.sort_values('pred_score', ascending=False)['relevance'].values
        map_l.append(average_precision(srel))
        mrr_l.append(reciprocal_rank(srel))
        pk_l.append(precision_at_k(srel, k))
    return {
        f'NDCG@{k}': round(float(np.mean(ndcg_l)), 4),
        'MAP':        round(float(np.mean(map_l)),  4),
        'MRR':        round(float(np.mean(mrr_l)),  4),
        f'P@{k}':     round(float(np.mean(pk_l)),   4),
        'n_queries':  len(ndcg_l),
    }

def group_sizes(arr):
    sizes, last, cur = [], None, 0
    for v in arr:
        if v != last:
            if last is not None: sizes.append(cur)
            cur, last = 1, v
        else: cur += 1
    if last is not None: sizes.append(cur)
    return sizes

def ndcg_from_arrays(y_true, y_pred, g_sizes, k=10):
    scores, start = [], 0
    for sz in g_sizes:
        end = start + sz
        scores.append(ndcg_at_k(y_true[start:end], y_pred[start:end], k))
        start = end
    return float(np.mean(scores))


def main():
    print("="*60)
    print("STEP 5 — Evaluation & Hyperparameter Tuning")
    print("="*60)

    # ── Part A: Evaluate base model ─────────────────────────────
    print("\n[PART A] Base model metrics...")
    ranked = pd.read_csv(RANKED_FILE,
                         dtype={'qid':int,'pid':int,'relevance':int})
    m_base = evaluate(ranked, k=10)
    print(f"  NDCG@10 : {m_base['NDCG@10']}")
    print(f"  MAP     : {m_base['MAP']}")
    print(f"  MRR     : {m_base['MRR']}")
    print(f"  P@10    : {m_base['P@10']}")
    print(f"  Queries : {m_base['n_queries']}")

    # ── Part B: Hyperparameter search ───────────────────────────
    print("\n[PART B] Hyperparameter random search (10 iterations)...")
    df = pd.read_csv(FEATURES_FILE,
                     dtype={'qid':int,'pid':int,'relevance':int})
    X = df[FEATURE_COLS].values
    y = df['relevance'].values
    g = df['qid'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, g))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr = group_sizes(g[tr_idx])
    g_te = group_sizes(g[te_idx])

    param_space = {
        'max_depth':        [4, 5, 6],
        'eta':              [0.05, 0.08, 0.1],
        'n_estimators':     [150, 200, 250],   # kept small so each iter is fast
        'subsample':        [0.7, 0.8, 0.9],
        'gamma':            [0.0, 0.5, 1.0],
        'min_child_weight': [0.1, 0.5, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

    best_ndcg, best_model, results = 0.0, None, []

    for i in range(10):
        params = {k: random.choice(v) for k, v in param_space.items()}
        m = xgb.XGBRanker(
            objective='rank:ndcg', eval_metric='ndcg@10',
            tree_method='hist', verbosity=0, random_state=42, **params
        )
        m.fit(
            X_tr, y_tr, group=g_tr,
            eval_set=[(X_te, y_te)], eval_group=[g_te],
            verbose=False
        )
        nd = ndcg_from_arrays(y_te, m.predict(X_te), g_te)
        star = "  ← BEST" if nd > best_ndcg else ""
        print(f"  Iter {i+1:2d}: NDCG@10={nd:.4f}  "
              f"depth={params['max_depth']} lr={params['eta']} "
              f"n_trees={params['n_estimators']}{star}")
        results.append({'iter': i+1, **params, 'NDCG@10': round(nd, 4)})
        if nd > best_ndcg:
            best_ndcg, best_model = nd, m

    # Full metrics on best model
    df_te = df.iloc[te_idx].copy()
    df_te['pred_score'] = best_model.predict(X_te)
    m_best = evaluate(df_te, k=10)

    print(f"\n  Best Model after tuning:")
    print(f"  NDCG@10 : {m_best['NDCG@10']}")
    print(f"  MAP     : {m_best['MAP']}")
    print(f"  MRR     : {m_best['MRR']}")
    print(f"  P@10    : {m_best['P@10']}")

    # Save outputs
    report = {
        'base_NDCG@10': m_base['NDCG@10'], 'best_NDCG@10': m_best['NDCG@10'],
        'base_MAP':      m_base['MAP'],     'best_MAP':      m_best['MAP'],
        'base_MRR':      m_base['MRR'],     'best_MRR':      m_best['MRR'],
        'base_P@10':     m_base['P@10'],    'best_P@10':     m_best['P@10'],
    }
    pd.DataFrame([report]).to_csv(REPORT_FILE, index=False)
    pd.DataFrame(results).to_csv('hyperparam_search_log.csv', index=False)
    with open(BEST_MODEL, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\nDONE")
    print(f"  {REPORT_FILE}  |  {BEST_MODEL}  |  hyperparam_search_log.csv")
    print("Next: python step6_demo.py")


if __name__ == "__main__":
    main()
