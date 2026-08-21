# CSE445: Search Engine Result Ranking Optimization
## Shafi · Tahmid · Adrito · Adiba · Sadia



## PROJECT STRUCTURE

```
search_ranking_project/
├── requirements.txt
├── 01_download_data.py        ← Sadia: Download from Hugging Face (internet needed)
├── step1_offline_fallback.py  ← Shafi: Use this if HuggingFace fails (no internet)
├── 02_clean_data.py           ← Shafi: Clean text
├── 03_feature_engineering.py  ← Tahmid: Build 12 features (TF-IDF, BM25, CTR, etc.)
├── 04_train_model.py          ← Adrito:    Train LambdaMART (XGBoost rank:ndcg)
├── 05_evaluate.py             ← Adiba:  NDCG, MAP, MRR, P@10 + hyperparameter tuning
└── 06_demo.py                 ← Final:  Interactive ranked search demo
```

---

## STEP-BY-STEP EXECUTION (VS Code, Windows, Python 3.12)

### Open VS Code Terminal
Press Ctrl+` (backtick). Navigate to the project folder:
```
cd path\to\search_ranking_project
```

---

### STEP 0 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### STEP 1 — Download MS MARCO dataset

**Option A — With internet (Hugging Face):**
```bash
python step1_download_data.py
```
Downloads ~50 MB from huggingface.co.

**Option B — If Hugging Face is blocked or slow:**
```bash
python step1_offline_fallback.py
```
Generates realistic synthetic data instantly. No internet needed.
The model will still train perfectly and show real NDCG improvement.

Both options produce: `raw_msmarco.csv`

Expected output:
```
relevance=0: 35,000 rows (87.5%)
relevance=1:  5,000 rows (12.5%)  
```

---

### STEP 2 — Clean the data
```bash
python step2_clean_data.py
```
Output: `clean_search_dataset.csv`


---

### STEP 3 — Build feature matrix
```bash
python step3_feature_engineering.py
```
Output: `features_dataset.csv` with 12 features per (query, passage) pair:
- query_len, query_avg_tf
- doc_len, keyword_density
- tfidf_cosine (TF-IDF cosine similarity)
- bm25_score (BM25 Okapi ranking score)
- ctr (click-through rate proxy)
- term_overlap, exact_match
- dwell_time_proxy, idf_sum, passage_uniq_ratio



---

### STEP 4 — Train LambdaMART model
```bash
python step4_train_model.py
```
Outputs: `trained_ranking_model.pkl`, `ranked_results.csv`, `model_metadata.json`

We will see NDCG@10 improving every 50 trees:
```
[0]    validation_0-ndcg@10: 0.93
[50]   validation_0-ndcg@10: 0.97
[100]  validation_0-ndcg@10: 0.99   ← model converges
```


---

### STEP 5 — Evaluate + hyperparameter tuning
```bash
python step5_evaluate.py
```
Outputs: `evaluation_report.csv`, `best_model.pkl`, `hyperparam_search_log.csv`

We will see 15 random search iterations, each with a NDCG@10 score.


---

### STEP 6 — Run the interactive demo
```bash
python step6_demo.py
```
Shows ranked results for 5 demo queries, then enters interactive mode.
Type any query → get top-10 ranked passages instantly.

---

## EXPECTED OUTPUT FILES

| File | Owner | What it contains |
|------|-------|-----------------|
| `raw_msmarco.csv` | Sadia | Raw query-passage pairs with relevance |
| `clean_search_dataset.csv` | Shafi | Cleaned text, confirmed relevance labels |
| `features_dataset.csv` | Tahmid | 12 numerical features per pair |
| `trained_ranking_model.pkl` | Adrito | LambdaMART model |
| `ranked_results.csv` | Adrito | Test set ranked results |
| `evaluation_report.csv` | Adiba | NDCG@10, MAP, MRR, P@10 |
| `best_model.pkl` | Adiba | Best tuned model |

---
