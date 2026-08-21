"""
Streamlit Frontend for Search Ranking System
Run: streamlit run app.py
"""

import json, pickle, re, math, warnings
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

warnings.filterwarnings("ignore")

MODEL_FILE  = "best_model.pkl"
META_FILE   = "model_metadata.json"
CORPUS_FILE = "clean_search_dataset.csv"
N_CANDS     = 100
TOP_K       = 10

# ── Text / feature helpers (identical to step6) ─────────────────────────────

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def avg_tf(text):
    t = text.split()
    return sum(Counter(t).values()) / len(t) if t else 0.0

def passage_uniq_ratio(text):
    t = text.split()
    return len(set(t)) / len(t) if t else 0.0

def jaccard(q_terms, d_terms):
    if not q_terms or not d_terms: return 0.0
    return len(q_terms & d_terms) / len(q_terms | d_terms)

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


# ── Cached model + corpus loader ────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_system():
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    with open(META_FILE) as f:
        feature_cols = json.load(f)['feature_cols']

    df = pd.read_csv(CORPUS_FILE, dtype={'qid': int, 'pid': int})
    passages = (df[['pid', 'passage_clean']]
                .drop_duplicates('pid')
                .reset_index(drop=True))
    texts = passages['passage_clean'].tolist()

    tfidf = TfidfVectorizer(max_features=60_000, sublinear_tf=True,
                             min_df=2, ngram_range=(1, 1))
    doc_matrix = tfidf.fit_transform(texts)
    vocab = tfidf.vocabulary_
    idf   = tfidf.idf_

    bm25 = BM25Okapi([t.split() for t in texts])

    return model, feature_cols, passages, texts, tfidf, doc_matrix, vocab, idf, bm25


def rank_query(query_text, model, feature_cols, passages, texts,
               tfidf, doc_matrix, vocab, idf, bm25,
               n_cands=N_CANDS, top_k=TOP_K):

    query   = clean_text(query_text)
    q_tok   = query.split()
    q_terms = set(q_tok)

    # Stage 1 — BM25 retrieval
    all_bm25 = bm25.get_scores(q_tok)
    top_idx  = np.argpartition(all_bm25, -n_cands)[-n_cands:]
    top_idx  = top_idx[np.argsort(all_bm25[top_idx])[::-1]]

    cands = passages.iloc[top_idx].copy().reset_index(drop=True)
    cands['bm25_score'] = all_bm25[top_idx]

    # Stage 2 — features
    q_vec = tfidf.transform([query])
    cands['tfidf_cosine'] = cosine_similarity(q_vec, doc_matrix[top_idx]).flatten()

    cands['p_tok']   = cands['passage_clean'].apply(str.split)
    cands['p_terms'] = cands['p_tok'].apply(set)

    q_len         = len(q_tok)
    idf_sum_val   = idf_sum_fn(q_tok, vocab, idf)

    cands['query_len']             = q_len
    cands['query_avg_tf']          = avg_tf(query)
    cands['idf_sum']               = idf_sum_val
    cands['ctr']                   = 0.0
    cands['doc_len']               = cands['p_tok'].apply(len)
    cands['passage_uniq_ratio']    = cands['passage_clean'].apply(passage_uniq_ratio)
    cands['query_doc_len_ratio']   = q_len / (cands['doc_len'] + 1)
    cands['term_overlap']          = cands['p_terms'].apply(lambda pt: len(q_terms & pt))
    cands['keyword_density']       = cands['term_overlap'] / (cands['doc_len'] + 1e-9)
    cands['exact_match']           = cands['passage_clean'].apply(lambda p: int(query in p))
    cands['jaccard_similarity']    = cands['p_terms'].apply(lambda pt: jaccard(q_terms, pt))
    cands['term_overlap_bigram']   = cands['p_tok'].apply(lambda pt: bigram_overlap(q_tok, pt))
    cands['first_occurrence_rank'] = cands['p_tok'].apply(lambda pt: first_occ(q_terms, pt))
    cands['dwell_time_proxy']      = np.log1p(cands['doc_len'])
    cands['avg_idf_of_overlap_terms'] = cands['p_terms'].apply(
        lambda pt: avg_idf_overlap(q_terms, pt, vocab, idf))
    cands['tf_idf_sum_overlap']    = cands.apply(
        lambda r: tfidf_sum_overlap(q_terms, r['p_terms'], r['p_tok'], vocab, idf), axis=1)

    X = cands[feature_cols].values
    cands['ltr_score'] = model.predict(X)

    return (cands.nlargest(top_k, 'ltr_score')
            [['pid', 'passage_clean', 'bm25_score', 'tfidf_cosine', 'ltr_score']]
            .reset_index(drop=True))


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Search Ranking Demo",
    page_icon="🔍",
    layout="wide",
)

# Header
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem 0'>
    <h1 style='font-size:2.4rem; margin-bottom:0'>🔍 Search Engine Ranking Demo</h1>
    <p style='color:#888; font-size:1rem; margin-top:0.3rem'>
        LambdaMART · BM25 · TF-IDF · MS MARCO · CSE445
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load system (cached — only runs once)
with st.spinner("Loading model and building indexes... (first run takes ~1 min)"):
    system = load_system()
    model, feature_cols, passages, texts, tfidf, doc_matrix, vocab, idf, bm25 = system

# Sidebar — settings
with st.sidebar:
    st.header("⚙️ Settings")
    top_k     = st.slider("Results to show", min_value=3, max_value=20, value=10)
    n_cands   = st.slider("BM25 candidates (Stage 1)", min_value=50, max_value=300, value=100, step=50)
    show_feat = st.toggle("Show feature scores", value=True)
    st.divider()
    st.markdown(f"**Corpus size:** {len(passages):,} passages")
    st.markdown(f"**Features:** {len(feature_cols)}")
    st.markdown("**Pipeline:**")
    st.markdown("1. BM25 → top-N candidates")
    st.markdown("2. LambdaMART re-ranks")
    st.divider()
    st.markdown("**Demo queries:**")
    demo_queries = [
        "treatment for diabetes",
        "how does photosynthesis work",
        "history of the roman empire",
        "what is machine learning",
        "how to improve memory",
        "causes of climate change",
        "symptoms of depression",
    ]
    for dq in demo_queries:
        if st.button(dq, use_container_width=True):
            st.session_state['query'] = dq

# Main search bar
query_input = st.text_input(
    "Enter your search query",
    value=st.session_state.get('query', ''),
    placeholder="e.g. treatment for diabetes",
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 4])
search_clicked = col1.button("🔍 Search", type="primary", use_container_width=True)
clear_clicked  = col2.button("✕ Clear",  use_container_width=True)

if clear_clicked:
    st.session_state['query'] = ''
    st.rerun()

query = query_input.strip()

if (search_clicked or query) and query:
    with st.spinner(f'Searching for "{query}"...'):
        results = rank_query(
            query, model, feature_cols, passages, texts,
            tfidf, doc_matrix, vocab, idf, bm25,
            n_cands=n_cands, top_k=top_k
        )

    st.markdown(f"**{len(results)} results** for: *{query}*")
    st.divider()

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        passage_text = row['passage_clean']

        # Highlight query terms in passage
        highlighted = passage_text
        for term in clean_text(query).split():
            if len(term) > 2:
                highlighted = re.sub(
                    f'\\b({re.escape(term)})\\b',
                    r'**\1**',
                    highlighted,
                    flags=re.IGNORECASE
                )

        # Score badge colour
        score = row['ltr_score']
        if score > 1.0:
            badge_colour = "#2e7d32"   # dark green
        elif score > 0:
            badge_colour = "#1565c0"   # blue
        else:
            badge_colour = "#c62828"   # red

        with st.container():
            header_cols = st.columns([0.05, 0.65, 0.3])
            header_cols[0].markdown(
                f"<div style='font-size:1.4rem; font-weight:700; "
                f"color:#555; padding-top:4px'>#{rank}</div>",
                unsafe_allow_html=True
            )
            header_cols[1].markdown(
                f"<span style='background:{badge_colour}; color:white; "
                f"padding:3px 10px; border-radius:12px; font-size:0.85rem; "
                f"font-weight:600'>LTR Score: {score:+.4f}</span>",
                unsafe_allow_html=True
            )
            if show_feat:
                header_cols[2].markdown(
                    f"<span style='color:#888; font-size:0.8rem'>"
                    f"BM25: {row['bm25_score']:.2f} &nbsp;|&nbsp; "
                    f"cos: {row['tfidf_cosine']:.3f}</span>",
                    unsafe_allow_html=True
                )

            st.markdown(
                f"<div style='background:#f8f9fa; border-left:3px solid {badge_colour}; "
                f"padding:10px 14px; border-radius:4px; margin:4px 0 12px 0; "
                f"font-size:0.93rem; line-height:1.6'>{highlighted}</div>",
                unsafe_allow_html=True
            )

elif not query:
    # Welcome state
    st.markdown("""
    <div style='text-align:center; padding:3rem 0; color:#aaa'>
        <div style='font-size:3rem'>🔍</div>
        <div style='font-size:1.1rem; margin-top:0.5rem'>
            Type a query above or click a demo query in the sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works cards
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Stage 1 — BM25 Retrieval**\nFast keyword matching retrieves top-N candidate passages from the full corpus")
    with c2:
        st.info("**Stage 2 — Feature Engineering**\n18 features computed: TF-IDF cosine, BM25, term overlap, Jaccard, IDF sum, and more")
    with c3:
        st.info("**Stage 3 — LambdaMART Re-ranking**\nXGBoost ranker trained with rank:ndcg objective re-orders candidates by relevance")
