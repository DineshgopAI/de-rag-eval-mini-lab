import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

STOPWORDS = {"the","a","an","and","or","to","of","in","on","is","are","was","were","für","der","die","das","und","oder","zu","von","im","ist"}
PROFANITY = {"shit","fuck","bitch","scheiße","arsch","bloody"}  # expand if you like

def tokenize(s: str):
    return [t for t in s.lower().split() if t.strip()]

def is_trash_query(q: str) -> str | None:
    toks = tokenize(q)
    if len(q.strip()) < 2 or len(toks) == 0:
        return "Empty/too short query."
    if len(toks) < 2:
        return "Query too short (need ≥2 tokens)."
    if all(t in STOPWORDS for t in toks):
        return "Only stopwords; try more specific words."
    if any(t in PROFANITY for t in toks):
        return "Profanity detected; no retrieval performed."
    return None


DATA_CSV = Path("data") / "de_en_qa.csv"
BM25_PICKLE = Path("reports") / "bm25_index.pkl"
EMB_PICKLE = Path("reports") / "dense_index.pkl"

def tokenize(s: str):
    return s.lower().split()

@st.cache_resource
def load_data():
    df = pd.read_csv(DATA_CSV, encoding="utf-8")
    contexts = df["context"].tolist()
    return df, contexts

@st.cache_resource
def load_bm25():
    with open(BM25_PICKLE, "rb") as f:
        pkg = pickle.load(f)
    return pkg["bm25"], [tokenize(c) for c in pkg["contexts"]]

@st.cache_resource
def load_dense():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None, None, None
    if not EMB_PICKLE.exists():
        return None, None, None
    with open(EMB_PICKLE, "rb") as f:
        pkg = pickle.load(f)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model, pkg["nn"], pkg["emb"]

def bm25_search(bm25, tokenized_docs, query, k):
    q_tok = tokenize(query)
    scores = bm25.get_scores(q_tok)
    # track token overlap
    overlaps = [len(set(q_tok) & set(doc)) for doc in tokenized_docs]
    idx = list(np.argsort(scores)[::-1][:k])
    return idx, [float(scores[i]) for i in idx], [int(overlaps[i]) for i in idx]


def dense_search(model, nn, emb, query, k):
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    dist, idx = nn.kneighbors(qv.reshape(1, -1), n_neighbors=k, return_distance=True)
    idx = list(idx[0]); dist = list(dist[0])
    # convert dist→sim for readability: sim = 1 - dist (because vectors normalized)
    sims = [float(1.0 - d) for d in dist]
    return idx, dist, sims

def analyze_bm25(scores: list[float], overlaps: list[int]) -> bool:
    # at least one overlapping token & a score not trivially low
    has_overlap = any(o > 0 for o in overlaps)
    if not has_overlap:
        return False
    # heuristic: candidate[0] must be reasonably above noise:
    top = scores[0]
    med = float(np.median(scores)) if scores else 0.0
    return top >= max(0.0, 0.2 * (med if med != 0 else top))

def analyze_dense(sims: list[float], min_sim: float = 0.30) -> bool:
    # require at least one hit above similarity threshold
    return max(sims) >= min_sim if sims else False

def main():
    st.title("DE/EN RAG Eval Mini-Lab — demo")
    df, contexts = load_data()
    n_docs = len(contexts)

    mode = st.radio("Retriever", ["BM25", "MiniLM (dense)"])
    k = st.slider("Top-k", 1, min(10, n_docs), 3)
    query = st.text_input("Query", value=df["question"].iloc[0])

    if st.button("Search"):
        # guardrail 1: trash queries
        guard_msg = is_trash_query(query)
        if guard_msg:
            st.warning(guard_msg)
            return

        if mode == "BM25":
            bm25, tok = load_bm25()
            t0 = time.perf_counter()
            idx, scores, overlaps = bm25_search(bm25, tok, query, k)
            dt = (time.perf_counter() - t0) * 1000
            ok = analyze_bm25(scores, overlaps)
            st.write(f"Latency: {dt:.2f} ms")
            
            if not ok:
                st.info("No confidently relevant results (BM25 abstained). Trying dense as fallback…")
                model, nn, emb = load_dense()
                if model is None:
                    st.warning("Dense index not available. No result.")
                    return
                t0 = time.perf_counter()
                i2, dist2, sims2 = dense_search(model, nn, emb, query, k)
                dt2 = (time.perf_counter() - t0) * 1000
                if not analyze_dense(sims2):
                    st.error("No confidently relevant results (both retrievers abstained).")
                    return
                st.write(f"Fallback Dense Latency: {dt2:.2f} ms")
                for rank, (j, d, s) in enumerate(zip(i2, dist2, sims2), 1):
                    st.markdown(f"**{rank}.** sim={s:.3f} dist={d:.3f}\n\n{contexts[j]}")
                return
            # BM25 accepted
            for rank, (j, sc, ov) in enumerate(zip(idx, scores, overlaps), 1):
                st.markdown(f"**{rank}.** score={sc:.3f} overlap={ov}\n\n{contexts[j]}")
        else:
            model, nn, emb = load_dense()
            if model is None:
                st.warning("Dense index not available. Build it first.")
                return
            t0 = time.perf_counter()
            idx, dist, sims = dense_search(model, nn, emb, query, k)
            dt = (time.perf_counter() - t0) * 1000
            st.write(f"Latency: {dt:.2f} ms (lower dist better; sim=1−dist)")
            ok = analyze_dense(sims)
            if not ok:
                st.info("No confidently relevant results (dense abstained). Trying BM25 as fallback…")
                bm25, tok = load_bm25()
                t0 = time.perf_counter()
                i2, scores2, overlaps2 = bm25_search(bm25, tok, query, k)
                dt2 = (time.perf_counter() - t0) * 1000
                if not analyze_bm25(scores2, overlaps2):
                    st.error("No confidently relevant results (both retrievers abstained).")
                    return
                st.write(f"Fallback BM25 Latency: {dt2:.2f} ms")
                for rank, (j, sc, ov) in enumerate(zip(i2, scores2, overlaps2), 1):
                    st.markdown(f"**{rank}.** score={sc:.3f} overlap={ov}\n\n{contexts[j]}")
                return
            # Dense accepted
            for rank, (j, d, s) in enumerate(zip(idx, dist, sims), 1):
                st.markdown(f"**{rank}.** sim={s:.3f} dist={d:.3f}\n\n{contexts[j]}")

if __name__ == "__main__":
    main()
