import json, time, pickle, math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

DATA_CSV = Path("data") / "de_en_qa.csv"
ARTIFACTS = Path("reports")
BM25_PICKLE = ARTIFACTS / "bm25_index.pkl"
EMB_PICKLE = ARTIFACTS / "dense_index.pkl"
METRICS_JSON = ARTIFACTS / "metrics.json"

K_LIST = [1, 3, 5]  # fits tiny datasets; we’ll auto-cap top_k below


def recall_at_k(gt_idx: int, ranked_idx: List[int], k: int) -> float:
    return 1.0 if gt_idx in ranked_idx[:k] else 0.0

def mrr_at_k(gt_idx: int, ranked_idx: List[int], k: int) -> float:
    for rank, idx in enumerate(ranked_idx[:k], start=1):
        if idx == gt_idx:
            return 1.0 / rank
    return 0.0

def rank_bm25_query(bm25, tokenized_docs, query_tokens, top_k=10) -> List[int]:
    scores = bm25.get_scores(query_tokens)
    # argsort descending
    return list(np.argsort(scores)[::-1][:top_k])

def tokenize(s: str) -> List[str]:
    return s.lower().split()

def rank_dense_query(nn, emb_matrix: np.ndarray, query_vec: np.ndarray, top_k=10) -> List[int]:
    # sklearn NearestNeighbors returns distances; we want ascending distance
    dist, idx = nn.kneighbors(query_vec.reshape(1, -1), n_neighbors=top_k, return_distance=True)
    return list(idx[0])

def build_dense_model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def main():
    if not DATA_CSV.exists() or not BM25_PICKLE.exists():
        raise FileNotFoundError("Missing data or BM25 index. Run build_indexes.py first.")

    df = pd.read_csv(DATA_CSV, encoding="utf-8")
    questions = df["question"].tolist()
    contexts  = df["context"].tolist()
    top_k = min(10, len(contexts))  # cap by available docs


    # load BM25
    with open(BM25_PICKLE, "rb") as f:
        bm25_pkg = pickle.load(f)
    bm25 = bm25_pkg["bm25"]
    tokenized_docs = [tokenize(c) for c in contexts]

    # try to load dense
    dense_ok, nn, emb, dense_model = False, None, None, None
    if EMB_PICKLE.exists():
        with open(EMB_PICKLE, "rb") as f:
            dense_pkg = pickle.load(f)
        nn, emb = dense_pkg["nn"], dense_pkg["emb"]
        dense_model = build_dense_model()
        dense_ok = dense_model is not None

    results: Dict[str, Dict] = {}

    # ---- BM25 eval ----
    bm25_metrics = {"recall": {str(k): [] for k in K_LIST}, "mrr_at_10": [], "latency_ms": []}
    for i, q in enumerate(questions):
        q_tok = tokenize(q)
        t0 = time.perf_counter()
        ranked = rank_bm25_query(bm25, tokenized_docs, q_tok, top_k=top_k)
        dt = (time.perf_counter() - t0) * 1000.0
        bm25_metrics["latency_ms"].append(dt)
        for k in K_LIST:
            bm25_metrics["recall"][str(k)].append(recall_at_k(i, ranked, k))
        bm25_metrics["mrr_at_10"].append(mrr_at_k(i, ranked, 10))
    results["bm25"] = {
        "recall": {k: float(np.mean(v)) for k, v in bm25_metrics["recall"].items()},
        "mrr_at_10": float(np.mean(bm25_metrics["mrr_at_10"])),
        "avg_latency_ms": float(np.mean(bm25_metrics["latency_ms"])),
        "n_queries": len(questions),
    }

    # ---- Dense eval (if available) ----
    if dense_ok:
        from sentence_transformers import util as st_util
        dense_metrics = {"recall": {str(k): [] for k in K_LIST}, "mrr_at_10": [], "latency_ms": []}
        # encode all queries once
        q_emb = dense_model.encode(questions, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
        for i in range(len(questions)):
            t0 = time.perf_counter()
            ranked = rank_dense_query(nn, emb, q_emb[i], top_k=top_k)
            dt = (time.perf_counter() - t0) * 1000.0
            dense_metrics["latency_ms"].append(dt)
            for k in K_LIST:
                dense_metrics["recall"][str(k)].append(recall_at_k(i, ranked, k))
            dense_metrics["mrr_at_10"].append(mrr_at_k(i, ranked, 10))
        results["dense_minilm"] = {
            "recall": {k: float(np.mean(v)) for k, v in dense_metrics["recall"].items()},
            "mrr_at_10": float(np.mean(dense_metrics["mrr_at_10"])),
            "avg_latency_ms": float(np.mean(dense_metrics["latency_ms"])),
            "n_queries": len(questions),
        }
    else:
        results["dense_minilm"] = {"note": "dense index not available — run build_indexes.py successfully with sentence-transformers installed."}

    # meta
    results["_meta"] = {
        "k_list": K_LIST,
        "n_docs": len(contexts),
        "n_queries": len(questions),
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(results, indent=2))
    print("Saved metrics →", METRICS_JSON.resolve())
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
