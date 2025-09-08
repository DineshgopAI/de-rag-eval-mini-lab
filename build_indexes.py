import json
import time
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.neighbors import NearestNeighbors

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # we'll handle gracefully


DATA_CSV = Path("data") / "de_en_qa.csv"
ARTIFACTS = Path("reports")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

BM25_PICKLE = ARTIFACTS / "bm25_index.pkl"
EMB_PICKLE = ARTIFACTS / "dense_index.pkl"
EMB_META = ARTIFACTS / "dense_meta.json"


def tokenize(text: str) -> List[str]:
    # super simple tokenization; good enough for tiny demo
    return text.lower().split()


def build_bm25(contexts: List[str]) -> Tuple[BM25Okapi, float]:
    tokenized = [tokenize(c) for c in contexts]
    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    build_s = time.perf_counter() - t0
    return bm25, build_s


def build_dense(contexts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available. Install it first.")
    model = SentenceTransformer(model_name)
    t0 = time.perf_counter()
    emb = model.encode(contexts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    build_s = time.perf_counter() - t0

    # sklearn NN over cosine distance = 1 - dot since we normalized vectors
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(emb)
    meta = {
        "model_name": model_name,
        "embeddings_shape": list(emb.shape),
        "built_seconds": build_s,
    }
    return nn, emb, meta


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing {DATA_CSV}")

    df = pd.read_csv(DATA_CSV, encoding="utf-8")
    contexts = df["context"].tolist()

    print("Building BM25…")
    bm25, bm25_build_s = build_bm25(contexts)
    with open(BM25_PICKLE, "wb") as f:
        pickle.dump({"bm25": bm25, "contexts": contexts}, f)
    print(f"BM25 built in {bm25_build_s:.3f}s → {BM25_PICKLE}")

    # Dense index (optional if installation fails)
    try:
        print("Building MiniLM dense index… (downloads model on first run)")
        nn, emb, meta = build_dense(contexts)
        with open(EMB_PICKLE, "wb") as f:
            pickle.dump({"nn": nn, "emb": emb, "contexts": contexts}, f)
        EMB_META.write_text(json.dumps(meta, indent=2))
        print(f"Dense index built in {meta['built_seconds']:.3f}s → {EMB_PICKLE}")
    except Exception as e:
        print(f"[warn] Dense index not built: {e}")
        print("You can still run BM25-only evaluation.")

    print("Done.")


if __name__ == "__main__":
    main()
