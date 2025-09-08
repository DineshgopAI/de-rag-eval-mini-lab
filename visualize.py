import json
from pathlib import Path
import matplotlib.pyplot as plt

METRICS_JSON = Path("reports") / "metrics.json"
PLOTS_DIR = Path("reports") / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    data = json.loads(METRICS_JSON.read_text())
    bm25 = data.get("bm25", {})
    dense = data.get("dense_minilm", {})

    # recall@k
    ks = [int(k) for k in bm25.get("recall", {}).keys()]
    ks.sort()
    bm25_rec = [bm25["recall"][str(k)] for k in ks]
    if "recall" in dense and isinstance(dense["recall"], dict):
        dense_rec = [dense["recall"].get(str(k), 0.0) for k in ks]
    else:
        dense_rec = []

    plt.figure()
    plt.plot(ks, bm25_rec, marker="o", label="BM25")
    if dense_rec:
        plt.plot(ks, dense_rec, marker="o", label="MiniLM")
    plt.xlabel("k"); plt.ylabel("Recall@k"); plt.title("Recall vs k")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "recall_at_k.png", bbox_inches="tight")
    plt.close()

    # MRR
    plt.figure()
    mrr_vals = [bm25.get("mrr_at_10", 0.0)]
    labels = ["BM25"]
    if "mrr_at_10" in dense and isinstance(dense.get("mrr_at_10"), (int, float)):
        mrr_vals.append(dense["mrr_at_10"]); labels.append("MiniLM")
    plt.bar(labels, mrr_vals)
    plt.title("MRR@10"); plt.ylabel("MRR")
    plt.savefig(PLOTS_DIR / "mrr.png", bbox_inches="tight")
    plt.close()

    # latency
    plt.figure()
    lat_vals = [bm25.get("avg_latency_ms", 0.0)]
    labels = ["BM25"]
    if "avg_latency_ms" in dense and isinstance(dense.get("avg_latency_ms"), (int, float)):
        lat_vals.append(dense["avg_latency_ms"]); labels.append("MiniLM")
    plt.bar(labels, lat_vals)
    plt.title("Avg Query Latency (ms)")
    plt.ylabel("ms")
    plt.savefig(PLOTS_DIR / "latency.png", bbox_inches="tight")
    plt.close()

    print("saved plots →", PLOTS_DIR.resolve())

if __name__ == "__main__":
    main()
