import pandas as pd
from pathlib import Path

CSV_PATH = Path("data") / "de_en_qa.csv"

REQUIRED_COLS = ["id", "question", "context", "answer", "lang"]

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH.resolve()}")

    # read + basic checks
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # enforce dtypes / strip whitespace
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    for col in ["question", "context", "answer", "lang"]:
        df[col] = df[col].astype(str).str.strip()

    # sanity checks
    problems = []
    if df["id"].isna().any():
        problems.append("Some ids are not numeric.")
    if df.duplicated(subset=["question", "context"]).any():
        problems.append("Duplicate question+context rows found.")
    if (df[["question", "context", "answer"]].replace("", pd.NA).isna().any().any()):
        problems.append("Empty strings found in question/context/answer.")

    # language distribution
    lang_counts = df["lang"].value_counts().to_dict()

    # simple length stats
    q_len = df["question"].str.split().str.len().describe().to_dict()
    c_len = df["context"].str.split().str.len().describe().to_dict()
    a_len = df["answer"].str.split().str.len().describe().to_dict()

    print("=== Dataset summary ===")
    print(f"rows: {len(df)}")
    print(f"languages: {lang_counts}")
    print("question length (words):", {k: round(v, 2) for k, v in q_len.items() if isinstance(v, (int, float))})
    print("context length (words):", {k: round(v, 2) for k, v in c_len.items() if isinstance(v, (int, float))})
    print("answer length (words):", {k: round(v, 2) for k, v in a_len.items() if isinstance(v, (int, float))})

    if problems:
        print("\nWARNINGS:")
        for p in problems:
            print(" -", p)
    else:
        print("\nNo basic issues detected.")

    # show a couple of samples
    print("\n=== samples ===")
    print(df.sample(min(3, len(df)), random_state=42)[["id","lang","question","answer"]])

if __name__ == "__main__":
    main()
