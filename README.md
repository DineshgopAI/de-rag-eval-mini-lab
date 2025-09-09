# DE/EN RAG Eval Mini-Lab

Tiny evaluation lab comparing **BM25** vs **MiniLM (dense)** on a small German/English QA set.  
Tracks **recall@k**, **MRR@10**, and **latency**, plus a 1-page **Streamlit** demo.

## Why this exists
Most RAG demos always return something—even for nonsense queries.  
This lab shows **honest abstain + fallback** behavior so you can see when retrieval is **not** confident.

## What’s inside
- `data/de_en_qa.csv` — tiny DE/EN QA/context dataset
- `src/build_indexes.py` — builds BM25 + MiniLM indexes
- `src/eval_retrieval.py` — computes recall@k / MRR@10 / latency → `reports/metrics.json`
- `src/visualize.py` — saves plots → `reports/plots/`
- `src/app_streamlit.py` — demo app with **abstain/fallback** guards

## Quickstart
```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt

python .\src\build_indexes.py
python .\src\eval_retrieval.py
python .\src\visualize.py
streamlit run .\src\app_streamlit.py
