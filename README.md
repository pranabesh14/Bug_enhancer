# Defect Enhancer — Contextual Training + POC (Jira + RAG + LoRA)
2) builds a retrieval index over your domain tickets (RAG),
3) optionally does **contextual training** via LoRA on your data (T5‑base), and
4) exposes a FastAPI endpoint that enhances vague defect text into a structured, app‑specific description.


## Quickstart


### 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
cp config.example.toml config.toml
```
Fill `.env` with your keys (Jira, OpenAI) or set `PROVIDER=ollama` to use local models via Ollama.


### 1) Ingest Jira (optional but recommended)
```bash
python scripts/ingest_jira.py --jql "project=APP AND created >= -180d" --limit 1000
```


### 2) Build training pairs (optional; improves fine‑tuning)
```bash
python scripts/make_training_pairs.py --input data/raw_jira_export.jsonl \
--output data/training_pairs.jsonl --eval data/eval_pairs.jsonl
```


### 3) Build retrieval index
```bash
python scripts/build_index.py --input data/raw_jira_export.jsonl --outdir index/
```


### 4) (Optional) Contextual training with LoRA (T5‑base)
```bash
python scripts/fine_tune_lora_t5.py --train data/training_pairs.jsonl --eval data/eval_pairs.jsonl \
--out models/t5-lora-domain
```


### 5) Run the POC service
```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload
```


### 6) Try it
```bash
curl -X POST http://localhost:8000/enhance \
-H 'Content-Type: application/json' \
-d '{
"project_key": "APP",
"vague_text": "login not working",
"update_jira": false
}'
```


## Notes
- RAG uses `sentence-transformers/all-MiniLM-L6-v2` + FAISS (CPU‑friendly).
- LLM can be: OpenAI (`PROVIDER=openai`), **Ollama** (`PROVIDER=ollama`, model e.g. `llama3`), or the fine‑tuned **T5‑base** LoRA via Transformers (`PROVIDER=huggingface`).
- The service can optionally **update the Jira issue** description.