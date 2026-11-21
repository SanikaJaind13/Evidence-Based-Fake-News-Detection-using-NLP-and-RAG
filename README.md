# 🔎 Evidence-Based Fake News Detection Using NLP and RAG

An AI-powered fact-checking app that does **not** just label news "fake" or "real".
It extracts factual claims, searches the web for evidence, ranks that evidence,
and uses an LLM (Retrieval-Augmented Generation) to decide whether each claim is
**Supported**, **Contradicted** or has **Insufficient Evidence** — with
reasoning and clickable source citations.

Runs fully on your Mac with **no PyTorch / Transformers / FAISS / model downloads**
(total install is roughly 150 MB, mostly scikit-learn and Streamlit).

---

## ✨ Features

| Component | How it is implemented |
|---|---|
| Claim extraction | Regex sentence splitter + check-worthiness scoring (numbers, dates, entities, factual verbs; penalises opinions/questions) |
| Evidence retrieval | Tavily Search API + `requests` / `BeautifulSoup` page extraction (parallel, with snippet fallback) |
| Semantic similarity | TF-IDF (1–2 grams) + cosine similarity + number/entity coverage, over overlapping sentence windows |
| RAG | Top passages → Groq LLM (JSON output, cites evidence ids) |
| Guardrails | Citations are validated; verdicts without valid citations are downgraded to *Insufficient Evidence* |
| Explainability | Reasoning, supporting / contradicting passages, similarity scores, source-reliability labels |
| History | SQLite (`data/history.db`), reopen / delete / clear, JSON report download |
| Input types | Claim, headline, full article text, or a URL |

## 🗂️ Folder structure

```
fakenews-rag/
├── app.py                  # Streamlit frontend
├── requirements.txt
├── .env.example            # copy to .env and add keys
├── setup_mac.sh            # optional one-command setup
├── src/
│   ├── config.py           # settings / env loading
│   ├── models.py           # dataclasses (Claim, Passage, ClaimResult, ...)
│   ├── claim_extractor.py  # NLP: sentence split + claim scoring
│   ├── retriever.py        # Tavily search + page scraping
│   ├── credibility.py      # domain reliability heuristic
│   ├── similarity.py       # TF-IDF + cosine evidence ranking
│   ├── verdict.py          # Groq RAG prompt, parsing, guardrails
│   ├── pipeline.py         # orchestration + aggregation
│   ├── database.py         # SQLite history
│   ├── errors.py, utils.py
├── tests/                  # 41 offline unit tests (no API keys needed)
├── scripts/evaluate.py     # accuracy / confusion matrix on labelled claims
├── data/sample_claims.json # 13 labelled claims for evaluation
└── docs/PROJECT_REPORT.md  # academic write-up (architecture, method, limits)
```

## 🍎 Setup on macOS (beginner friendly)

### 1. Install Python 3.11
Open **Terminal** (Cmd + Space → "Terminal") and run:
```bash
# Install Homebrew first if you don't have it: https://brew.sh
brew install python@3.11
python3.11 --version     # should print Python 3.11.x
```

### 2. Get free API keys
- **Groq**: sign up at https://console.groq.com/keys → *Create API Key*
- **Tavily**: sign up at https://app.tavily.com → copy your API key

### 3. Open the project in VS Code
Unzip the project, then in Terminal:
```bash
cd ~/Downloads/fakenews-rag      # adjust to where you unzipped it
code .                           # opens VS Code (or use File → Open Folder)
```
Use VS Code's built-in terminal (**Terminal → New Terminal**) for the next steps.

### 4. Create a virtual environment and install dependencies
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
*(Shortcut: `bash setup_mac.sh` does steps 4 and the `.env` copy for you.)*

If VS Code asks to select an interpreter, choose `.venv` (Cmd+Shift+P → "Python: Select Interpreter").

### 5. Add your API keys
```bash
cp .env.example .env
```
Open `.env` in VS Code and paste your keys:
```
GROQ_API_KEY=gsk_xxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxx
```

### 6. Run the app
```bash
streamlit run app.py
```
Your browser opens at http://localhost:8501. Try one of the example buttons, then press **Verify**.
Stop the app with `Ctrl + C`. Next time, just run `source .venv/bin/activate` then `streamlit run app.py`.

## 🧪 Running the tests
No API keys or internet needed (HTTP calls are faked):
```bash
python -m pytest -q          # or: python -m unittest discover -s tests -t .
```

## 📊 Evaluating accuracy (for your report)
```bash
python scripts/evaluate.py --limit 13
```
Prints accuracy, a confusion matrix and per-class precision/recall/F1, and saves
`data/evaluation_results.csv`. Add your own labelled claims to `data/sample_claims.json`.

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: python3.11` | `brew install python@3.11`, then reopen Terminal |
| "Missing API key(s)" in the app | Check `.env` is in the project root (not `.env.example`) and restart Streamlit |
| "Groq rejected the API key" / "Tavily rejected…" | Re-copy the key; no quotes or spaces |
| "rate limit reached" | Free tiers are limited; wait ~1 minute or lower *Max claims* in the sidebar |
| Many claims come back *Insufficient Evidence* | Rephrase as a plain, specific factual sentence; increase *Web pages per claim* |
| `ModuleNotFoundError` | Activate the venv: `source .venv/bin/activate` |
| Groq model not found | Models get retired; set `GROQ_MODEL` in `.env` to a current one from console.groq.com/docs/models |

## ⚠️ Limitations
- TF-IDF matches **words, not meaning**: paraphrases can be ranked lower. (Trade-off for staying lightweight.)
- The web can itself contain misinformation; the source-reliability heuristic is a simple domain list.
- LLMs can make mistakes; the app mitigates this with citation validation but cannot remove it entirely.
- Very recent events may have little indexed evidence; paywalled pages fall back to search snippets.
- This is an educational tool, not a replacement for professional fact-checkers.

## 🚀 Future work
Multilingual claims (Hindi/Marathi), stance classification with a small local model, caching of search results,
time-aware evidence weighting, and a larger benchmark (e.g. FEVER / LIAR style data).
