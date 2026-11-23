# 🔎 NEWSINTEL AI — Evidence-Based Fake News Detection with NLP + RAG

NEWSINTEL AI is a Streamlit app that does **not** just label news "fake" or "real". It extracts factual
claims, searches the web for evidence, ranks that evidence, and uses an LLM (Retrieval-Augmented Generation)
to decide whether each claim is **Supported**, **Contradicted** or has **Insufficient Evidence**, with
reasoning and clickable source citations.

Around the fact-checker it adds a set of NLP tools: sentiment analysis, named entity recognition, topic
detection, news summarization, personalized news with topic modeling, and OCR for news screenshots.

There is no PyTorch, Transformers, FAISS or large model download. All matching is done with scikit-learn
(TF-IDF), and the LLM and search are hosted APIs (Groq and Tavily).

---

## ✨ Features

### 1. Fact-check tab
| Step | How it is implemented |
|---|---|
| Input | Plain text (claim, headline, article), a **URL**, or an **uploaded image** (PNG/JPG/JPEG) |
| OCR | Tesseract via `pytesseract` (grayscale + auto-contrast, English). Extracted text is editable before verifying |
| Claim extraction | Regex sentence splitter plus a check-worthiness score (numbers, dates, named entities, factual verbs; penalises opinions and questions). Near-duplicates are removed |
| Evidence retrieval | Tavily Search API, then parallel page download with `requests` + `BeautifulSoup` (falls back to the search snippet for paywalled or blocked pages) |
| Evidence ranking | Overlapping 3-sentence windows → TF-IDF (1–2 grams) cosine similarity + number/entity coverage → hybrid score `0.75·cosine + 0.25·key-term coverage` |
| Source credibility | Transparent domain-list heuristic (fact-checkers, institutions, major news, social media). Shown to the user and given to the LLM as context; it never decides the verdict on its own |
| RAG verdict | Top passages → Groq LLM (JSON output, cites evidence ids) |
| Guardrails | Invalid citation ids are discarded; a *Supported*/*Contradicted* verdict with no valid citation is downgraded to *Insufficient Evidence*; web text is treated as untrusted (prompt-injection rule) |
| Aggregation | Any *Contradicted* claim → overall *Contradicted*; all *Supported* → *Supported*; otherwise *Insufficient Evidence*. One failed claim does not abort the others |
| Report | Per-claim reasoning, supporting/contradicting passages, relevance scores, and a **JSON report download** |

### 2. NLP Lab (opens from the Fact-check tab after a check)
1. **Text preprocessing** (NLTK): lowercasing, URL removal, tokenization, stop-word removal, lemmatization
2. **Sentiment analysis** (VADER): Positive / Negative / Neutral with compound, positive, negative and neutral scores
3. **Named Entity Recognition** (spaCy `en_core_web_sm`): entity table plus counts per category
4. **Topic detection** (TF-IDF + cosine over hand-written category profiles): 15 categories (Politics, Sports, Technology, Science, Business, Health, Entertainment, Education, Environment, Disaster, Crime & Law, International Affairs, Transportation, Agriculture, Defence & Security). Returns *General* when the signal is weak, with a confidence score and runner-up categories
5. **TF-IDF & similarity analysis**: every evidence passage with its hybrid similarity score and source link

### 3. Summarize tab
Paste an article or give a URL. The Groq LLM returns a **short summary**, a **detailed summary** and **key points**.

### 4. History tab
Every completed analysis is saved per user in SQLite. Reopen, delete, or clear your history.

### 5. Personalized News tab
- Pick a category (Technology, Politics, Sports, Science, Business, Health, Entertainment, Education, Environment) or enter a custom topic.
- Fetches up to 7 recent articles (last 7 days) through Tavily's news search, filters out category/video/homepage results, and writes a 2–3 sentence AI summary for each (full page when it can be downloaded, otherwise the search snippet).
- **Topic Modeling (NMF)**: discovers shared themes across the fetched articles using only their AI summaries (TF-IDF + NMF, scikit-learn). Shows a topic distribution chart, keywords per topic, grouped articles, and an "Unassigned / Low Confidence" group for weak matches. Needs at least 3 articles with usable summaries.

### 6. Accounts and UI
- Register / login with email and password. Passwords are hashed with **scrypt** (per-user salt). Sessions use a random token of which only the **SHA-256 hash** is stored.
- History is strictly scoped to the logged-in user.
- Light / dark theme toggle on every page.

---

## 🗂️ Project structure

```
fakenews-rag/
├── app.py                      # Streamlit frontend (login, 4 tabs, NLP Lab, theme)
├── requirements.txt
├── packages.txt                # system packages for deployment (tesseract-ocr)
├── setup_mac.sh                # optional one-command macOS setup
├── pytest.ini
├── .env.example                # copy to .env and add your keys
├── src/
│   ├── config.py               # settings / env loading
│   ├── models.py               # dataclasses (Claim, Passage, ClaimResult, AnalysisResult, ...)
│   ├── errors.py, utils.py     # custom exceptions, small helpers
│   ├── auth.py                 # registration, login (scrypt), sessions
│   ├── database.py             # SQLite schema (v2), migrations, history CRUD
│   ├── pipeline.py             # orchestration + aggregation
│   ├── claim_extractor.py      # sentence splitting + check-worthiness scoring
│   ├── retriever.py            # Tavily search + page scraping
│   ├── credibility.py          # domain reliability heuristic
│   ├── similarity.py           # TF-IDF + cosine hybrid evidence ranking
│   ├── verdict.py              # Groq RAG prompt, parsing, citation guardrails
│   ├── summarizer.py           # article + news summaries (Groq)
│   ├── personalized_news.py    # news fetching, filtering, per-article summaries
│   ├── topic_modeling.py       # NMF topic modeling over news articles
│   ├── topic.py                # single-article topic detection (15 categories)
│   ├── preprocessing.py        # NLTK text preprocessing
│   ├── sentiment.py            # VADER sentiment
│   ├── ner.py                  # spaCy named entity recognition
│   └── ocr.py                  # Tesseract OCR for uploaded images
├── tests/                      # offline unit tests (HTTP and LLM calls are faked)
├── scripts/evaluate.py         # accuracy / confusion matrix on labelled claims
├── data/
│   ├── sample_claims.json      # 13 labelled claims for evaluation
│   └── history.db              # created on first run (git-ignored)
└── docs/PROJECT_REPORT.md      # academic write-up
```

---

## 🍎 Setup on macOS

### 1. Install Python 3.11 and Tesseract
```bash
# Install Homebrew first if you don't have it: https://brew.sh
brew install python@3.11 tesseract
python3.11 --version     # Python 3.11.x
```
Tesseract is only needed for the image/OCR feature. `spacy==3.8.0` and `numpy==1.26.4` in `requirements.txt` work on Python 3.9–3.12.

### 2. Get free API keys
- **Groq**: https://console.groq.com/keys → *Create API Key*
- **Tavily**: https://app.tavily.com → copy your API key

### 3. Create a virtual environment and install dependencies
```bash
cd fakenews-rag
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
This also installs the spaCy English model (`en_core_web_sm`) from its GitHub release URL, so you need an internet connection.
*(Shortcut: `bash setup_mac.sh` does this and copies `.env.example` to `.env`.)*

> **First run:** NLTK data (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`) is downloaded automatically the first time the app imports `src/preprocessing.py`. This needs internet once.

### 4. Add your API keys
```bash
cp .env.example .env
```
Open `.env` and fill in your own keys (never commit this file):
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501, **register an account**, log in, then try one of the example buttons and press **Verify**.
Stop with `Ctrl + C`. Next time: `source .venv/bin/activate` then `streamlit run app.py`.

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | – | **Required.** Groq API key |
| `TAVILY_API_KEY` | – | **Required.** Tavily API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | LLM used for verdicts and summaries. If the model is unavailable, `openai/gpt-oss-120b` and `openai/gpt-oss-20b` are tried as fallbacks |
| `MAX_CLAIMS` | `3` | Claims verified per submission |
| `RESULTS_PER_CLAIM` | `5` | Web pages retrieved per claim |
| `PASSAGES_PER_CLAIM` | `6` | Evidence passages sent to the LLM |
| `REQUEST_TIMEOUT` | `12` | Seconds per HTTP request (Groq calls use 3× this) |
| `DB_PATH` | `data/history.db` | SQLite file location |

`.env.example` also lists `SESSION_TTL_HOURS`, `NEWS_CACHE_MINUTES`, `TRENDING_WINDOW_DAYS` and `OCR_LANGS`. These are read by `src/config.py` but are **not yet used** by the app: sessions currently last 7 days, news results are not cached, there is no trending view, and OCR runs in English only.

---

## 🧪 Running the tests

The unit tests need no API keys or internet (HTTP and LLM calls are faked):
```bash
python -m pytest -q --ignore=tests/test_summarizer.py
```
`tests/test_ner.py`, `tests/test_preprocessing.py` and `tests/test_summarizer.py` are small demo scripts rather than assertion tests. `test_summarizer.py` calls the live Groq API when imported, so it is excluded above (run it only with valid keys).

Covered areas: claim extraction, TF-IDF ranking, retriever and pipeline (with fakes), verdict parsing and guardrails, database schema/migration and per-user scoping, credibility rules, personalized news filtering, topic detection, and NMF topic modeling.

## 📊 Evaluating accuracy

```bash
python scripts/evaluate.py --limit 13
```
Uses real API keys. Runs the pipeline on `data/sample_claims.json` (6 contradicted, 5 supported, 2 insufficient-evidence claims; one claim per run), prints accuracy, a confusion matrix and per-class precision/recall/F1, and saves `data/evaluation_results.csv`. Add your own labelled claims to the JSON file.

## ☁️ Deployment notes (Streamlit Community Cloud)

- `packages.txt` installs the system package `tesseract-ocr` for OCR.
- Add `GROQ_API_KEY` and `TAVILY_API_KEY` in the platform's **Secrets** settings, not in the repository.
- The SQLite database lives on the app's local disk, so accounts and history may be reset when the app restarts or redeploys.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: python3.11` | `brew install python@3.11`, then reopen Terminal |
| "Missing API key(s)" in the app | Check `.env` is in the project root (not `.env.example`) and restart Streamlit |
| "Groq rejected the API key" / "Tavily rejected…" | Re-copy the key; no quotes or spaces |
| "rate limit reached" | Free tiers are limited; wait about a minute or lower `MAX_CLAIMS` |
| Many claims come back *Insufficient Evidence* | Rephrase as a plain, specific factual sentence, or increase `RESULTS_PER_CLAIM` |
| `ModuleNotFoundError` | Activate the venv: `source .venv/bin/activate` |
| `OSError: [E050] Can't find model 'en_core_web_sm'` | `pip install -r requirements.txt` again, or `python -m spacy download en_core_web_sm` |
| `TesseractNotFoundError` when extracting text from an image | `brew install tesseract` (or install `tesseract-ocr` on Linux) |
| NLTK `LookupError` | Connect to the internet and restart once so the data can download |
| Groq model not found | Set `GROQ_MODEL` in `.env` to a current model from https://console.groq.com/docs/models |

---

## ⚠️ Limitations
- TF-IDF matches **words, not meaning**, so paraphrases can rank lower (the trade-off for staying lightweight).
- The web can itself contain misinformation, and the source-reliability score is a simple domain list.
- LLMs can make mistakes. Citation validation reduces this but cannot remove it.
- Very recent events may have little indexed evidence; paywalled pages fall back to search snippets.
- Claim extraction, sentiment, NER, topic detection and OCR are English-only.
- Topic detection uses hand-written keyword profiles, so it can return *General* for niche subjects.
- This is an educational tool, not a replacement for professional fact-checkers.

## 🚀 Future work
Multilingual claims and OCR (Hindi/Marathi), stance classification with a small local model, caching of search
results, time-aware evidence weighting, a trending-news view, and a larger benchmark (FEVER / LIAR style data).
