"""Topic Modeling (NMF) over a set of news articles.

This is separate from ``topic.py`` (Topic Detection), which labels ONE article with a
fixed category. Topic Modeling looks at SEVERAL articles together (the results of the
Personalized News feature) and discovers the themes they share, with no predefined labels.

How it works
------------
1. Clean   - ONLY the article's AI-generated ``summary`` is used as text. Titles, raw
             ``content`` / search snippets, URLs, sources and dates never reach the model
             (titles and URLs are kept only to display the grouped articles). The summary
             is stripped of URLs, hashtags, promotional / navigation text and generic news
             phrases. Articles with a missing, empty or too short summary are skipped.
2. Preprocess - the project's existing ``preprocess_text`` (stop-words, lemmatization),
             then plural forms are merged only when both forms occur in the articles.
3. TF-IDF  - scikit-learn, with settings that depend on how many articles there are
             (see ``_tfidf_settings``): words must be shared by at least two articles on
             larger collections, very common words are dropped, and two-word phrases are
             used once there are enough articles to support them.
4. NMF     - splits the TF-IDF matrix into W (articles x topics) and H (topics x words).
5. Group   - an article joins a topic only if that topic clearly dominates it AND the
             topic's vocabulary actually matches the article (cosine similarity). Weak
             articles go to the "Unassigned / Low Confidence" group instead of being
             forced into a topic. A topic needs at least ``MIN_TOPIC_ARTICLES`` articles.

Only scikit-learn / numpy are used, so there are no new dependencies.

Graceful handling
-----------------
``run_topic_modeling`` never raises for bad input. It always returns a dict whose
``status`` is one of:
    "ok"            topics were found
    "empty"         no articles were given (or none had any usable text)
    "insufficient"  too few articles, too little vocabulary, or no shared themes
    "error"         NMF itself failed (message explains)
"""
from __future__ import annotations

import logging
import re

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .utils import clean_text

logger = logging.getLogger(__name__)

MIN_ARTICLES = 3          # NMF needs at least a few documents to separate themes
MIN_WORDS = 8             # a summary with fewer content words than this is skipped
MIN_VOCAB = 6             # distinct terms needed across all articles
MAX_TOPICS = 5            # upper bound for the automatic topic count
KEYWORDS_PER_TOPIC = 8
MIN_KEYWORDS = 2          # a topic with fewer usable keywords is not a real theme
REL_KEYWORD_WEIGHT = 0.25 # a keyword must carry >= 25% of the topic's strongest keyword weight
LABEL_KEYWORDS = 3        # words used to name a topic, e.g. "Topic 1: space, launch, nasa"
MAX_CHARS = 4000          # per article; a headline + summary is plenty
RANDOM_STATE = 42         # fixed so the same articles always give the same topics

# --- weak-association rules (an article failing either one is "Unassigned / Low Confidence") ---
MIN_DOMINANCE = 0.55      # the best topic must hold >= 55% of the article's total topic weight
MIN_FIT = 0.20            # cosine similarity between the article and that topic's vocabulary
MIN_TOPIC_ARTICLES = 2    # a "theme" needs at least two articles

UNASSIGNED_LABEL = "Unassigned / Low Confidence"

# --------------------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------------------
_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|io|gov|edu|co\.uk|co\.in)(?:/\S*)?\b", re.I)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_SOCIAL = re.compile(r"(?<!\w)[#@]\w+")                        # #hashtags and @mentions
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_MARK = re.compile(r"[*_`>]{1,3}|#{1,6}\s*")

# Each pattern removes the phrase AND the rest of its sentence (promo/navigation lines are
# usually a whole sentence, e.g. "Subscribe to our newsletter for daily updates.").
_REST = r"[^.!?\n]*[.!?]?"
_BOILERPLATE = [re.compile(p + _REST, re.I) for p in (
    # promotional / subscription
    r"\b(?:subscribe|sign up|signup|log in|login|register)\b(?:\s+(?:to|for|now|today|here))?",
    r"\b(?:download|install|get)\s+(?:our|the)\s+(?:app|newsletter)",
    r"\bnewsletters?\b", r"\bfollow us\b", r"\bfollow\s+(?:@|on\s+(?:twitter|facebook|instagram|x)\b)",
    r"\b(?:advertisement|advertise with us|sponsored(?: content)?|promoted|paid content|partner content)\b",
    r"\b(?:click|tap)\s+here\b", r"\bjoin (?:our|the)\s+(?:whatsapp|telegram|community|channel)\b",
    r"\b(?:free|special|exclusive)\s+(?:offer|trial|deal)\b", r"\bunlimited access\b",
    # website navigation / page furniture
    r"\b(?:read|view|see)\s+(?:more|also|full|the full|related|next)\b",
    r"\b(?:continue|keep)\s+reading\b", r"\bstory continues\b", r"\bshare (?:this|on)\b",
    r"\b(?:related|trending|popular|top|more)\s+(?:articles?|stories|news|videos?|posts?)\b",
    r"\b(?:most read|top stories|top headlines|latest headlines|more from|you may also like|also read)\b",
    r"\b(?:skip to|jump to|back to)\s+(?:main\s+)?(?:content|top|navigation)\b",
    r"\b(?:home|menu|search|e-?paper)\s*[|»›>]\s*\w+",
    r"\b(?:cookies?|privacy policy|terms of (?:use|service)|all rights reserved|copyright\b|©)",
    r"\b(?:photo|image|picture|video)\s*(?:credit|courtesy|:)",
    r"\b(?:getty images|file photo|file image|screenshot)\b",
    r"\b(?:this|the)\s+(?:article|story|report)\s+(?:was\s+)?(?:first\s+|originally\s+)?(?:appeared|published)\b",
    r"\b(?:updated|published|posted|last updated)\s*(?:on|at|:)", r"\b(?:ist|gmt|utc|est|pst)\b",
    # generic news phrases (no theme of their own)
    r"\b(?:breaking(?: news)?|latest news|news live|live updates?|watch live|developing story)\b[:\-]?",
    r"\baccording to (?:a |an |the )?(?:report|sources?|officials?|statement|reuters|ap|pti|ani)\b",
    r"\b(?:told|tell|telling)\s+reporters\b", r"\bin a statement\b",
    r"\bon condition of anonymity\b", r"\bdid not (?:immediately )?respond to (?:a )?requests?\b",
    r"\b(?:reuters|associated press|press trust of india|pti|ani)\b\s*[-–—)]?",
)]


def _strip_noise(text: str) -> str:
    """Remove URLs, hashtags, promotional / navigation text and generic news phrases."""
    text = clean_text(str(text or ""))
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _SOCIAL.sub(" ", text)
    text = _MD_MARK.sub(" ", text)
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" .-|")


def _article_text(article: dict) -> str:
    """Modeling text for one article: its cleaned AI summary and nothing else.

    The title, raw ``content`` / search snippet, URL, source and date are deliberately ignored,
    so website headlines, snippets and page furniture cannot influence the discovered topics.
    Returns "" when the summary is missing or empty.
    """
    if not isinstance(article, dict):
        return ""
    return _strip_noise(article.get("summary"))[:MAX_CHARS].strip()


# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------
# Common in news text but carry no theme. Lemmatized forms are listed too (e.g. "announce").
_NEWS_NOISE = frozenset({
    # reporting verbs / filler
    "say", "said", "says", "tell", "told", "report", "reports", "reported", "reporter", "news",
    "according", "announce", "announced", "reveal", "revealed", "state", "statement", "add", "added",
    "claim", "claimed", "mention", "note", "noted", "explain", "confirm", "confirmed", "expect",
    "expected", "plan", "plans", "make", "take", "get", "got", "go", "come", "use", "include",
    "see", "seen", "read", "show", "give", "put", "set", "keep", "want", "need", "know", "think",
    # generic nouns
    "new", "latest", "update", "updates", "article", "story", "summary", "headline", "headlines",
    "official", "officials", "spokesperson", "source", "sources", "media", "press", "conference",
    "people", "person", "man", "woman", "group", "member", "members", "country", "world", "area",
    "thing", "way", "case", "part", "lot", "number", "point", "line", "page", "site", "video",
    # time words
    "year", "years", "day", "days", "week", "weeks", "month", "months", "time", "today",
    "yesterday", "tomorrow", "tonight", "recent", "recently", "ago", "last", "next", "first",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august", "september", "october",
    "november", "december",
    # fillers / quantities
    "also", "one", "two", "three", "may", "could", "would", "might", "will", "like", "just",
    "still", "even", "much", "many", "several", "per", "cent", "percent", "million", "billion",
    "ap", "reuters", "pti", "ani", "ist", "gmt", "utc", "com", "www", "http", "https",
})
_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | _NEWS_NOISE)
_STOP_SET = frozenset(_STOP_WORDS)
_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"   # alphabetic words of 3+ letters


def _preprocess(text: str) -> str:
    """Use the project's own preprocessing; fall back to a light version if NLTK data is missing."""
    try:
        from .preprocessing import preprocess_text
        processed = preprocess_text(text).get("processed_text", "")
        if processed:
            return processed
    except Exception as exc:  # LookupError (NLTK data), import problems, ...
        logger.info("Falling back to simple preprocessing for topic modeling: %s", exc)
    return " ".join(re.findall(r"[a-z]{3,}", text.lower()))


def _content_tokens(processed: str) -> list[str]:
    return [t for t in re.findall(_TOKEN_PATTERN, processed.lower()) if t not in _STOP_SET]


def _fold_plurals(token_lists: list[list[str]]) -> list[str]:
    """Merge "election"/"elections" only when BOTH forms occur, so names like "Jones" stay intact."""
    vocab = {t for toks in token_lists for t in toks}
    fold = {t: t[:-1] for t in vocab
            if len(t) > 3 and t.endswith("s") and not t.endswith("ss") and t[:-1] in vocab}
    return [" ".join(fold.get(t, t) for t in toks) for toks in token_lists]


# --------------------------------------------------------------------------------------
# TF-IDF / topic count
# --------------------------------------------------------------------------------------
def _tfidf_settings(n_docs: int) -> dict:
    """TF-IDF settings that suit the collection size.

    * ngram_range - single words for small sets; one- and two-word phrases ("interest rate")
      only from 8 articles, when a phrase can be repeated in at least two of them.
    * min_df - on 5+ articles a word must appear in 2+ articles. A word found in one article
      cannot describe a shared theme (it is how stray names such as "Modi" or "IRS" ended up
      inside unrelated topics). Tiny sets keep every word.
    * max_df - words found in most articles (the search topic itself, "government") say nothing
      about how the articles differ, so they are dropped once there are 4+ articles.
    """
    if n_docs < 4:
        max_df = 1.0
    elif n_docs < 6:
        max_df = 0.9
    else:
        max_df = 0.6
    return {
        "ngram_range": (1, 2) if n_docs >= 8 else (1, 1),
        "min_df": 1 if n_docs < 10 else max(2, int(0.05 * n_docs)),
        "max_df": max_df,
    }


def _vectorize(docs: list[str]):
    """Fit TF-IDF with the size-based settings; relax them if too little vocabulary survives."""
    attempts = [_tfidf_settings(len(docs)), {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0}]
    for settings in attempts:
        try:
            vectorizer = TfidfVectorizer(stop_words=_STOP_WORDS, token_pattern=_TOKEN_PATTERN,
                                         sublinear_tf=True, **settings)
            matrix = vectorizer.fit_transform(docs)
        except ValueError:      # empty vocabulary after filtering
            continue
        if matrix.shape[1] >= MIN_VOCAB:
            return vectorizer, matrix
    return None, None


def choose_n_topics(n_articles: int, n_terms: int | None = None) -> int:
    """Automatic topic count from the corpus size (about sqrt(articles)).

    3-6 articles -> 2, 7-11 -> 3, 12-19 -> 4, 20+ -> 5 (the maximum). The count is also capped
    so every topic can have at least two articles (articles // 2) and by the vocabulary
    (about 3 terms per topic). It is deliberately not lower than sqrt(n): too few topics
    would merge unrelated stories into one. If it is slightly too high that is harmless,
    because a topic that ends up with fewer than two confident articles is dissolved into the
    Unassigned / Low Confidence group instead of being shown.
    """
    if n_articles < MIN_ARTICLES:
        return 0
    k = int(round(n_articles ** 0.5))
    k = min(k, MAX_TOPICS, n_articles // MIN_TOPIC_ARTICLES)
    if n_terms is not None:
        k = min(k, n_terms // 3)
    return max(2, k)


# --------------------------------------------------------------------------------------
# Result helpers
# --------------------------------------------------------------------------------------
def _result(status: str, message: str = "", **extra) -> dict:
    base = {
        "status": status,
        "message": message,
        "n_articles": 0,
        "n_topics": 0,
        "topics": [],
        "distribution": [],
        "unassigned": [],
        "skipped": [],
    }
    base.update(extra)
    return base


def _label(rank: int, keywords: list[dict]) -> str:
    words = ", ".join(k["term"] for k in keywords[:LABEL_KEYWORDS])
    return f"Topic {rank}: {words}" if words else f"Topic {rank}"


def _topic_keywords(h_row: np.ndarray, terms: np.ndarray, member_rows: np.ndarray) -> list[dict]:
    """Top words of a topic, using only words shared by at least two of the topic's articles.

    Requiring shared words stops two unrelated articles from being presented as one topic: if
    they have fewer than ``MIN_KEYWORDS`` strong words in common, an empty list is returned and
    the caller dissolves the topic.
    """
    order = h_row.argsort()[::-1]
    top = float(h_row[order[0]])
    if top <= 1e-9 or len(member_rows) < MIN_TOPIC_ARTICLES:
        return []
    shared = (member_rows > 0).sum(axis=0) >= MIN_TOPIC_ARTICLES
    chosen = []
    for i in order[:KEYWORDS_PER_TOPIC * 4]:
        if h_row[i] < REL_KEYWORD_WEIGHT * top:
            break
        if shared[i]:
            chosen.append(i)
        if len(chosen) == KEYWORDS_PER_TOPIC:
            break
    if len(chosen) < MIN_KEYWORDS:
        return []
    # a phrase such as "interest rate" makes its single words ("interest", "rate") redundant
    covered = {tok for i in chosen if " " in terms[i] for tok in terms[i].split()}
    chosen = [i for i in chosen if " " in terms[i] or terms[i] not in covered]
    return [{"term": str(terms[i]), "weight": round(float(h_row[i]), 4)} for i in chosen]


def _fit_and_group(matrix, terms, kept, k):
    """Fit NMF with ``k`` topics and group the articles. Returns (topics, unassigned)."""
    model = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=1000)
    W = model.fit_transform(matrix)        # articles x topics
    H = model.components_                  # topics x words
    X = matrix.toarray()

    row_sum = W.sum(axis=1)
    best = W.argmax(axis=1)
    h_norm = np.linalg.norm(H, axis=1)
    x_norm = np.linalg.norm(X, axis=1)

    members: dict[int, list[int]] = {t: [] for t in range(k)}
    unassigned: list[dict] = []
    fits = np.zeros(len(kept))
    for i, info in enumerate(kept):
        t = int(best[i])
        if row_sum[i] <= 1e-9 or x_norm[i] <= 1e-9 or h_norm[t] <= 1e-9:
            unassigned.append({**info, "confidence": 0.0, "reason": "No topic signal"})
            continue
        dominance = float(W[i, t] / row_sum[i])
        fit = float(X[i] @ H[t] / (x_norm[i] * h_norm[t]))   # cosine: does the topic describe the article?
        fits[i] = fit
        if dominance < MIN_DOMINANCE or fit < MIN_FIT:
            unassigned.append({**info, "confidence": round(fit, 2),
                               "reason": "Weak association with every topic"})
        else:
            members[t].append(i)

    topics = []
    for t in range(k):
        idx = sorted(members[t], key=lambda i: W[i, t], reverse=True)
        keywords = _topic_keywords(H[t], terms, X[idx]) if idx else []
        if len(idx) < MIN_TOPIC_ARTICLES or len(keywords) < MIN_KEYWORDS:
            for i in idx:   # a single article (or a topic without clear words) is not a shared theme
                unassigned.append({**kept[i], "confidence": round(float(fits[i]), 2),
                                   "reason": "Not enough related articles"})
            continue
        topics.append({
            "id": 0, "label": "", "keywords": keywords, "article_count": len(idx), "share": 0.0,
            "articles": [{**kept[i], "weight": round(float(W[i, t]), 4),
                          "confidence": round(float(fits[i]), 2)} for i in idx],
        })
    unassigned.sort(key=lambda a: a["index"])
    return topics, unassigned


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def run_topic_modeling(articles: list[dict] | None, n_topics: int | None = None) -> dict:
    """Discover topics in ``articles`` (the Personalized News results) with NMF.

    Returns a dict (see module docstring for ``status``)::

        {
          "status", "message", "n_articles", "n_topics",
          "topics": [{
              "id", "label", "keywords": [{"term", "weight"}], "article_count", "share",
              "articles": [{"index", "title", "url", "weight", "confidence"}]   # strongest first
          }],
          "distribution": [{"topic", "label", "article_count", "share"}],   # includes the
                          # "Unassigned / Low Confidence" group (topic id 0) when it is not empty
          "unassigned": [{"index", "title", "url", "confidence", "reason"}],
          "skipped":    [{"index", "title", "url"}],   # missing / empty / too short summary
        }

    ``share`` is the fraction of all analysed articles, so the distribution adds up to 1.
    """
    if not articles or not isinstance(articles, (list, tuple)):
        return _result("empty", "No articles are available. Fetch news in the Personalized News tab first.")

    # 1. Clean + preprocess the AI summaries; skip articles whose summary is missing or too short.
    token_lists, kept, skipped = [], [], []
    for index, art in enumerate(articles):
        art = art if isinstance(art, dict) else {}
        info = {"index": index, "title": str(art.get("title") or "Untitled"), "url": str(art.get("url") or "")}
        tokens = _content_tokens(_preprocess(_article_text(art)))
        if len(tokens) < MIN_WORDS:
            skipped.append(info)
            continue
        token_lists.append(tokens)
        kept.append(info)

    if not kept:
        return _result("empty", "None of the articles have an AI summary long enough for topic modeling.",
                       n_articles=len(articles), skipped=skipped)
    if len(kept) < MIN_ARTICLES:
        return _result(
            "insufficient",
            f"Topic modeling needs at least {MIN_ARTICLES} articles with a usable AI summary "
            f"(found {len(kept)}). Fetch a broader news topic and try again.",
            n_articles=len(kept), skipped=skipped)
    docs = _fold_plurals(token_lists)

    # 2. TF-IDF.
    vectorizer, matrix = _vectorize(docs)
    if matrix is None:
        return _result("insufficient", "The articles share too little distinct vocabulary to find topics.",
                       n_articles=len(kept), skipped=skipped)
    terms = np.array(vectorizer.get_feature_names_out())

    # 3. NMF + grouping.
    k = n_topics if n_topics else choose_n_topics(len(kept), len(terms))
    k = max(2, min(int(k), len(kept) - 1, len(terms) - 1))
    topics, unassigned = [], []
    try:
        # Automatic count: try k, k-1 ... 2 and keep the one that places the most articles in
        # a coherent topic (larger k wins ties). An explicit ``n_topics`` is tried as given.
        candidates = [k] if n_topics else list(range(k, 1, -1))
        best_assigned = -1
        for k_try in candidates:
            topics_try, unassigned_try = _fit_and_group(matrix, terms, kept, k_try)
            assigned = sum(tp["article_count"] for tp in topics_try)
            if assigned > best_assigned:
                best_assigned, topics, unassigned = assigned, topics_try, unassigned_try
        if not topics and k > 2:
            topics, unassigned = _fit_and_group(matrix, terms, kept, 2)
    except Exception as exc:
        logger.warning("NMF failed: %s", exc)
        return _result("error", f"Topic modeling could not be completed: {exc}",
                       n_articles=len(kept), skipped=skipped)

    if not topics:
        return _result(
            "insufficient",
            "The articles do not share clear common themes, so no reliable topics were found. "
            "Try a broader news topic or fetch more articles.",
            n_articles=len(kept), unassigned=unassigned, skipped=skipped)

    # Largest topic first, numbered 1..n so labels and charts read in order.
    topics.sort(key=lambda tp: tp["article_count"], reverse=True)
    for rank, tp in enumerate(topics, start=1):
        tp["id"] = rank
        tp["label"] = _label(rank, tp["keywords"])
        tp["share"] = tp["article_count"] / len(kept)

    distribution = [{"topic": tp["id"], "label": tp["label"],
                     "article_count": tp["article_count"], "share": tp["share"]} for tp in topics]
    if unassigned:
        distribution.append({"topic": 0, "label": UNASSIGNED_LABEL,
                             "article_count": len(unassigned), "share": len(unassigned) / len(kept)})

    return _result("ok", n_articles=len(kept), n_topics=len(topics), topics=topics,
                   distribution=distribution, unassigned=unassigned, skipped=skipped)
