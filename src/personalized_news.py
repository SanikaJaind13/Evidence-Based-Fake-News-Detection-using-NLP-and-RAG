"""Personalized news fetching utilities."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from .config import Settings
from .errors import ConfigError
from .retriever import fetch_page_text, search_web
from .summarizer import summarize_news_article

logger = logging.getLogger(__name__)

NEWS_CATEGORIES = [
    "Technology",
    "Politics",
    "Sports",
    "Science",
    "Business",
    "Health",
    "Entertainment",
    "Education",
    "Environment",
    "Other / Customize",
]

MAX_ARTICLES = 7           # articles shown per search
SEARCH_RESULTS = 15        # candidates requested from Tavily before filtering
MIN_FULL_TEXT_CHARS = 300  # a downloaded page shorter than this is treated as blocked / paywalled
MIN_SNIPPET_CHARS = 80     # shortest search snippet worth summarizing
SUMMARY_WORKERS = 3        # parallel page downloads + LLM calls

NO_SUMMARY_MESSAGE = "A summary could not be generated for this article."


def is_individual_article(article: dict) -> bool:
    """Keep individual news articles and reject category or media pages."""
    title = article.get("title", "").strip().lower()
    url = article.get("url", "").strip().lower()
    content = article.get("content", "").strip()

    if not title or not url or len(content) < 80:
        return False

    # Reject video and media pages
    if any(part in url for part in ("/video/", "/watch/", "/clip/")):
        return False

    # Reject known news homepages and category pages
    blocked_url_patterns = [
        "nowallnews.com/sports",
        "foxsports.com/stories",
        "nba.com/news",
        "tmz.com/sports",
        "marca.com/en",
        "sports.yahoo.com/nfl/news",
        "news.google.com",
    ]
    if any(pattern in url for pattern in blocked_url_patterns):
        return False

    # Reject generic category titles
    generic_title_words = [
        "sports news",
        "latest sports news",
        "sports news today",
        "fox sports stories",
        "nba news",
        "google news",
        "latest updates",
    ]
    if any(phrase in title for phrase in generic_title_words):
        return False

    # Reject malformed URLs
    if '"null' in url or "null,null" in url:
        return False

    # Reject video-related articles
    if any(word in title for word in ("watch ", "video", "highlights", "game highlights")):
        return False

    # Reject pages containing unrelated news listings or newsletter content
    noisy_phrases = [
        "see all newsletters",
        "get all your news in one place",
        "breaking:",
        "latest headlines",
    ]
    content_lower = content.lower()
    if any(phrase in content_lower for phrase in noisy_phrases):
        return False

    return True


def clean_snippet(text: str, limit: int | None = 350) -> str:
    """Strip Markdown/whitespace from a search snippet; optionally cut it at a word boundary."""
    text = text or ""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if limit and len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def _summarize_one(article: dict, settings: Settings) -> dict:
    """Return a copy of ``article`` with ``summary``, ``summary_source`` and ``summary_error``.

    The full page is used when it can be downloaded; otherwise the Tavily search
    snippet is summarized, so blocked or paywalled sites still get a summary.
    """
    result = dict(article)
    url = article.get("url", "")
    text, source, error = "", "", ""

    try:
        _, full_text = fetch_page_text(url, settings)
        if len(full_text.strip()) >= MIN_FULL_TEXT_CHARS:
            text, source = full_text, "article"
    except Exception as exc:  # blocked, timeout, not HTML ...
        logger.info("Could not download %s: %s", url, exc)

    if not text:
        snippet = clean_snippet(article.get("content", ""), limit=None)
        if len(snippet) >= MIN_SNIPPET_CHARS:
            text, source = snippet, "snippet"

    summary = ""
    if text:
        try:
            summary = summarize_news_article(text, settings)
        except ConfigError:
            raise  # a bad API key affects every article: surface it to the user
        except Exception as exc:
            logger.warning("Summary failed for %s: %s", url, exc)
            error = str(exc)
    else:
        error = "No readable text was available for this article."

    result.update(summary=summary, summary_source=source if summary else "", summary_error=error)
    return result


def fetch_personalized_news(topic: str, settings: Settings) -> list[dict]:
    """Fetch recent news for ``topic`` and add an AI summary to each article.

    Every returned article keeps the Tavily fields (title, url, content, published_date, ...) plus:
      summary         plain-text 2-3 sentence summary ("" if it could not be generated)
      summary_source  "article" (full page), "snippet" (search preview) or ""
      summary_error   reason when no summary was produced, otherwise ""
    """
    if not topic or not topic.strip():
        raise ValueError("Please select or enter a news topic.")

    results = search_web(f"latest {topic.strip()} news headlines", settings,
                         topic="news", days=7, max_results=SEARCH_RESULTS)

    seen, candidates = set(), []
    for article in results:
        url = article.get("url", "")
        if url in seen or not is_individual_article(article):
            continue
        seen.add(url)
        candidates.append(article)
        if len(candidates) == MAX_ARTICLES:
            break
    if not candidates:
        return []

    with ThreadPoolExecutor(max_workers=min(SUMMARY_WORKERS, len(candidates))) as pool:
        return list(pool.map(lambda a: _summarize_one(a, settings), candidates))  # order preserved