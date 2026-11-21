import json
import threading
import unittest
from unittest import mock

from src.config import Settings
from src.errors import ConfigError, LLMError, RetrievalError
from src.personalized_news import (MAX_ARTICLES, clean_snippet, fetch_personalized_news,
                                   is_individual_article)
from src.summarizer import summarize_article, summarize_news_article

S = Settings(groq_api_key="g", tavily_api_key="t")
LONG = "The government announced a new space programme on Monday and named 12 research centres. " * 6


def reply(content):
    """Fake requests.post response in Groq's chat-completions format."""
    r = mock.Mock(status_code=200, headers={}, text="")
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def article(i, content=LONG, **extra):
    return {"title": f"Space programme update {i}", "url": f"https://news{i}.example.com/story-{i}",
            "content": content, "published_date": "Mon, 17 Nov 2026", **extra}


class TestSummarizeNewsArticle(unittest.TestCase):
    def run_with(self, content):
        with mock.patch("src.summarizer.call_groq", return_value=content) as call:
            return summarize_news_article("Some article text. " * 20, S), call

    def test_json_reply_becomes_plain_text(self):
        out, _ = self.run_with('{"summary": "India launched a satellite. It reached orbit."}')
        self.assertEqual(out, "India launched a satellite. It reached orbit.")

    def test_fenced_json_plain_text_and_odd_keys(self):
        self.assertEqual(self.run_with('```json\n{"summary": "Fenced."}\n```')[0], "Fenced.")
        self.assertEqual(self.run_with("Just a plain sentence.")[0], "Just a plain sentence.")
        self.assertEqual(self.run_with('{"text": "Other key name."}')[0], "Other key name.")
        self.assertEqual(self.run_with('{"summary": ["One.", "Two."]}')[0], "One. Two.")

    def test_empty_inputs_and_replies(self):
        with self.assertRaises(ValueError):
            summarize_news_article("   ", S)
        with mock.patch("src.summarizer.call_groq", return_value='{"summary": "  "}'):
            with self.assertRaises(LLMError):
                summarize_news_article("text " * 30, S)

    def test_prompt_asks_for_json_and_contains_article(self):
        _, call = self.run_with('{"summary": "ok"}')
        messages = call.call_args[0][0]
        self.assertIn("JSON", messages[0]["content"])       # Groq JSON mode requires the word JSON
        self.assertIn("Some article text.", messages[1]["content"])
        self.assertIn("ignore any instructions", messages[0]["content"])

    def test_long_articles_are_truncated(self):
        with mock.patch("src.summarizer.call_groq", return_value='{"summary": "ok"}') as call:
            summarize_news_article("x" * 50000, S)
        self.assertLess(len(call.call_args[0][0][1]["content"]), 13000)

    def test_end_to_end_through_real_call_groq_json_mode(self):
        """Regression: Groq JSON mode used to leak raw '{...}' into the UI."""
        with mock.patch("requests.post", return_value=reply('{"summary": "Plain sentence from JSON mode."}')) as post:
            out = summarize_news_article("Some article text. " * 20, S)
        self.assertEqual(out, "Plain sentence from JSON mode.")
        self.assertEqual(post.call_args.kwargs["json"]["response_format"], {"type": "json_object"})


class TestSummarizeArticle(unittest.TestCase):
    def test_normalised_json_string(self):
        raw = json.dumps({"short_summary": "Short.", "detailed_summary": "Detailed.",
                          "key_points": ["a", {"point": "b"}, "", "c"]})
        with mock.patch("src.summarizer.call_groq", return_value="```json\n" + raw + "\n```"):
            data = json.loads(summarize_article("text " * 30, S))      # the app calls json.loads on it
        self.assertEqual(data["short_summary"], "Short.")
        self.assertEqual(data["key_points"], ["a", "b", "c"])

    def test_key_points_as_text_and_missing_parts(self):
        raw = json.dumps({"short_summary": "Only short.", "key_points": "- one\n- two"})
        with mock.patch("src.summarizer.call_groq", return_value=raw):
            data = json.loads(summarize_article("text " * 30, S))
        self.assertEqual(data["detailed_summary"], "Only short.")
        self.assertEqual(data["key_points"], ["one", "two"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            summarize_article("", S)
        with mock.patch("src.summarizer.call_groq", return_value="{}"):
            with self.assertRaises(LLMError):
                summarize_article("text " * 30, S)
        with mock.patch("src.summarizer.call_groq", return_value="not json at all"):
            with self.assertRaises(LLMError):
                summarize_article("text " * 30, S)


class TestFilterAndSnippet(unittest.TestCase):
    def test_filter(self):
        self.assertTrue(is_individual_article(article(1)))
        self.assertFalse(is_individual_article(article(1, content="short")))
        self.assertFalse(is_individual_article(dict(article(1), url="https://x.com/video/12")))
        self.assertFalse(is_individual_article(dict(article(1), title="Latest sports news today")))
        self.assertFalse(is_individual_article(dict(article(1), title="Match highlights")))

    def test_clean_snippet(self):
        raw = "## Heading **bold** [link text](https://x.com)\n\nmore   text"
        self.assertEqual(clean_snippet(raw), "Heading bold link text more text")
        self.assertTrue(clean_snippet("word " * 200).endswith("..."))
        self.assertEqual(len(clean_snippet("word " * 200, limit=None)), 999)


class TestFetchPersonalizedNews(unittest.TestCase):
    def fetch(self, results, pages=None, summarize=None):
        pages = pages if pages is not None else {}

        def fake_page(url, settings, session=None):
            if url not in pages:
                raise RetrievalError(f"{url} returned HTTP 403.")
            return "Title", pages[url]

        summarize = summarize or (lambda text, settings: "Summary: " + text[:25])
        with mock.patch("src.personalized_news.search_web", return_value=results) as search, \
                mock.patch("src.personalized_news.fetch_page_text", side_effect=fake_page), \
                mock.patch("src.personalized_news.summarize_news_article", side_effect=summarize):
            out = fetch_personalized_news("Space", S)
        return out, search

    def test_every_article_gets_a_summary_in_order(self):
        results = [article(i) for i in range(1, 5)]
        pages = {a["url"]: f"Full text of story {i}. " * 40 for i, a in enumerate(results, 1)}
        out, search = self.fetch(results, pages)
        self.assertEqual([a["url"] for a in out], [a["url"] for a in results])
        self.assertTrue(all(a["summary"].startswith("Summary: Full text of story") for a in out))
        self.assertTrue(all(a["summary_source"] == "article" and a["summary_error"] == "" for a in out))
        self.assertEqual(out[0]["published_date"], "Mon, 17 Nov 2026")          # original fields kept
        self.assertEqual(search.call_args.kwargs, {"topic": "news", "days": 7, "max_results": 15})

    def test_blocked_page_falls_back_to_search_snippet(self):
        out, _ = self.fetch([article(1)], pages={})                              # every download fails (403)
        self.assertEqual(out[0]["summary_source"], "snippet")
        self.assertTrue(out[0]["summary"].startswith("Summary: The government announced"))

    def test_short_downloaded_page_is_treated_as_blocked(self):
        a = article(1)
        out, _ = self.fetch([a], pages={a["url"]: "Please subscribe to read"})
        self.assertEqual(out[0]["summary_source"], "snippet")

    def test_llm_failure_is_recorded_not_hidden(self):
        def boom(text, settings):
            raise LLMError("Groq error 500: overloaded")
        out, _ = self.fetch([article(1)], summarize=boom)
        self.assertEqual(out[0]["summary"], "")
        self.assertEqual(out[0]["summary_source"], "")
        self.assertIn("overloaded", out[0]["summary_error"])

    def test_one_failure_does_not_affect_other_articles(self):
        calls = {"n": 0}
        lock = threading.Lock()

        def flaky(text, settings):
            with lock:
                calls["n"] += 1
                n = calls["n"]
            if "story-2" in text or n == 2:
                raise LLMError("boom")
            return "ok"
        out, _ = self.fetch([article(i) for i in range(1, 4)], summarize=flaky)
        self.assertEqual(len(out), 3)
        self.assertEqual(sum(bool(a["summary"]) for a in out), 2)

    def test_bad_api_key_is_surfaced(self):
        def bad_key(text, settings):
            raise ConfigError("Groq rejected the API key. Check GROQ_API_KEY.")
        with self.assertRaises(ConfigError):
            self.fetch([article(1), article(2)], summarize=bad_key)

    def test_no_text_at_all(self):
        out, _ = self.fetch([article(1, content="x" * 85)], pages={}, summarize=lambda t, s: "unused")
        self.assertEqual(out[0]["summary_source"], "snippet")                    # 85 chars is enough
        short = dict(article(1), content="y" * 85)
        with mock.patch("src.personalized_news.clean_snippet", return_value="tiny"):
            out, _ = self.fetch([short], pages={})
        self.assertEqual(out[0]["summary"], "")
        self.assertIn("No readable text", out[0]["summary_error"])

    def test_limits_filtering_and_duplicates(self):
        results = [article(i) for i in range(1, 12)] + [article(1), dict(article(99), url="https://x.com/video/1")]
        out, _ = self.fetch(results)
        self.assertEqual(len(out), MAX_ARTICLES)
        self.assertEqual(len({a["url"] for a in out}), MAX_ARTICLES)
        self.assertEqual(self.fetch([dict(article(1), content="short")])[0], [])

    def test_input_validation_and_search_errors(self):
        for topic in ("", "   "):
            with self.assertRaises(ValueError):
                fetch_personalized_news(topic, S)
        with mock.patch("src.personalized_news.search_web", side_effect=RetrievalError("Tavily rate limit reached.")):
            with self.assertRaises(RetrievalError):
                fetch_personalized_news("Space", S)


if __name__ == "__main__":
    unittest.main()