"""Phase 0: new settings and the optional model fields."""
import json
import os
import unittest
from unittest import mock

from src.config import load_settings
from src.models import AnalysisResult, ClaimResult, Claim, Document, Passage

NEW_VARS = ("SESSION_TTL_HOURS", "NEWS_CACHE_MINUTES", "TRENDING_WINDOW_DAYS", "OCR_LANGS")


def env_without_new_vars(**extra):
    env = {k: v for k, v in os.environ.items() if k not in NEW_VARS}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


class TestNewSettings(unittest.TestCase):
    def test_defaults(self):
        with env_without_new_vars():
            s = load_settings()
        self.assertEqual((s.session_ttl_hours, s.news_cache_minutes, s.trending_window_days),
                         (168, 60, 7))
        self.assertEqual(s.ocr_langs, "eng+hin+mar")

    def test_environment_overrides(self):
        with env_without_new_vars(SESSION_TTL_HOURS="24", NEWS_CACHE_MINUTES="15",
                                  TRENDING_WINDOW_DAYS="3", OCR_LANGS="eng"):
            s = load_settings()
        self.assertEqual((s.session_ttl_hours, s.news_cache_minutes, s.trending_window_days), (24, 15, 3))
        self.assertEqual(s.ocr_langs, "eng")

    def test_invalid_or_non_positive_values_fall_back(self):
        with env_without_new_vars(SESSION_TTL_HOURS="abc", NEWS_CACHE_MINUTES="0",
                                  TRENDING_WINDOW_DAYS="-5"):
            s = load_settings()
        self.assertEqual((s.session_ttl_hours, s.news_cache_minutes, s.trending_window_days),
                         (168, 60, 7))

    def test_ocr_langs_rejects_unsafe_values(self):
        for bad in ("eng; rm -rf /", "eng+", "+eng", "", "eng hin", "-l eng", "e"):
            with env_without_new_vars(OCR_LANGS=bad):
                self.assertEqual(load_settings().ocr_langs, "eng+hin+mar", bad)
        with env_without_new_vars(OCR_LANGS="eng+hin+mar+osd"):
            self.assertEqual(load_settings().ocr_langs, "eng+hin+mar+osd")

    def test_existing_behaviour_unchanged(self):
        with env_without_new_vars(MAX_CLAIMS="0", REQUEST_TIMEOUT="oops"):
            s = load_settings()
        self.assertEqual(s.max_claims, 0)        # old _int_env accepted 0 ...
        self.assertEqual(s.request_timeout, 12)  # ... and fell back on garbage
        with env_without_new_vars(GROQ_API_KEY="", TAVILY_API_KEY=""):
            self.assertEqual(load_settings().missing_keys(), ["GROQ_API_KEY", "TAVILY_API_KEY"])

    def test_overrides_still_work(self):
        with env_without_new_vars():
            s = load_settings(max_claims=5, session_ttl_hours=2)
        self.assertEqual((s.max_claims, s.session_ttl_hours), (5, 2))


class TestModels(unittest.TestCase):
    def test_existing_positional_construction_still_works(self):
        p = Passage("E1", "text", "https://a.com", "A", "a.com", 0.6, 0.9, "Wire")
        self.assertIsNone(p.published_date)
        self.assertEqual(p.source_name, "")
        d = Document("https://a.com", "A", "a.com", "text", 0.9, "Wire")
        self.assertIsNone(d.published_date)
        self.assertEqual(d.source_name, "")

    def test_new_passage_fields(self):
        p = Passage("E1", "t", "https://a.com", "A", "a.com", 0.5, 0.5, "x",
                    published_date="2026-11-02 08:00:00", source_name="The A Times")
        self.assertEqual((p.published_date, p.source_name), ("2026-11-02 08:00:00", "The A Times"))

    def test_analysis_result_new_fields_default_to_unknown(self):
        r = AnalysisResult("t", [], "Supported", 0.8, "s", "2026-11-01 10:00:00 UTC")
        d = r.to_dict()
        for key in ("input_type", "language", "sentiment", "category"):
            self.assertIn(key, d)
            self.assertIsNone(d[key])
        self.assertIsNone(d["id"])
        json.dumps(d)  # still JSON-serialisable

    def test_nested_passage_fields_serialise(self):
        p = Passage("E1", "t", "https://a.com", "A", "a.com", 0.5, 0.5, "x", "2026-11-02", "A Times")
        cr = ClaimResult(claim=Claim("C1", "claim"), supporting=[p])
        d = AnalysisResult("t", [cr], "Supported", 0.8, "s", "now").to_dict()
        sup = d["claim_results"][0]["supporting"][0]
        self.assertEqual((sup["published_date"], sup["source_name"]), ("2026-11-02", "A Times"))


if __name__ == "__main__":
    unittest.main()