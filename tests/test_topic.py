"""Tests for automatic topic detection (src/topic.py) and its pipeline / UI integration."""
import json
import os
import tempfile
import unittest
from unittest import mock

from src import database, topic
from src.config import Settings
from src.models import INSUFFICIENT, SUPPORTED, AnalysisResult, Document
from src.personalized_news import NEWS_CATEGORIES
from src.pipeline import analyze
from src.topic import GENERAL, TOPIC_PROFILES, detect_topic

# (expected label, text). Written independently of the profiles' wording where possible.
LABELLED = [
    ("Politics", "Opposition parties walked out of Parliament on Tuesday, accusing the government of rushing the bill through."),
    ("Politics", "The Prime Minister met the French President in Paris to discuss a new defence treaty."),
    ("Politics", "Voters in three states head to the polls next week as both candidates make their final campaign push."),
    ("Sports", "Virat Kohli smashed a century as India beat Australia by six wickets in the second ODI at the stadium."),
    ("Sports", "Argentina edged Brazil 2-1 in a thrilling World Cup qualifier, with the winning goal in stoppage time."),
    ("Sports", "Neeraj Chopra won gold in the javelin throw, adding another medal to India's Olympic tally."),
    ("Technology", "Apple unveiled a new iPhone with an upgraded chip and faster 5G modem at its annual launch event."),
    ("Technology", "A ransomware attack forced the software provider to shut down servers, and hackers demanded payment."),
    ("Technology", "OpenAI released a new language model that can write code, summarise documents and answer questions."),
    ("Science", "ISRO's Chandrayaan-3 lander touched down near the Moon's south pole, a first for any space agency."),
    ("Science", "Astronomers using a space telescope discovered a distant galaxy that formed shortly after the Big Bang."),
    ("Science", "Researchers sequenced the genome of an extinct species of wolf, revealing how it evolved."),
    ("Business", "The Sensex surged 800 points as investors cheered strong quarterly earnings from banking companies."),
    ("Business", "The Reserve Bank of India kept the repo rate unchanged, saying inflation remains above its target."),
    ("Business", "The retail giant announced layoffs after reporting a drop in quarterly revenue and profit."),
    ("Health", "The World Health Organization declared the outbreak a public health emergency as hospitals reported infections."),
    ("Health", "A clinical trial found the drug reduced the risk of heart attack in patients with diabetes."),
    ("Health", "Doctors warn that vitamin deficiency and poor diet are driving a rise in obesity among children."),
    ("Entertainment", "The Bollywood actor's new film opened strongly at the box office over the weekend."),
    ("Entertainment", "Netflix renewed the hit web series for a third season and revealed the full cast of the sequel."),
    ("Entertainment", "The singer won three Grammy awards, including song of the year, at Sunday night's ceremony."),
    ("Education", "The CBSE announced the date sheet for the Class 12 board exams, with the first paper in February."),
    ("Education", "Thousands of students protested outside the university after the entrance exam results were delayed."),
    ("Education", "Colleges extended the admission deadline and announced new scholarship schemes for students."),
    ("Environment", "Delhi's air quality plunged to severe levels as stubble burning and vehicle emissions worsened pollution."),
    ("Environment", "Global warming is accelerating glacier melt and driving up sea levels, a climate report warns."),
    ("Environment", "Wildfires fuelled by a record heatwave and drought have destroyed thousands of hectares of forest."),
    ("General", "The quick brown fox jumps over the lazy dog near the river."),
    ("General", "Lorem ipsum dolor sit amet, consectetur adipiscing elit."),
    ("General", "भारत ने ऑस्ट्रेलिया को छह विकेट से हराया और श्रृंखला जीत ली।"),
]


class TestProfiles(unittest.TestCase):
    def test_categories_match_the_app_taxonomy(self):
        """Topic names must stay in sync with the Personalized News categories."""
        expected = {c for c in NEWS_CATEGORIES if not c.startswith("Other")}
        self.assertEqual(set(TOPIC_PROFILES), expected)

    def test_every_profile_term_is_usable(self):
        for category, terms in TOPIC_PROFILES.items():
            self.assertEqual(len(terms), len(set(terms)), f"duplicate term in {category}")
            for term in terms:
                content_words = [f for f in topic._analyze(term) if " " not in f]
                self.assertLessEqual(len(content_words), 2, f"{category}: {term!r} has more than 2 content words")
                self.assertIsNotNone(topic._term_key(term), f"{category}: {term!r} is only stop-words")
                self.assertEqual(term, term.lower())


class TestDetectTopic(unittest.TestCase):
    def test_result_shape(self):
        r = detect_topic("The Prime Minister and the Cabinet discussed the new election policy in Parliament.")
        self.assertEqual(set(r), {"label", "confidence", "alternatives", "note"})
        self.assertEqual(r["label"], "Politics")
        self.assertTrue(0.0 < r["confidence"] <= 1.0)
        self.assertLessEqual(len(r["alternatives"]), topic.MAX_ALTERNATIVES)
        self.assertNotIn(r["label"], [a["label"] for a in r["alternatives"]])
        self.assertEqual(r["note"], "")
        json.dumps(r)  # must be JSON-serialisable (stored in history)

    def test_labelled_samples(self):
        wrong = []
        for expected, text in LABELLED:
            got = detect_topic(text)["label"]
            if got != expected:
                wrong.append((expected, got, text[:50]))
        accuracy = 1 - len(wrong) / len(LABELLED)
        self.assertGreaterEqual(accuracy, 0.9, wrong)
        # When unsure the detector must say General, never pick a different topic.
        self.assertTrue(all(got == GENERAL for _, got, _ in wrong), wrong)

    def test_empty_text_is_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                detect_topic(bad)

    def test_unusable_text_is_general_with_a_note(self):
        for text in ("12345 !!! ???", "the of and to a", "भारत ने ऑस्ट्रेलिया को हराया"):
            r = detect_topic(text)
            self.assertEqual((r["label"], r["confidence"], r["alternatives"]), (GENERAL, 0.0, []), text)
            self.assertTrue(r["note"])

    def test_plurals_are_folded(self):
        self.assertEqual(detect_topic("Elections, voters and lawmakers dominated the politics of the week.")["label"],
                         "Politics")

    def test_confidence_grows_with_evidence(self):
        weak = detect_topic("Dengue cases rise sharply in Mumbai.")
        strong = detect_topic("Doctors at the hospital treated dengue patients; the health ministry "
                              "ordered a vaccination and public health drive against the outbreak and infection.")
        self.assertEqual((weak["label"], strong["label"]), ("Health", "Health"))
        self.assertLess(weak["confidence"], 0.4)
        self.assertGreater(strong["confidence"], weak["confidence"])
        self.assertGreater(strong["confidence"], 0.6)

    def test_longer_text_needs_two_distinct_topic_terms(self):
        filler = " ".join(["The committee met on a rainy afternoon and talked about various local arrangements"] * 3)
        self.assertEqual(detect_topic(filler + " The stadium was mentioned once.")["label"], GENERAL)

    def test_only_the_start_of_a_long_text_is_used(self):
        padding = "Nothing much happened in the quiet village that year. " * 120   # > MAX_CHARS
        self.assertGreater(len(padding), topic.MAX_CHARS)
        late_sports = "The cricket team won the tournament final with a late wicket and the captain lifted the trophy."
        self.assertEqual(detect_topic(padding + late_sports)["label"], GENERAL)
        self.assertEqual(detect_topic(late_sports + " " + padding)["label"], "Sports")

    def test_ocr_style_noise_is_tolerated(self):
        ocr = "BREAKING  NEWS\n\nThe  Reserve Bank kept the repo  rate unchanged\nas inflation re-\nmains high |  \x0c"
        self.assertEqual(detect_topic(ocr)["label"], "Business")


# ---------------------------------------------------------------- pipeline integration
def fake_search(query, settings):
    return [Document("https://reuters.com/x", "Match report", "reuters.com",
                     "India beat Australia by six wickets in the second ODI. Virat Kohli scored a century "
                     "at the stadium. The team leads the series.", 0.95, "Wire")]


def fake_llm(messages, settings):
    return json.dumps({"verdict": "Supported", "confidence": 0.9, "reasoning": "ok [E1]", "supporting": ["E1"]})


SPORTS = "Virat Kohli scored a century as India beat Australia by six wickets in the second ODI at the stadium."


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(groq_api_key="g", tavily_api_key="t",
                                 db_path=os.path.join(self.tmp.name, "h.db"))

    def tearDown(self):
        self.tmp.cleanup()

    def run_analyze(self, text, **kw):
        return analyze(text, self.settings, search_fn=fake_search, llm_fn=fake_llm, **kw)

    def test_text_input_gets_a_topic_and_existing_results_are_unchanged(self):
        res = self.run_analyze(SPORTS, save=False)
        self.assertEqual(res.topic["label"], "Sports")
        self.assertEqual(res.category, "Sports")
        self.assertEqual(res.overall_verdict, SUPPORTED)          # fact-checking untouched
        self.assertEqual(len(res.claim_results), 1)
        self.assertIsNone(res.sentiment)                          # sentiment is still set by the app only

    def test_url_input_uses_the_downloaded_article_not_the_url(self):
        with mock.patch("src.pipeline.fetch_page_text", return_value=("t", SPORTS)):
            res = self.run_analyze("https://cricket-news.example.com/politics-and-elections", save=False)
        self.assertIn("Kohli", res.input_text)
        self.assertEqual(res.category, "Sports")    # the URL itself contains "politics" and "elections"

    def test_image_ocr_text_goes_through_the_same_path(self):
        ocr_text = "INDIA  WIN   SERIES\n" + SPORTS.replace(" as ", "\nas ") + "\x0c"
        res = self.run_analyze(ocr_text, save=False)
        self.assertEqual(res.category, "Sports")

    def test_topic_is_saved_in_history(self):
        res = self.run_analyze(SPORTS)
        self.assertIsNotNone(res.id)
        stored = database.get_analysis(self.settings.db_path, res.id)
        self.assertEqual(stored["topic"]["label"], "Sports")
        self.assertEqual(stored["category"], "Sports")
        row = database.list_history(self.settings.db_path)[0]
        self.assertEqual(row["category"], "Sports")               # existing history column is now filled

    def test_unclear_topic_is_recorded_as_general(self):
        res = self.run_analyze("Gustave Eiffel designed the tower that was completed in 1889.", save=False)
        self.assertEqual((res.category, res.topic["label"]), (GENERAL, GENERAL))

    def test_topic_failure_never_blocks_fact_checking(self):
        with mock.patch("src.pipeline.detect_topic", side_effect=RuntimeError("boom")):
            res = self.run_analyze(SPORTS)
        self.assertEqual(res.overall_verdict, SUPPORTED)
        self.assertIsNone(res.topic)
        self.assertIsNone(res.category)
        self.assertIsNotNone(res.id)                              # still saved to history

    def test_old_history_rows_without_a_topic_still_load(self):
        old = AnalysisResult("t", [], INSUFFICIENT, 0.0, "s", "2026-01-01 00:00:00 UTC").to_dict()
        old.pop("topic")
        rid = database.save_analysis(self.settings.db_path, old)
        self.assertNotIn("topic", database.get_analysis(self.settings.db_path, rid))


# ---------------------------------------------------------------- UI (real Streamlit render)
class TestTopicInUI(unittest.TestCase):
    """Runs app.py with Streamlit's AppTest and checks the section order on screen."""

    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"streamlit testing unavailable: {exc}")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        env = {"DB_PATH": os.path.join(self.tmp.name, "ui.db"), "GROQ_API_KEY": "g", "TAVILY_API_KEY": "t"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def render(self, topic_value):
        from streamlit.testing.v1 import AppTest

        def fake_analyze(text, settings, **kwargs):
            return AnalysisResult(
                input_text=text, claim_results=[], overall_verdict=SUPPORTED, overall_confidence=0.9,
                summary="0 claim(s) checked.", created_at="2026-11-21 10:00:00 UTC",
                category=(topic_value or {}).get("label"), topic=topic_value)

        at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=30)
        at.session_state["auth_user"] = {"id": 1, "email": "t@example.com"}
        with mock.patch("src.pipeline.analyze", fake_analyze):
            at.run()
            at.text_area(key="input_text").set_value("India beat Australia in the cricket match.")
            at.button[[b.label for b in at.button].index("Verify")].click()
            at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return [m.value for m in at.markdown]

    @staticmethod
    def index_of(blocks, needle):
        return next(i for i, b in enumerate(blocks) if needle in b)

    def test_topic_is_shown_directly_above_sentiment(self):
        blocks = self.render({"label": "Sports", "confidence": 0.82,
                              "alternatives": [{"label": "Entertainment", "confidence": 0.1}], "note": ""})
        verdict, topic_i = self.index_of(blocks, "Overall assessment"), self.index_of(blocks, "Detected Topic")
        topic_card, sentiment = topic_i + 1, self.index_of(blocks, "Sentiment Analysis")
        self.assertLess(verdict, topic_i)
        self.assertEqual(topic_card + 1, sentiment)               # nothing between the two sections
        self.assertIn("Sports", blocks[topic_card])
        self.assertIn("82%", blocks[topic_card])
        self.assertIn("Entertainment (10%)", blocks[topic_card])
        self.assertNotIn("Low confidence", blocks[topic_card])

    def test_low_confidence_and_general_states(self):
        low = self.render({"label": "Health", "confidence": 0.25, "alternatives": [], "note": ""})
        self.assertIn("Low confidence", low[self.index_of(low, "Detected Topic") + 1])

        general = self.render({"label": "General", "confidence": 0.0, "alternatives": [],
                               "note": "No clear topic signal."})
        card = general[self.index_of(general, "Detected Topic") + 1]
        self.assertIn("General", card)
        self.assertIn("No clear topic signal.", card)
        self.assertNotIn("Confidence", card)

    def test_results_without_a_topic_render_as_before(self):
        blocks = self.render(None)
        self.assertFalse(any("Detected Topic" in b for b in blocks))
        self.assertTrue(any("Sentiment Analysis" in b for b in blocks))

    def test_label_is_html_escaped(self):
        blocks = self.render({"label": "<script>x</script>", "confidence": 0.5, "alternatives": [], "note": ""})
        card = blocks[self.index_of(blocks, "Detected Topic") + 1]
        self.assertNotIn("<script>", card)


if __name__ == "__main__":
    unittest.main()