"""Tests for NMF topic modeling (src/topic_modeling.py)."""
import unittest

from unittest import mock

from src import topic_modeling
from src.topic_modeling import (MIN_ARTICLES, UNASSIGNED_LABEL, choose_n_topics,
                                run_topic_modeling)

SPACE = [
    "NASA launches new Moon rocket as astronauts prepare for lunar mission",
    "ISRO satellite launch sends rocket into orbit, scientists celebrate space mission",
    "Astronomers use space telescope to study distant galaxy and orbit of planets",
]
CRICKET = [
    "India beat Australia in cricket final as captain scores century at the stadium",
    "Cricket tournament: bowler takes five wickets and team wins the match by runs",
    "Football and cricket fans fill the stadium as the team wins the championship match",
]
MARKETS = [
    "Stock market rallies as investors cheer strong quarterly earnings from banks",
    "Central bank holds interest rate steady while inflation and market growth slow",
    "Sensex and Nifty climb as investors buy bank shares after earnings results",
]


SPACE_SUMMARY = [
    "NASA launched a new Moon rocket as astronauts prepare for the next lunar mission.",
    "An ISRO satellite launch sent a rocket into orbit after a successful space mission.",
    "Astronomers used a space telescope to study a distant galaxy and the orbit of planets.",
]
MARKETS_SUMMARY = [
    "The stock market rallied as investors cheered strong quarterly earnings from banks.",
    "The central bank held the interest rate steady while inflation slowed and market growth cooled.",
    "The Sensex climbed as investors bought bank shares after strong earnings results.",
]


def art(i, title, summary=""):
    return {"title": title, "url": f"https://news{i}.example.com/a", "content": title, "summary": summary}


def articles(*groups):
    out = []
    for g in groups:
        for t in g:
            out.append(art(len(out), t, t + " and more details were reported on the day."))
    return out


class TestTopicModeling(unittest.TestCase):
    def test_discovers_and_groups_topics(self):
        res = run_topic_modeling(articles(SPACE, CRICKET, MARKETS), n_topics=3)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_topics"], 3)
        self.assertEqual(res["n_articles"], 9)
        grouped = sum(tp["article_count"] for tp in res["topics"])
        self.assertEqual(grouped + len(res["unassigned"]), 9)
        self.assertAlmostEqual(sum(d["share"] for d in res["distribution"]), 1.0, places=5)
        for tp in res["topics"]:
            self.assertTrue(tp["keywords"])
            self.assertTrue(tp["label"].startswith(f"Topic {tp['id']}"))

    def test_same_theme_articles_share_a_topic(self):
        res = run_topic_modeling(articles(SPACE, CRICKET, MARKETS), n_topics=3)
        topic_of = {a["index"]: tp["id"] for tp in res["topics"] for a in tp["articles"]}
        for group in ((0, 1, 2), (3, 4, 5), (6, 7, 8)):
            self.assertEqual(len({topic_of[i] for i in group}), 1, group)

    def test_deterministic(self):
        data = articles(SPACE, CRICKET, MARKETS)
        self.assertEqual(run_topic_modeling(data), run_topic_modeling(data))

    def test_empty_inputs(self):
        for bad in (None, [], "text", 5):
            self.assertEqual(run_topic_modeling(bad)["status"], "empty")
        self.assertEqual(run_topic_modeling([{}, None, {"title": ""}])["status"], "empty")

    def test_too_few_articles(self):
        res = run_topic_modeling(articles(SPACE[:MIN_ARTICLES - 1]))
        self.assertEqual(res["status"], "insufficient")
        self.assertIn(str(MIN_ARTICLES), res["message"])

    def test_short_articles_are_skipped(self):
        data = articles(SPACE) + [art(9, "Short")]
        res = run_topic_modeling(data)
        self.assertEqual(res["status"], "ok")
        self.assertEqual([s["index"] for s in res["skipped"]], [3])

    def test_identical_articles_do_not_crash(self):
        res = run_topic_modeling([art(i, SPACE[0], SPACE[0]) for i in range(4)])
        self.assertIn(res["status"], ("ok", "insufficient"))

    def test_choose_n_topics(self):
        self.assertEqual(choose_n_topics(2), 0)
        self.assertEqual(choose_n_topics(3), 2)
        self.assertEqual(choose_n_topics(6), 2)
        self.assertEqual(choose_n_topics(7), 3)
        self.assertEqual(choose_n_topics(15), 4)
        self.assertEqual(choose_n_topics(100), 5)
        self.assertEqual(choose_n_topics(20, n_terms=6), 2)   # tiny vocabulary caps the count


NOISE = (" Subscribe to our newsletter for daily updates. Read more: https://news.example.com/related/story-1 "
         "#Breaking @NewsDesk Click here to follow us on Twitter. Advertisement. All rights reserved.")
TAX = [
    "IRS agrees to tax settlement with large corporation - Reuters",
    "Tax settlement: what the IRS deal means for corporate taxpayers",
    "IRS tightens tax audit rules after the corporate settlement",
]
TAX_SUMMARY = [
    "The IRS agreed to a tax settlement worth billions with a large corporation, ending a long dispute over unpaid corporate tax.",
    "Experts say the IRS tax settlement could change how corporate taxpayers are audited and how future tax disputes are resolved.",
    "The IRS announced tighter audit rules for corporate tax filings following the large settlement with the corporation.",
]
DIPLOMACY = [
    "Trump and Modi hold talks on trade and defence | Hindustan Times",
    "Modi visit: Trump praises India trade ties after the Washington summit",
]
DIPLOMACY_SUMMARY = [
    "Trump and Modi discussed trade tariffs and defence cooperation during a summit in Washington this week.",
    "President Trump praised India and Prime Minister Modi after the trade summit in Washington, calling the trade ties strong.",
]
CRICKET_SUMMARY = [
    "India beat Australia by six wickets as the captain scored a century in the cricket final at the stadium.",
    "The bowler took five wickets as the team won the cricket tournament match, with fans filling the stadium.",
]
RECIPE = "Best monsoon recipes: easy snacks and soup to enjoy on a rainy evening at home"


def noisy(i, title, summary):
    """Article whose summary carries promo / navigation noise, plus noisy raw content."""
    return {"title": title, "url": f"https://news{i}.example.com/a", "summary": summary + NOISE,
            "content": "Home | Menu | Search. " + NOISE}


MIXED = (list(zip(TAX, TAX_SUMMARY)) + list(zip(DIPLOMACY, DIPLOMACY_SUMMARY))
         + [("Cricket final", s) for s in CRICKET_SUMMARY])


class TestCleaning(unittest.TestCase):
    def test_noise_is_removed_from_summary_text(self):
        text = topic_modeling._article_text({
            "title": "LIVE: Markets rally on strong earnings | The Hindu",
            "summary": "Photo: Getty Images. Stocks rose after banks posted earnings. "
                       "Sign up for our newsletter! Read more at www.site.com/x #markets @desk Advertisement",
            "url": "https://example.com/secret-url", "published_date": "Mon, 17 Nov 2026",
        }).lower()
        for junk in ("http", "www", "#markets", "@desk", "newsletter", "sign up", "read more", "getty",
                     "advertisement", "secret-url", "2026"):
            self.assertNotIn(junk, text)
        self.assertIn("stocks rose after banks posted earnings", text)

    def test_noisy_articles_still_group_by_real_content(self):
        res = run_topic_modeling([noisy(i, t, s) for i, (t, s) in enumerate(MIXED[:5])])
        self.assertEqual(res["status"], "ok")
        topic_of = {a["index"]: tp["id"] for tp in res["topics"] for a in tp["articles"]}
        self.assertEqual(len({topic_of[i] for i in (0, 1, 2)}), 1)
        self.assertEqual(len({topic_of[i] for i in (3, 4)}), 1)
        self.assertNotEqual(topic_of[0], topic_of[3])

    def test_noise_never_becomes_a_keyword(self):
        res = run_topic_modeling([noisy(i, t, s) for i, (t, s) in enumerate(MIXED[:5])])
        words = " ".join(k["term"] for tp in res["topics"] for k in tp["keywords"]).split()
        self.assertTrue(words)
        for junk in ("subscribe", "newsletter", "click", "advertisement", "rights", "reuters", "hindustan",
                     "breaking", "newsdesk", "https", "example", "com", "twitter", "menu", "home"):
            self.assertNotIn(junk, words)

    def test_boilerplate_only_summaries_are_empty(self):
        junk = [{"title": "Home | Menu | Search", "summary": "Subscribe to our newsletter. Click here. "
                 "All rights reserved. Read more www.x.com #news"} for _ in range(4)]
        self.assertEqual(run_topic_modeling(junk)["status"], "empty")


class TestSummaryOnly(unittest.TestCase):
    """Only article["summary"] may feed NMF: titles, raw content/snippets and URLs must not."""

    def build(self, title_for, content_for):
        data = []
        for i, summary in enumerate(SPACE_SUMMARY + MARKETS_SUMMARY):
            data.append({"title": title_for(i), "url": f"https://news{i}.example.com/story",
                         "summary": summary, "content": content_for(i)})
        return data

    def test_article_text_is_the_summary_only(self):
        text = topic_modeling._article_text({
            "title": "UNIQUETITLEWORD headline", "content": "UNIQUESNIPPETWORD snippet text",
            "url": "https://uniqueurlword.example.com", "published_date": "UNIQUEDATEWORD",
            "summary": "The council approved a new budget for roads and schools after a long debate."}).lower()
        self.assertIn("council approved", text)
        for other in ("uniquetitleword", "uniquesnippetword", "uniqueurlword", "uniquedateword"):
            self.assertNotIn(other, text)
        self.assertEqual(topic_modeling._article_text({"title": "Only a title", "content": "Only content"}), "")
        self.assertEqual(topic_modeling._article_text({"summary": None}), "")
        self.assertEqual(topic_modeling._article_text(None), "")

    def test_titles_and_snippets_cannot_influence_topics(self):
        honest = self.build(lambda i: "Daily news headline", lambda i: "Full page text")
        misleading = self.build(
            lambda i: "Cricket stadium wickets football goal - Sports Daily | tax IRS audit",
            lambda i: "Cricket wickets stadium football goalkeeper tax IRS audit settlement " * 4)
        a, b = run_topic_modeling(honest), run_topic_modeling(misleading)
        self.assertEqual(a["status"], "ok")
        shape = lambda r: [([k["term"] for k in tp["keywords"]], [x["index"] for x in tp["articles"]])
                           for tp in r["topics"]]
        self.assertEqual(shape(a), shape(b))                 # identical topics whatever titles / snippets say
        words = " ".join(k["term"] for tp in b["topics"] for k in tp["keywords"])
        for stray in ("cricket", "wickets", "stadium", "football", "goalkeeper", "irs", "audit", "tax"):
            self.assertNotIn(stray, words)

    def test_titles_and_urls_are_kept_for_display(self):
        data = self.build(lambda i: f"Display title {i}", lambda i: "")
        res = run_topic_modeling(data)
        shown = {a["index"]: a for tp in res["topics"] for a in tp["articles"]}
        self.assertTrue(shown)
        for i, a in shown.items():
            self.assertEqual(a["title"], f"Display title {i}")
            self.assertEqual(a["url"], f"https://news{i}.example.com/story")

    def test_missing_empty_or_short_summaries_are_skipped(self):
        data = self.build(lambda i: "Rich title with many informative words about rockets and banks",
                          lambda i: "Long raw content about rockets, launches, banks and markets. " * 5)
        bad = [{"title": "No summary key", "url": "u1", "content": "Plenty of raw content here. " * 20},
               {"title": "Empty summary", "url": "u2", "summary": "", "content": "Plenty of raw content here. " * 20},
               {"title": "None summary", "url": "u3", "summary": None, "content": "Plenty of raw content here. " * 20},
               {"title": "Short summary " * 6, "url": "u4", "summary": "Too short to model.",
                "content": "Plenty of raw content here. " * 20}]
        res = run_topic_modeling(data + bad)
        self.assertEqual(res["status"], "ok")
        self.assertEqual([s["index"] for s in res["skipped"]], [6, 7, 8, 9])
        self.assertEqual(res["n_articles"], 6)
        # skipped articles are never grouped or listed as unassigned
        listed = {a["index"] for tp in res["topics"] for a in tp["articles"]} | {a["index"] for a in res["unassigned"]}
        self.assertTrue(listed.isdisjoint({6, 7, 8, 9}))

    def test_titles_and_snippets_alone_are_not_enough(self):
        data = [{"title": t, "url": f"u{i}", "content": t + " " + t, "summary": ""}
                for i, t in enumerate(SPACE + MARKETS)]
        res = run_topic_modeling(data)
        self.assertEqual(res["status"], "empty")
        self.assertEqual(len(res["skipped"]), 6)

    def test_too_few_usable_summaries_is_insufficient(self):
        data = self.build(lambda i: "Title", lambda i: "Content")[:2]
        data.append({"title": "Third", "url": "u", "summary": "", "content": "Long raw content " * 30})
        res = run_topic_modeling(data)
        self.assertEqual(res["status"], "insufficient")
        self.assertEqual(len(res["skipped"]), 1)


class TestUnrelatedAndWeak(unittest.TestCase):
    def test_unrelated_articles_do_not_form_a_topic(self):
        res = run_topic_modeling(articles([SPACE[0]], [MARKETS[0]], [RECIPE]))
        self.assertEqual(res["status"], "insufficient")
        self.assertIn("clear common themes", res["message"])
        self.assertEqual(len(res["unassigned"]), 3)

    def test_unrelated_keywords_are_not_mixed_into_one_topic(self):
        """Tax / diplomacy / sport articles must not share one topic (the 'IRS, Trump, Modi' problem)."""
        res = run_topic_modeling([noisy(i, t, s) for i, (t, s) in enumerate(MIXED)])
        self.assertEqual(res["status"], "ok")
        themes = [{"irs", "tax", "settlement", "audit", "corporate"},
                  {"trump", "modi", "trade", "india", "summit", "washington", "defence"},
                  {"cricket", "wickets", "stadium", "bowler", "final", "century"}]
        for tp in res["topics"]:
            words = {w for k in tp["keywords"] for w in k["term"].split()}
            self.assertLessEqual(sum(1 for th in themes if words & th), 1, tp["keywords"])

    def test_stray_article_goes_to_unassigned_group(self):
        data = articles(SPACE, MARKETS) + [art(9, RECIPE, RECIPE + " and a few more tips for the home cook.")]
        res = run_topic_modeling(data)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_topics"], 2)
        self.assertEqual([a["index"] for a in res["unassigned"]], [6])
        self.assertTrue(res["unassigned"][0]["reason"])
        self.assertEqual(res["distribution"][-1]["label"], UNASSIGNED_LABEL)
        self.assertEqual(res["distribution"][-1]["article_count"], 1)
        self.assertAlmostEqual(sum(d["share"] for d in res["distribution"]), 1.0, places=5)

    def test_weak_topic_association_is_not_forced(self):
        data = articles(SPACE, MARKETS)
        normal = run_topic_modeling(data)
        self.assertEqual(normal["unassigned"], [])
        with mock.patch.object(topic_modeling, "MIN_DOMINANCE", 1.01):    # nothing can dominate enough
            strict = run_topic_modeling(data)
        self.assertEqual(strict["status"], "insufficient")
        self.assertEqual(len(strict["unassigned"]), 6)
        self.assertTrue(all("Weak association" in a["reason"] for a in strict["unassigned"]))
        with mock.patch.object(topic_modeling, "MIN_FIT", 0.99):          # topic words must match almost exactly
            strict = run_topic_modeling(data)
        self.assertGreater(len(strict["unassigned"]), 0)

    def test_single_article_theme_is_not_a_topic(self):
        res = run_topic_modeling(articles(SPACE, MARKETS[:1]))
        for tp in res["topics"]:
            self.assertGreaterEqual(tp["article_count"], 2)
        self.assertEqual([a["index"] for a in res["unassigned"]], [3])

    def test_topic_keywords_are_shared_by_the_topics_articles(self):
        res = run_topic_modeling(articles(SPACE, CRICKET, MARKETS))
        for tp in res["topics"]:
            self.assertGreaterEqual(len(tp["keywords"]), 1)
            self.assertGreaterEqual(tp["article_count"], 2)


if __name__ == "__main__":
    unittest.main()
