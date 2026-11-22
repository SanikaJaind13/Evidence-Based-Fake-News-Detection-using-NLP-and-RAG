"""Automatic topic detection (news category) with TF-IDF + cosine similarity.

No training data, no model downloads and no extra dependencies: it reuses the same
scikit-learn TF-IDF setup as ``similarity.py`` and the ``clean_text`` preprocessing
from ``utils.py``.

How it works
------------
Each category has a *profile*: a hand-written list of characteristic words and
short phrases (``TOPIC_PROFILES``). The article and the nine profiles are turned
into TF-IDF vectors; the category whose profile is closest (cosine similarity) wins.
TF-IDF down-weights words that appear in several profiles ("government", "market"),
so shared vocabulary does not decide the result.

Reliability guards
------------------
* ``confidence`` = the winner's share of the total similarity across all categories,
  scaled down when only a few distinct profile terms matched (a single keyword can
  never give a high-confidence answer).
* If the text shares too few terms with the winning profile (``MIN_MATCHED_TERMS``;
  just one for short headlines/claims), or the winner's share is below ``MIN_SHARE``,
  the topic is reported as ``General`` instead of guessing.
* English only (like the rest of the processing pipeline). Text in other scripts has
  no vocabulary overlap and therefore comes back as ``General``.

The category names match ``personalized_news.NEWS_CATEGORIES`` (minus "Other / Customize").
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import clean_text

GENERAL = "General"
MAX_CHARS = 4000        # a headline + opening paragraphs are enough to identify the topic
MIN_MATCHED_TERMS = 2   # distinct profile terms that must occur in a longer text
SHORT_TEXT_WORDS = 20   # headlines / one-line claims: a single matched term is enough
STRONG_MATCH = 4        # this many matched terms = full confidence (share is not reduced)
MIN_SHARE = 0.30        # winner must hold at least this share of the total similarity
MAX_ALTERNATIVES = 2

# Terms are matched after stop-word removal and plural folding, so "elections" matches
# "election". Keep terms to one or two content words (stop-words like "of" don't count).
TOPIC_PROFILES: dict[str, tuple[str, ...]] = {
    "Politics": (
        "government", "election", "vote", "voter", "ballot", "parliament", "assembly",
        "senate", "congress", "lawmaker", "legislation", "legislature", "president",
        "prime minister", "chief minister", "minister", "cabinet", "opposition",
        "coalition", "campaign", "candidate", "political party", "party leader", "policy",
        "democrat", "republican", "bjp", "lok sabha", "rajya sabha", "white house",
        "supreme court", "governor", "mayor", "referendum", "impeachment", "diplomat",
        "diplomatic", "ambassador", "embassy", "treaty", "sanction", "summit",
        "foreign policy", "manifesto", "rally", "constituency", "mla", "regime",
        "politician", "political", "politics", "parliamentary",
    ),
    "Sports": (
        "cricket", "football", "soccer", "tennis", "basketball", "baseball", "hockey",
        "badminton", "golf", "rugby", "olympics", "tournament", "championship", "league",
        "match", "team", "player", "coach", "goal", "wicket", "innings", "batsman",
        "bowler", "captain", "stadium", "medal", "semifinal", "final", "world cup",
        "fifa", "ipl", "nba", "nfl", "grand slam", "athlete", "referee", "season",
        "striker", "midfielder", "goalkeeper", "penalty", "umpire", "trophy", "fixture",
        "champion", "squad", "cup", "formula one", "marathon", "wrestling", "boxing",
    ),
    "Technology": (
        "technology", "tech", "software", "hardware", "app", "smartphone", "iphone",
        "android", "google", "microsoft", "apple", "openai", "chatgpt", "artificial intelligence",
        "machine learning", "chatbot", "algorithm", "cybersecurity", "hacker",
        "data breach", "cloud computing", "startup", "internet", "chip", "semiconductor",
        "processor", "robot", "robotics", "blockchain", "5g", "wifi", "laptop", "gadget",
        "developer", "open source", "tech giant", "social media", "platform",
        "quantum computing", "virtual reality", "internet of things", "encryption",
        "malware", "ransomware", "gpu", "nvidia", "samsung", "ai", "ai model",
        "large language", "generative ai", "drone",
    ),
    "Science": (
        "scientist", "research", "researcher", "study", "discovery", "experiment",
        "physics", "chemistry", "biology", "astronomy", "nasa", "isro", "space", "mission",
        "rocket", "satellite", "telescope", "planet", "galaxy", "moon", "lunar", "mars",
        "asteroid", "genome", "dna", "species", "fossil", "laboratory", "particle",
        "quantum", "astronaut", "orbit", "solar system", "black hole", "spacecraft",
        "chandrayaan", "space station", "big bang", "neuroscience", "brain", "evolution",
        "microscope", "journal", "scientific", "gravitational", "cosmic", "supernova",
        "nobel",
    ),
    "Business": (
        "economy", "economic", "market", "stock", "share price", "sensex", "nifty",
        "investor", "investment", "profit", "revenue", "earnings", "quarterly results",
        "company", "ceo", "merger", "acquisition", "bank", "inflation", "rate hike",
        "repo rate", "rbi", "federal reserve", "gdp", "trade", "tariff", "export",
        "import", "ipo", "shareholder", "currency", "rupee", "dollar", "crude oil",
        "oil price", "budget", "tax", "finance", "financial", "industry", "manufacturing",
        "consumer", "layoff", "unemployment", "wall street", "retail", "sales",
        "funding round", "venture capital", "billion", "corporate",
    ),
    "Health": (
        "health", "healthcare", "hospital", "doctor", "patient", "disease", "virus",
        "vaccine", "vaccination", "covid", "pandemic", "outbreak", "cancer", "diabetes",
        "treatment", "therapy", "medicine", "drug", "clinical trial", "symptom",
        "infection", "health organization", "surgery", "nutrition", "diet",
        "obesity", "mental health", "depression", "cdc", "fda", "public health",
        "epidemic", "medical", "medication", "nurse", "immunity", "heart attack",
        "blood pressure", "health ministry", "coronavirus", "flu", "dengue", "malaria",
        "tuberculosis", "antibiotic", "pharmaceutical", "wellness", "vitamin",
        "common cold", "side effect", "diagnosis", "stroke", "pregnancy",
    ),
    "Entertainment": (
        "film", "movie", "cinema", "bollywood", "hollywood", "actor", "actress",
        "director", "box office", "trailer", "album", "singer", "song", "music",
        "concert", "celebrity", "oscar", "grammy", "netflix", "web series", "television",
        "reality show", "streaming", "film festival", "red carpet", "theatre", "cast",
        "blockbuster", "sequel", "franchise", "comedian", "star", "premiere",
        "filmmaker", "screenplay", "soundtrack", "awards", "bigg boss", "ott", "disney",
        "tollywood", "talk show", "fans", "performance",
    ),
    "Education": (
        "education", "school", "student", "teacher", "university", "college", "exam",
        "examination", "admission", "curriculum", "syllabus", "scholarship", "degree",
        "campus", "cbse", "neet", "jee", "board exam", "ugc", "academic", "tuition",
        "literacy", "professor", "classroom", "entrance exam", "online learning",
        "education ministry", "dropout", "enrolment", "graduate", "vice chancellor",
        "nep", "iit", "kindergarten", "learning outcomes",
    ),
    "Environment": (
        "climate", "climate change", "global warming", "emission", "carbon", "pollution",
        "air quality", "renewable", "solar energy", "wind energy", "fossil fuel",
        "deforestation", "wildlife", "biodiversity", "conservation", 
         "glacier", "sea level", "plastic", "waste", "recycling",
        "forest", "ecosystem", "endangered", "greenhouse gas", "environmental",
        "sustainability", "net zero", "heatwave", "rainfall", "temperature", "cop",
        "tiger reserve", "national park", "ozone", "coral reef", "monsoon", "landfill",
        "clean energy", "environment ministry",
    ),

    "Disaster": (
        "disaster", "natural disaster", "flood", "flooding", "flash flood",
        "earthquake", "tsunami", "landslide", "cyclone", "hurricane",
        "tornado", "wildfire", "volcanic eruption", "volcano", "storm",
        "rescue team", "rescue operation", "evacuation", "evacuate",
        "displaced", "casualty", "casualties", "death toll",
        "devastated", "destroyed", "destroying", "destruction",
        "emergency response", "property damage", "homes destroyed",
        "buildings collapsed", "people missing", "heavy rainfall",
        "flood victims", "relief operation", "affected villages",
    ),
        "Crime & Law": (
        "crime", "criminal", "police", "arrest", "arrested", "murder", "homicide", "robbery",
        "theft", "burglary", "assault", "fraud", "scam", "kidnapping", "rape", "violence",
        "investigation", "suspect", "victim", "court", "lawsuit", "judge", "lawyer", "trial",
        "verdict", "sentence", "conviction", "prison", "jail", "bail", "prosecution", "prosecutor",
        "legal case", "criminal case", "cybercrime", "extortion", "corruption case", "charges filed",
        "law enforcement", "high court", "supreme court", "legal ruling", "court order",
    ),
    "International Affairs": (
        "international", "global affairs", "diplomacy", "diplomatic relations", "foreign relations",
        "bilateral relations", "summit", "embassy", "ambassador", "foreign minister", "treaty",
        "sanctions", "geopolitical", "geopolitics", "united nations", "un security council",
        "nato", "world leaders", "peace talks", "ceasefire", "international conflict", "border dispute",
        "trade agreement", "foreign policy", "global summit", "world bank", "imf", "g20", "g7",
        "refugee crisis", "international aid", "diplomatic mission", "cross border", "bilateral agreement",
    ),
    "Transportation": (
        "transportation", "transport", "railway", "train", "rail", "metro", "subway", "flight",
        "airline", "airport", "aviation", "aircraft", "plane", "helicopter", "road accident",
        "traffic accident", "car crash", "vehicle", "bus", "truck", "highway", "expressway",
        "traffic jam", "traffic congestion", "derailment", "train accident", "ship", "shipping",
        "port", "ferry", "bridge", "road safety", "transport ministry", "public transport",
        "passengers", "commuter", "railway station", "flight delay", "flight cancellation",
    ),
    "Agriculture": (
        "agriculture", "farmer", "farmers", "farming", "crop", "crops", "harvest", "harvesting",
        "irrigation", "fertilizer", "pesticide", "soil", "livestock", "dairy", "tractor", "rural economy",
        "agricultural", "agri", "monsoon crop", "kharif", "rabi", "wheat", "rice production",
        "sugarcane", "cotton", "food production", "food security", "minimum support price", "msp",
        "farm loan", "crop failure", "crop damage", "farmers protest", "agricultural market",
        "mandi", "procurement", "seed", "pest attack", "drought impact", "farm equipment",
    ),
    "Defence & Security": (
        "defence", "defense", "military", "armed forces", "army", "navy", "air force", "soldier",
        "troops", "border security", "border patrol", "military operation", "defence operation",
        "national security", "security forces", "armed conflict", "missile", "fighter jet", "warship",
        "military aircraft", "defence minister", "defense ministry", "weapons system", "military base",
        "army chief", "naval exercise", "military exercise", "counterterrorism", "terror attack",
        "terrorism", "insurgent", "ceasefire violation", "border clash", "military deployment",
        "defence deal", "defense deal", "security threat", "intelligence agency",
    ),

}

# ---------------------------------------------------------------- text analysis
# Same tokenisation, English stop-words and 1-2-gram setup as similarity.py.
_BASE_ANALYZER = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), strip_accents="unicode"
).build_analyzer()
_SEGMENT = re.compile(r"[|;.!?\n]+")  # never build a bigram across a sentence / term boundary


def _fold(word: str) -> str:
    """Fold simple plurals so "elections" == "election" (applied to terms and text alike)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _analyze(text: str) -> list[str]:
    features: list[str] = []
    for segment in _SEGMENT.split(text):
        for token in _BASE_ANALYZER(segment):
            features.append(" ".join(_fold(w) for w in token.split()))
    return features


def _term_key(term: str) -> str | None:
    """The single feature that represents a profile term (its bigram if it has two words)."""
    features = _analyze(term)
    return features[-1] if features else None


def _profile_keys() -> dict[str, set[str]]:
    return {c: {k for k in map(_term_key, terms) if k} for c, terms in TOPIC_PROFILES.items()}


_PROFILE_KEYS = _profile_keys()
_CATEGORIES = list(TOPIC_PROFILES)
_PROFILE_DOCS = ["|".join(TOPIC_PROFILES[c]) for c in _CATEGORIES]


def _general(note: str = "") -> dict:
    return {"label": GENERAL, "confidence": 0.0, "alternatives": [], "note": note}


def detect_topic(text: str) -> dict:
    """Classify ``text`` into a news category.

    Returns ``{"label", "confidence", "alternatives", "note"}`` where ``confidence`` is
    0-1, ``alternatives`` lists up to two runner-up categories and ``note`` is non-empty
    only when the result is ``General``. Raises ``ValueError`` for empty text (same
    contract as ``analyze_sentiment``).
    """
    text = clean_text(text)
    if not text:
        raise ValueError("Text cannot be empty.")
    text = text[:MAX_CHARS]

    # The text goes into the TF-IDF model too, so its unmatched words count towards its
    # length and a single stray keyword in a long article cannot dominate.
    vectorizer = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(_PROFILE_DOCS + [text])
    except ValueError:  # nothing usable (e.g. only stop-words or symbols)
        return _general("Not enough text to detect a topic.")
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    total = float(scores.sum())
    if total <= 0:
        return _general("No clear topic signal (topic detection works on English text).")

    features = set(_analyze(text))
    ranked = sorted(range(len(_CATEGORIES)), key=lambda i: scores[i], reverse=True)
    best = ranked[0]
    matched = len(_PROFILE_KEYS[_CATEGORIES[best]] & features)
    share = float(scores[best]) / total
    needed = 1 if len(text.split()) <= SHORT_TEXT_WORDS else MIN_MATCHED_TERMS
    if matched < needed or share < MIN_SHARE:
        return _general("No clear topic signal (topic detection works on English text).")
    confidence = share * min(1.0, matched / STRONG_MATCH)

    alternatives = [
        {"label": _CATEGORIES[i], "confidence": round(float(scores[i]) / total, 3)}
        for i in ranked[1:1 + MAX_ALTERNATIVES] if scores[i] > 0
    ]
    return {"label": _CATEGORIES[best], "confidence": round(confidence, 3),
            "alternatives": alternatives, "note": ""}