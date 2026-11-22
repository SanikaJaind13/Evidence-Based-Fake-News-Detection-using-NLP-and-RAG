
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
def download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


download_nltk_resources()


def preprocess_text(text):
    """
    Clean and preprocess text using NLP techniques.
    """

    if not text or not isinstance(text, str):
        return {
            "cleaned_text": "",
            "tokens": [],
            "filtered_tokens": [],
            "lemmatized_tokens": [],
            "processed_text": ""
        }

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Preserve important numerical formats
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    # 4. Remove unwanted special characters while preserving
    # numbers, decimals, percentages and currency symbols
    text = re.sub(r"[^a-zA-Z0-9\s.%₹$€£+-]", " ", text)

    # 5. Normalize extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Tokenization
        # Tokenization: preserve decimals, percentages, currency values
    # and hyphenated words
    tokens = re.findall(
        r"[₹$€£]?\d+(?:\.\d+)?%?|[a-zA-Z]+(?:-[a-zA-Z0-9]+)*|[a-zA-Z0-9]+",
        text
    )

    # 4. Stopword removal
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
    lemmatizer.lemmatize(word, pos="v")
    for word in filtered_tokens
]

    return {
        "cleaned_text": text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "lemmatized_tokens": lemmatized_tokens,
        "processed_text": " ".join(lemmatized_tokens)
    }
