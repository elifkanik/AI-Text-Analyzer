"""
word_frequency.py
-----------------
Tokenization → Stop word removal → Lemmatization → Frequency count
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
import pandas as pd


def _ensure_nltk_data():
    """Download required NLTK resources if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def basic_stats(text: str) -> dict:
    """
    Compute basic text statistics.

    Returns
    -------
    dict with keys:
        word_count        – total tokens (including stop words & punctuation)
        sentence_count    – number of sentences
        unique_word_count – unique lemmatized non-stop words
        avg_word_length   – average character length of clean words
    """
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)

    # Clean tokens: lowercase, no punctuation
    clean = [
        t.lower() for t in tokens
        if t.isalpha() and t.lower() not in _stop_words
    ]
    lemmas = [_lemmatizer.lemmatize(w) for w in clean]

    avg_len = round(sum(len(w) for w in clean) / len(clean), 2) if clean else 0.0

    return {
        "word_count": len([t for t in tokens if t.isalpha()]),
        "sentence_count": len(sentences),
        "unique_word_count": len(set(lemmas)),
        "avg_word_length": avg_len,
    }


def word_frequency(text: str, top_n: int = 15) -> pd.DataFrame:
    """
    Compute word frequency after preprocessing.

    Parameters
    ----------
    text  : raw input text
    top_n : number of top words to return

    Returns
    -------
    pd.DataFrame with columns: Word, Frequency, Percentage
    """
    tokens = word_tokenize(text)

    # Lowercase, alpha only, remove stop words
    clean = [
        t.lower() for t in tokens
        if t.isalpha() and t.lower() not in _stop_words
    ]

    # Lemmatize
    lemmas = [_lemmatizer.lemmatize(w) for w in clean]

    if not lemmas:
        return pd.DataFrame(columns=["Word", "Frequency", "Percentage"])

    counts = Counter(lemmas)
    total = sum(counts.values())
    top = counts.most_common(top_n)

    df = pd.DataFrame(top, columns=["Word", "Frequency"])
    df["Percentage"] = (df["Frequency"] / total * 100).round(2).astype(str) + "%"
    df.index = df.index + 1  # 1-based rank
    df.index.name = "Rank"
    return df
