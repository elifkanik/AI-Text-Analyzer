"""
sentiment.py
------------
Dual-engine sentiment analysis:
  - VADER  (vaderSentiment) — lexicon + rule-based, great for short/social text
  - TextBlob               — pattern-based, adds subjectivity score
  - Sentence-level VADER   — per-sentence compound scores
"""

import nltk
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd


def _ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


_ensure_nltk_data()

_vader = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    return "Neutral"


def _textblob_label(polarity: float) -> str:
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    return "Neutral"


def _subjectivity_label(subjectivity: float) -> str:
    if subjectivity >= 0.6:
        return "Highly Subjective"
    elif subjectivity >= 0.4:
        return "Moderately Subjective"
    return "Objective"


# ---------------------------------------------------------------------------
# VADER analysis
# ---------------------------------------------------------------------------

def analyze_vader(text: str) -> dict:
    """
    Run VADER on the full text.

    Returns
    -------
    dict with keys:
        compound, pos, neu, neg  – raw scores (floats)
        label                    – 'Positive' | 'Negative' | 'Neutral'
    """
    scores = _vader.polarity_scores(text)
    return {
        "compound": round(scores["compound"], 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
        "label": _vader_label(scores["compound"]),
    }


# ---------------------------------------------------------------------------
# TextBlob analysis
# ---------------------------------------------------------------------------

def analyze_textblob(text: str) -> dict:
    """
    Run TextBlob on the full text.

    Returns
    -------
    dict with keys:
        polarity        – float in [-1, 1]
        subjectivity    – float in [0, 1]
        label           – 'Positive' | 'Negative' | 'Neutral'
        subjectivity_label – human-readable subjectivity interpretation
    """
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "label": _textblob_label(polarity),
        "subjectivity_label": _subjectivity_label(subjectivity),
    }


# ---------------------------------------------------------------------------
# Sentence-level analysis
# ---------------------------------------------------------------------------

def analyze_sentences(text: str) -> pd.DataFrame:
    """
    Run VADER on each sentence individually.

    Returns
    -------
    pd.DataFrame with columns:
        #, Sentence, Compound Score, Label
    """
    sentences = sent_tokenize(text)
    rows = []
    for i, sent in enumerate(sentences, start=1):
        scores = _vader.polarity_scores(sent)
        compound = round(scores["compound"], 4)
        rows.append({
            "#": i,
            "Sentence": sent,
            "Compound Score": compound,
            "Label": _vader_label(compound),
        })
    return pd.DataFrame(rows).set_index("#")


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_engines(vader_result: dict, textblob_result: dict) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of the two engines.

    Returns
    -------
    pd.DataFrame with columns: Engine, Label, Key Score, Agreement
    """
    agree = vader_result["label"] == textblob_result["label"]
    agreement_str = "✅ Agree" if agree else "⚠️ Disagree"

    rows = [
        {
            "Engine": "VADER",
            "Label": vader_result["label"],
            "Key Score": f"compound = {vader_result['compound']}",
            "Agreement": agreement_str,
        },
        {
            "Engine": "TextBlob",
            "Label": textblob_result["label"],
            "Key Score": f"polarity = {textblob_result['polarity']}",
            "Agreement": agreement_str,
        },
    ]
    return pd.DataFrame(rows).set_index("Engine")
