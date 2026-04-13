"""
formatter.py
------------
Helpers that turn raw analysis results into Streamlit-ready
metric values and plain DataFrames (no Styler / no matplotlib).
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

LABEL_ICONS = {
    "Positive": "🟢",
    "Negative": "🔴",
    "Neutral":  "⚪",
}


def label_badge(label: str) -> str:
    """Return an emoji-prefixed label string."""
    icon = LABEL_ICONS.get(label, "❓")
    return f"{icon} {label}"


# ---------------------------------------------------------------------------
# VADER metric dict  →  list of (label, value, help_text) tuples
# ---------------------------------------------------------------------------

def vader_metrics(vader: dict) -> list[tuple]:
    """
    Return a list of (metric_label, value, help_text) tuples
    suitable for st.metric().
    """
    return [
        ("Compound", vader["compound"],
         "Overall sentiment score. Range: −1 (most negative) to +1 (most positive). "
         "|compound| < 0.05 → Neutral."),
        ("Positive", vader["pos"],
         "Proportion of text that is positive (0–1)."),
        ("Neutral", vader["neu"],
         "Proportion of text that is neutral (0–1)."),
        ("Negative", vader["neg"],
         "Proportion of text that is negative (0–1)."),
    ]


# ---------------------------------------------------------------------------
# TextBlob metric list
# ---------------------------------------------------------------------------

def textblob_metrics(tb: dict) -> list[tuple]:
    return [
        ("Polarity", tb["polarity"],
         "Sentiment polarity. Range: −1 (very negative) to +1 (very positive)."),
        ("Subjectivity", tb["subjectivity"],
         "0 = fully objective/factual, 1 = fully subjective/opinionated."),
    ]


# ---------------------------------------------------------------------------
# Sentence-level DataFrame — adds a Sentiment indicator column
# ---------------------------------------------------------------------------

def format_sentence_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an emoji Sentiment column for quick visual scanning.
    Returns a plain DataFrame (no Styler).
    """
    out = df.copy()
    out["Sentiment"] = out["Label"].map(LABEL_ICONS).fillna("❓")
    # Reorder columns
    out = out[["Sentence", "Compound Score", "Sentiment", "Label"]]
    return out


# ---------------------------------------------------------------------------
# Word frequency DataFrame — plain, no styling
# ---------------------------------------------------------------------------

def format_freq_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return the frequency DataFrame as-is (plain, no Styler)."""
    return df.copy()

