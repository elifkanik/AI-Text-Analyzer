"""
app.py
------
AI Text Analyzer — Streamlit web application
Dual-engine sentiment analysis (VADER + TextBlob) with
word frequency statistics. All output is table & metric-based.
"""

import streamlit as st
from analyzer.word_frequency import basic_stats, word_frequency
from analyzer.sentiment import analyze_vader, analyze_textblob, analyze_sentences, compare_engines
from analyzer.formatter import (
    label_badge,
    vader_metrics,
    textblob_metrics,
    format_sentence_df,
    format_freq_df,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Text Analyzer",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.4);
    }
    .main-header h1 {
        color: #e0e7ff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #93c5fd;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        padding: 0.4rem 0;
        border-left: 4px solid #6366f1;
        padding-left: 0.75rem;
        margin-bottom: 0.75rem;
    }

    /* Stat card override */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Badge */
    .sentiment-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        margin: 0.25rem 0 1rem 0;
    }
    .badge-positive { background: #dcfce7; color: #166534; }
    .badge-negative { background: #fee2e2; color: #991b1b; }
    .badge-neutral  { background: #f1f5f9; color: #475569; }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 2rem 0;
    }

    /* Dataframe tweaks */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Analyze button */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: opacity 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        opacity: 0.88;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 AI Text Analyzer</h1>
        <p>Analyze. Understand. Visualize. — Powered by VADER &amp; TextBlob</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
text_input = st.text_area(
    label="Enter your text below",
    placeholder="Paste or type any English text here…",
    height=200,
    key="text_input",
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    analyze_clicked = st.button("🔍 Analyze", key="analyze_btn")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    raw = text_input.strip()

    if not raw:
        st.warning("⚠️ Please enter some text before analyzing.")
        st.stop()

    if len(raw.split()) < 3:
        st.warning("⚠️ Please enter at least a few words for a meaningful analysis.")
        st.stop()

    # Run all analyses
    with st.spinner("Analyzing…"):
        stats      = basic_stats(raw)
        freq_df    = word_frequency(raw, top_n=15)
        vader      = analyze_vader(raw)
        tb         = analyze_textblob(raw)
        sent_df    = analyze_sentences(raw)
        compare_df = compare_engines(vader, tb)

    # -----------------------------------------------------------------------
    # Section 1 — General Statistics
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">📊 General Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Word Count",         stats["word_count"],
              help="Total alphabetic tokens (stop words included).")
    c2.metric("Sentence Count",     stats["sentence_count"])
    c3.metric("Unique Words",       stats["unique_word_count"],
              help="Unique lemmatized words, stop words excluded.")
    c4.metric("Avg Word Length",    f"{stats['avg_word_length']} ch",
              help="Average character length of clean (non-stop) words.")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 2 — Word Frequency
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">🔡 Word Frequency (Top 15)</div>', unsafe_allow_html=True)
    st.caption("Stop words removed · Lemmatized · Sorted by frequency")

    if freq_df.empty:
        st.info("Not enough content-bearing words to display a frequency table.")
    else:
        st.dataframe(format_freq_df(freq_df), width='stretch')

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 3 — Sentiment Analysis
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)

    col_v, col_tb = st.columns(2, gap="large")

    # --- VADER ---
    with col_v:
        st.markdown("**🔵 VADER**")
        st.caption("Rule-based · Optimized for real-world text · Compound score drives the label")

        badge_class = f"badge-{vader['label'].lower()}"
        st.markdown(
            f'<span class="sentiment-badge {badge_class}">{label_badge(vader["label"])}</span>',
            unsafe_allow_html=True,
        )

        v_cols = st.columns(4)
        for col, (lbl, val, tip) in zip(v_cols, vader_metrics(vader)):
            col.metric(lbl, val, help=tip)

    # --- TextBlob ---
    with col_tb:
        st.markdown("**🟣 TextBlob**")
        st.caption("Pattern-based · Adds subjectivity dimension · Works on sentence patterns")

        badge_class = f"badge-{tb['label'].lower()}"
        st.markdown(
            f'<span class="sentiment-badge {badge_class}">{label_badge(tb["label"])}</span>',
            unsafe_allow_html=True,
        )
        sub_label = tb["subjectivity_label"]
        st.markdown(
            f'<span class="sentiment-badge badge-neutral" style="font-size:0.85rem;">'
            f'📝 {sub_label}</span>',
            unsafe_allow_html=True,
        )

        tb_cols = st.columns(2)
        for col, (lbl, val, tip) in zip(tb_cols, textblob_metrics(tb)):
            col.metric(lbl, val, help=tip)

    # --- Comparison Table ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**⚖️ Engine Comparison**")
    st.dataframe(compare_df, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 4 — Sentence-Level Sentiment
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-title">📋 Sentence-Level Sentiment (VADER)</div>', unsafe_allow_html=True)
    st.caption("Each sentence analyzed individually · Compound score: −1 (most negative) → +1 (most positive)")

    if sent_df.empty:
        st.info("Could not detect individual sentences.")
    else:
        st.dataframe(format_sentence_df(sent_df), width='stretch', height=350)

elif not analyze_clicked:
    st.markdown(
        """
        <div style="text-align:center; color:#94a3b8; padding: 3rem 0; font-size:1rem;">
            ↑ Enter some English text above and click <strong>Analyze</strong> to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )
