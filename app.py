import re
import string
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import streamlit as st
import nltk

# ---------------------------------------------------------
# Ensure NLTK Lexicon & Tokenizer Resources are Available
# ---------------------------------------------------------
for resource in ['vader_lexicon', 'punkt', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Disaster Tweet Predictor",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Modern, Clean & Simple Styling
# ---------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Card */
    .app-header {
        text-align: center;
        padding: 20px 10px;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #64748B;
        max-width: 550px;
        margin: 0 auto;
    }

    /* Disaster Alert Box */
    .result-disaster {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 8px 20px -4px rgba(239, 68, 68, 0.2);
    }
    .result-disaster-title {
        color: #DC2626;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    /* Safe / Non-Disaster Box */
    .result-safe {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.2);
    }
    .result-safe-title {
        color: #059669;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .result-desc {
        color: #475569;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Input area styling */
    .stTextArea textarea {
        font-size: 1.05rem;
        border-radius: 12px;
        padding: 14px;
        border: 1.5px solid #CBD5E1;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }

    /* Primary button */
    .stButton button[kind="primary"] {
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load NLTK Sentiment Intensity Analyzer
# ---------------------------------------------------------
@st.cache_resource
def get_sentiment_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        class FallbackAnalyzer:
            def polarity_scores(self, text):
                return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0}
        return FallbackAnalyzer()

analyzer = get_sentiment_analyzer()

# ---------------------------------------------------------
# Load ONLY PKL Files
# ---------------------------------------------------------
@st.cache_resource
def load_pkl_artifacts():
    try:
        model = joblib.load("final_model.pkl")
        tfidf = joblib.load("final_tfidf_vectorizer.pkl")
        scaler = joblib.load("final_feature_scaler.pkl")
        return model, tfidf, scaler, None
    except Exception as e:
        return None, None, None, str(e)

model, tfidf, scaler, load_error = load_pkl_artifacts()

# ---------------------------------------------------------
# Safe Text Cleaning & Feature Extraction
# ---------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")

def clean_text(text: str) -> str:
    """Cleans tweet text matching the training pipeline."""
    if text is None:
        return ""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_meta_features(raw_text: str) -> pd.DataFrame:
    """Extracts engineered metadata features safely."""
    raw_text = str(raw_text) if raw_text is not None else ""
    try:
        sentiment_score = float(analyzer.polarity_scores(raw_text)["compound"])
    except Exception:
        sentiment_score = 0.0
        
    return pd.DataFrame([{
        "tweet_length": len(raw_text),
        "word_count": len(raw_text.split()),
        "has_hashtag": int(bool(re.search(r"#\w+", raw_text))),
        "has_mention": int(bool(re.search(r"@\w+", raw_text))),
        "sentiment": sentiment_score,
    }])

def predict_sentence(raw_text: str):
    """Predicts whether sentence is a disaster or not using .pkl files."""
    cleaned = clean_text(raw_text)
    
    try:
        tfidf_feat = tfidf.transform([cleaned])
    except Exception:
        tfidf_feat = csr_matrix((1, len(tfidf.vocabulary_)))
        
    extra_df = extract_meta_features(raw_text)
    extra_feat = scaler.transform(extra_df)
    
    # Combined features
    X = hstack([tfidf_feat, csr_matrix(extra_feat)]).tocsr()
    
    pred = model.predict(X)[0]
    
    if hasattr(model, "predict_proba"):
        disaster_prob = float(model.predict_proba(X)[0][1])
    else:
        score = float(model.decision_function(X)[0])
        disaster_prob = float(1 / (1 + np.exp(-score)))
        
    return {
        "is_disaster": bool(pred == 1),
        "disaster_prob": disaster_prob,
        "safe_prob": 1.0 - disaster_prob
    }

# ---------------------------------------------------------
# Main UI Header
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🚨 Disaster Tweet Predictor</div>
    <div class="app-subtitle">
        Enter any sentence or tweet to instantly check if it reports a real disaster/emergency or not.
    </div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"❌ Error loading model `.pkl` files: `{load_error}`")
    st.stop()

# ---------------------------------------------------------
# Quick Preset Examples (Optional 1-Click test)
# ---------------------------------------------------------
# Single source of truth for the text box's content. The text_area below is
# bound with key="tweet_input", so this session_state key IS the widget's
# value.
#
# IMPORTANT: we change st.session_state.tweet_input ONLY from inside
# on_click callbacks (set_preset / clear_input), never in the main body of
# the script. Streamlit forbids writing to a widget's own session_state key
# after that widget has already been drawn in the current run -- and Clear
# sits below the text_area in the layout, so a plain `if st.button(...):`
# handler was writing to it too late and got silently ignored. Callbacks run
# BEFORE the script reruns and redraws widgets, so they can always update
# the value safely, regardless of where the button sits on the page.
if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""

def set_preset(text: str):
    st.session_state.tweet_input = text

def clear_input():
    st.session_state.tweet_input = ""

st.caption("💡 Quick test samples:")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

with col_s1:
    st.button(
        "🔥 Wildfire Emergency",
        use_container_width=True,
        on_click=set_preset,
        args=("Massive wildfire spreading fast near the highway, evacuations ordered immediately!",)
    )
with col_s2:
    st.button(
        "🌊 Flash Flood Warning",
        use_container_width=True,
        on_click=set_preset,
        args=("Flash flood warning in effect for low lying areas after heavy storm!",)
    )
with col_s3:
    st.button(
        "☀️ Sunny Day (Safe)",
        use_container_width=True,
        on_click=set_preset,
        args=("Enjoying a lovely sunny afternoon in the park with my friends and ice cream.",)
    )
with col_s4:
    st.button(
        "🎬 Movie Night (Safe)",
        use_container_width=True,
        on_click=set_preset,
        args=("Watched an awesome movie with family tonight, had such a blast!",)
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Custom Sentence Input
# ---------------------------------------------------------
# Bound directly via key="tweet_input" and no `value=` param, so this one
# key is always the single source of truth -- whether it got its content
# from typing, a preset click, or Clear.
user_input = st.text_area(
    "Enter your sentence or tweet:",
    key="tweet_input",
    placeholder="Type any sentence here (e.g., 'Huge fire broke out in the building downtown')...",
    height=120
)

col_btn1, col_btn2 = st.columns([1.5, 4])
with col_btn1:
    predict_clicked = st.button("🔍 Predict Disaster", type="primary", use_container_width=True)
with col_btn2:
    st.button("🧹 Clear", use_container_width=False, on_click=clear_input)

# ---------------------------------------------------------
# Prediction Result Display
# ---------------------------------------------------------
if predict_clicked or user_input.strip():
    if not user_input.strip():
        st.warning("⚠️ Please type or paste a sentence above to predict.")
    else:
        result = predict_sentence(user_input)
        
        if result["is_disaster"]:
            st.markdown(f"""
            <div class="result-disaster">
                <div style="font-size: 3rem; margin-bottom: 6px;">🚨</div>
                <div class="result-disaster-title">DISASTER TWEET</div>
                <p class="result-desc">This sentence indicates a real emergency, accident, or natural disaster.</p>
                <div style="margin-top: 12px; font-size: 1.1rem; font-weight: 700; color: #DC2626;">
                    Confidence: {result['disaster_prob'] * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size: 3rem; margin-bottom: 6px;">✅</div>
                <div class="result-safe-title">NOT A DISASTER</div>
                <p class="result-desc">This sentence is safe / non-disaster (casual post, news, or conversation).</p>
                <div style="margin-top: 12px; font-size: 1.1rem; font-weight: 700; color: #059669;">
                    Confidence: {result['safe_prob'] * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"**Disaster Probability:** `{result['disaster_prob'] * 100:.1f}%`")
        st.progress(result["disaster_prob"])