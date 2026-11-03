"""
features.py
-----------
Shared feature-engineering code used by both train.py and app.py.
Keeping it in a separate module lets the pickled pipeline be loaded
safely inside the Streamlit app.

The model analyses ONLY the customer's (client's) response. The agent's
(caller's) lines are scripted and always upbeat, so they add noise and
let filler words skew predictions.
"""

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TEXT_COL = "customer_response"   # client speech - the only text the model sees
AGENT_COL = "agent_lines"        # caller speech - kept for display only
META_COLS = ["call_time", "customer_segment"]
SENT_COL = "vader_compound"

CALL_TIMES = ["Morning", "Afternoon", "Evening"]
SEGMENTS = ["Salaried", "Self-Employed", "Student"]

# Filler / function words that carry no intent but can pick up spurious
# weight (e.g. "am" from "I am in a meeting").  Negations such as "not",
# "no", "don't" are deliberately NOT included - they are strong signals.
STOP_WORDS = [
    "am", "i", "is", "are", "was", "be", "a", "an", "the", "me", "my", "you",
    "your", "it", "this", "that", "to", "of", "on", "in", "for", "and", "or",
    "so", "will", "can", "do", "just", "ji", "haan", "achha", "okay", "ok",
    "sir", "ma'am", "madam", "please", "right", "now", "one", "these", "there",
    "what", "how", "which", "from", "at", "with", "have", "has", "abhi",
]

_analyzer = SentimentIntensityAnalyzer()


def vader_score(text: str) -> float:
    """VADER compound sentiment score (-1 .. +1) of the customer's response."""
    return float(_analyzer.polarity_scores(str(text))["compound"])


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SENT_COL] = df[TEXT_COL].fillna("").apply(vader_score)
    return df


def build_preprocessor() -> ColumnTransformer:
    """TF-IDF on customer response (1-2 grams, max 300) + VADER + one-hot metadata."""
    return ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=300,
                                      lowercase=True, sublinear_tf=True,
                                      stop_words=STOP_WORDS,
                                      token_pattern=r"(?u)\b[a-zA-Z']{2,}\b"),
             TEXT_COL),
            ("sentiment", StandardScaler(), [SENT_COL]),
            ("meta", OneHotEncoder(categories=[CALL_TIMES, SEGMENTS],
                                   handle_unknown="ignore"), META_COLS),
        ],
        sparse_threshold=0.0,  # dense output: small data, works with every model + SHAP
    )


def _split_meta(n: str):
    raw = n[len("meta__"):]
    for col in META_COLS:
        if raw.startswith(col + "_"):
            return col, raw[len(col) + 1:]
    return "meta", raw


def readable_feature_names(preprocessor: ColumnTransformer) -> list:
    names = []
    for n in preprocessor.get_feature_names_out():
        if n.startswith("tfidf__"):
            names.append(f'word: "{n[len("tfidf__"):]}"')
        elif n.startswith("sentiment__"):
            names.append("VADER sentiment")
        elif n.startswith("meta__"):
            col, val = _split_meta(n)
            names.append(f"{col} = {val}")
        else:
            names.append(n)
    return names


def make_input(customer_response: str, call_time: str, segment: str) -> pd.DataFrame:
    df = pd.DataFrame([{TEXT_COL: customer_response, "call_time": call_time,
                        "customer_segment": segment}])
    return add_sentiment(df)


def word_impact(pipeline, customer_response: str, call_time: str, segment: str) -> pd.DataFrame:
    """Model-agnostic leave-one-word-out analysis on the customer's response.

    Each distinct (non-filler) word is removed and the response re-scored;
    the change in probability is that word's impact. Positive = the word
    increased the conversion probability."""
    base = float(pipeline.predict_proba(make_input(customer_response, call_time, segment))[0, 1])

    stop = set(STOP_WORDS)
    vocab = []
    for t in re.findall(r"[A-Za-z']+", customer_response):
        tl = t.lower()
        if len(tl) > 1 and tl not in stop and tl not in vocab:
            vocab.append(tl)
    if not vocab:
        return pd.DataFrame(columns=["word", "impact"])

    variants = [re.sub(rf"(?<![A-Za-z']){re.escape(w)}(?![A-Za-z'])", "",
                       customer_response, flags=re.IGNORECASE) for w in vocab]
    batch = add_sentiment(pd.DataFrame({TEXT_COL: variants,
                                        "call_time": call_time,
                                        "customer_segment": segment}))
    probs = pipeline.predict_proba(batch)[:, 1]
    out = pd.DataFrame({"word": vocab, "impact": base - probs})
    out = out[np.abs(out["impact"]) > 1e-4]
    return out.sort_values("impact", key=np.abs, ascending=False).reset_index(drop=True)
