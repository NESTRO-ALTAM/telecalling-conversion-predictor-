"""
app.py
------
Streamlit dashboard for the Telecalling Conversion Predictor.

Run with:
    .\\venv\\Scripts\\streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from features import CALL_TIMES, SEGMENTS, make_input, vader_score, word_impact

BASE = Path(__file__).parent
st.set_page_config(page_title="Telecalling Conversion Predictor",
                   page_icon="📞", layout="wide")

# Each sample: 6 alternating turns -> Caller, Client, Caller, Client, Caller, Client
SAMPLES = {
    "Positive - Credit card": (
        "Good morning sir, I am Rahul from Sahyog Bank. You are pre-approved for our Platinum credit card with lifetime free membership.",
        "Yes, that sounds great! I was looking for this only.",
        "Wonderful sir, it also gives 5 percent cashback on online shopping.",
        "Perfect, how do I apply?",
        "I will just verify a few details and complete the application.",
        "Sure, go ahead."),
    "Curious - Annual fee question": (
        "Hello ma'am, this is Priya from Bharat FinServ. We have a special cashback credit card offer selected only for you.",
        "Is there an annual fee for this card?",
        "No ma'am, the joining fee and annual fee are both waived for you.",
        "Okay, and what is the credit limit?",
        "Your limit will be around 2 lakh.",
        "Sounds good, tell me the process."),
    "Curious - Personal loan rate": (
        "Hello Mr. Verma, I am Ankit from Trust Capital. You have a pre-approved personal loan of up to 5 lakh.",
        "What is the interest rate?",
        "The rate starts at 10.5 percent and the money comes in 24 hours.",
        "What documents are required?",
        "Only PAN, Aadhaar and three months salary slip.",
        "Okay, that is easy."),
    "Objection - In a meeting": (
        "Good afternoon Mr. Reddy, Vikram calling from Unity Finance. We can disburse an instant personal loan within 24 hours.",
        "I am in a meeting right now, call me later.",
        "Sure sir, what time should I call you back?",
        "I don't know, I am busy all day.",
        "No problem, I will try in the evening.",
        "Okay."),
    "Objection - Send on WhatsApp": (
        "Hello Mrs. Gupta, I am Sneha from Suraksha Life. We have a family health insurance plan with 10 lakh cover.",
        "Just send the details on WhatsApp, I will check.",
        "Sure ma'am, it covers your parents also.",
        "Send me the brochure, I will see later.",
        "I am sending the details on WhatsApp right now.",
        "Okay."),
    "Rejection - Not interested": (
        "Hi sir, this is Karan from Navyug Bank. Our term plan also gives tax saving under section 80C.",
        "Not interested.",
        "Sir, it is just 800 rupees per month for 1 crore cover.",
        "No, I am not interested, please don't call again.",
        "Sorry for the disturbance sir.",
        "Remove my number from your list."),
}


@st.cache_resource
def load_artifacts():
    with open(BASE / "model_artifacts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    with open(BASE / "metrics.json") as f:
        return json.load(f)


def priority_badge(prob: float) -> str:
    if prob >= 0.65:
        label, color, tip = "HIGH PRIORITY", "#16a34a", "Hot lead - push for closure on this call."
    elif prob >= 0.35:
        label, color, tip = "MEDIUM PRIORITY", "#f59e0b", "Warm lead - answer questions, schedule a follow-up."
    else:
        label, color, tip = "LOW PRIORITY", "#dc2626", "Cold lead - log it and move to the next call."
    return (f"<div style='display:inline-block;padding:10px 22px;border-radius:999px;"
            f"background:{color};color:white;font-weight:700;font-size:1.1rem'>{label}</div>"
            f"<p style='margin-top:8px;color:#555'>{tip}</p>")


def impact_chart(impacts: pd.DataFrame):
    top = impacts.head(12).iloc[::-1]
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in top["impact"]]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.38 * len(top))))
    ax.barh(top["word"], top["impact"] * 100, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in conversion probability (percentage points)")
    ax.set_title("Word impact (green = increases, red = decreases)")
    fig.tight_layout()
    return fig


def highlight_text(transcript: str, impacts: pd.DataFrame) -> str:
    import html
    import re
    lookup = dict(zip(impacts["word"], impacts["impact"]))
    max_abs = max([abs(v) for v in lookup.values()] + [1e-9])
    out_lines = []
    for line in transcript.splitlines():
        parts = re.split(r"([A-Za-z']+)", line)
        buf = []
        for p in parts:
            v = lookup.get(p.lower())
            if v is None or abs(v) < 0.005:
                buf.append(html.escape(p))
            else:
                alpha = 0.2 + 0.6 * abs(v) / max_abs
                rgb = "22,163,74" if v > 0 else "220,38,38"
                buf.append(f"<span style='background:rgba({rgb},{alpha:.2f});"
                           f"padding:1px 3px;border-radius:4px'>{html.escape(p)}</span>")
        out_lines.append("".join(buf))
    return ("<div style='line-height:2;font-size:1rem;border:1px solid #ddd;"
            "border-radius:8px;padding:12px'>" + "<br>".join(out_lines) + "</div>")


# ---------------------------------------------------------------------------
st.title("📞 Telecalling Conversion Predictor")
st.caption("Predicts the probability that a telemarketing call converts, "
           "by analysing only the customer's response + call metadata.")

missing = [p for p in ("model_artifacts.pkl", "metrics.json") if not (BASE / p).exists()]
if missing:
    st.error(f"Missing {', '.join(missing)}. Run `python generate_data.py` and "
             f"`python train.py` first.")
    st.stop()

art = load_artifacts()
metrics = load_metrics()

tab1, tab2 = st.tabs(["🎯 Live Call Predictor", "📊 Algorithm Comparison & Metrics"])

# ---------------------------- TAB 1 ----------------------------------------
with tab1:
    model_names = list(art["all_pipelines"].keys())
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Call details")
        c1, c2 = st.columns(2)
        call_time = c1.selectbox("Call Time", CALL_TIMES, index=2)
        segment = c2.selectbox("Customer Segment", SEGMENTS, index=0)
        sample = st.selectbox("Load a sample transcript", list(SAMPLES.keys()))
        st.markdown("**Conversation** (caller and client alternate - only the "
                    "🙋 client lines are analysed by the model)")
        turns = SAMPLES[sample]
        lines = []
        for i in range(6):
            is_caller = i % 2 == 0
            label = (f"📢 Caller - line {i // 2 + 1}" if is_caller
                     else f"🙋 Client - line {i // 2 + 1}")
            lines.append(st.text_input(label, value=turns[i], key=f"turn_{sample}_{i}"))
        transcript = " ".join(l.strip() for l in lines[1::2] if l.strip())
        model_name = st.selectbox(
            "Model", model_names, index=model_names.index(art["best_model_name"]),
            help="Defaults to the best model selected during training.")
        predict = st.button("Predict Conversion Probability", type="primary",
                            width="stretch")

    with right:
        st.subheader("Prediction")
        if predict:
            if not transcript.strip():
                st.warning("Please enter the customer's response.")
            else:
                pipe = art["all_pipelines"][model_name]
                X = make_input(transcript, call_time, segment)
                prob = float(pipe.predict_proba(X)[0, 1])
                sent = vader_score(transcript)

                st.markdown(
                    f"<div style='font-size:4rem;font-weight:800;line-height:1'>"
                    f"{prob*100:.1f}%</div><div style='color:#666'>conversion "
                    f"probability ({model_name})</div>", unsafe_allow_html=True)
                st.progress(min(max(prob, 0.0), 1.0))
                st.markdown(priority_badge(prob), unsafe_allow_html=True)

                s_label = ("Positive" if sent >= 0.05 else
                           "Negative" if sent <= -0.05 else "Neutral")
                st.metric("VADER sentiment (customer speech)", f"{sent:+.3f}", s_label,
                          delta_color="normal" if sent >= 0 else "inverse")

                st.session_state["last"] = (pipe, transcript, call_time, segment)
        else:
            st.info("Fill in the call details and click **Predict Conversion Probability**.")

    if predict and transcript.strip():
        st.divider()
        st.subheader("🔍 Word impact breakdown")
        st.caption("Leave-one-word-out analysis on the customer's response only: each "
                   "meaningful word is removed and the response re-scored. Filler words "
                   "(am, is, the, ...) are ignored.")
        with st.spinner("Analysing word impact..."):
            impacts = word_impact(pipe, transcript, call_time, segment)
        if impacts.empty:
            st.write("No individual word changed the prediction noticeably.")
        else:
            st.markdown(highlight_text(transcript, impacts), unsafe_allow_html=True)
            st.write("")
            g1, g2 = st.columns([3, 2])
            with g1:
                st.pyplot(impact_chart(impacts))
            with g2:
                pos = impacts[impacts["impact"] > 0].head(6)
                neg = impacts[impacts["impact"] < 0].head(6)
                st.markdown("**⬆ Increased conversion**")
                for _, r in pos.iterrows():
                    st.markdown(f"- `{r.word}` &nbsp; **+{r.impact*100:.1f} pp**")
                if pos.empty:
                    st.write("_none_")
                st.markdown("**⬇ Decreased conversion**")
                for _, r in neg.iterrows():
                    st.markdown(f"- `{r.word}` &nbsp; **{r.impact*100:.1f} pp**")
                if neg.empty:
                    st.write("_none_")

# ---------------------------- TAB 2 ----------------------------------------
with tab2:
    st.subheader("Logistic Regression vs Random Forest vs XGBoost")
    st.caption(f"Hold-out test set: {metrics['n_test']} calls  |  Training set: "
               f"{metrics['n_train']} calls  |  Best model selected by "
               f"{metrics['selection_criterion']}")

    keys = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    table = pd.DataFrame({m: {k: v[k] for k in keys}
                          for m, v in metrics["models"].items()}).T
    table.index.name = "Algorithm"

    best = metrics["best_model"]
    st.success(f"🏆 Best model: **{best}**  (F1 = {table.loc[best, 'F1-Score']:.3f}, "
               f"ROC-AUC = {table.loc[best, 'ROC-AUC']:.3f})")

    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#bbf7d0"),
                 width="stretch")

    st.markdown("#### Metric comparison")
    st.bar_chart(table.reset_index().melt(id_vars="Algorithm", var_name="Metric",
                                          value_name="Score"),
                 x="Metric", y="Score", color="Algorithm", stack=False, height=420)

    st.markdown("#### Confusion matrices (test set)")
    cols = st.columns(len(metrics["models"]))
    for col, (name, m) in zip(cols, metrics["models"].items()):
        cm = m["Confusion Matrix"]
        col.markdown(f"**{name}**")
        col.dataframe(pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"],
                                   columns=["Pred: No", "Pred: Yes"]),
                      width="stretch")

    shap_png = BASE / "shap_summary.png"
    if shap_png.exists():
        with st.expander(f"SHAP global feature importance ({best})", expanded=False):
            st.image(str(shap_png))

    with st.expander("Methodology"):
        st.markdown("""
- **Data:** 1,000 synthetic Indian telemarketing call openings (credit cards, loans, insurance), labels sampled with realistic noise.
- **Features:** TF-IDF (unigrams + bigrams, top 300, filler words removed) on the **customer's response only**, VADER compound sentiment of the customer's response, one-hot encoded call time and customer segment. The caller's scripted lines are stored in a separate column and excluded from the model.
- **Split:** stratified 80 / 20 train-test.
- **Models:** Logistic Regression, Random Forest (300 trees), XGBoost (250 boosted trees).
- **Explainability:** leave-one-word-out impact for live predictions, SHAP for global importance.
""")
