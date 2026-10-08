import re
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "final_model"   # folder saved by your notebook
MAX_LEN = 512                                      # same as training

EXAMPLES = {
    "Phishing example": (
        "Dear customer, your account has been suspended due to unusual activity. "
        "Verify your password immediately at http://secure-login-update.com/verify "
        "or your account will be permanently closed within 24 hours."
    ),
    "Legitimate example": (
        "Hi team, attached are the notes from yesterday's budget review meeting. "
        "Please send me any corrections by Friday and we will finalize the numbers next week. Thanks, Louise"
    ),
}

st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Loading BERT model...")
def load_model():
    if not MODEL_DIR.exists():
        st.error(f"Model folder not found: {MODEL_DIR}\n\nPut app.py in the same folder as `final_model`.")
        st.stop()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()
    return tokenizer, model, device


def normalize_like_training(text: str) -> str:
    """The training data looks lowercased with punctuation removed
    (e.g. 'info@site.com' -> 'infositecom'). Apply the same cleaning."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def predict(text, tokenizer, model, device):
    inputs = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu()
    n_tokens = len(tokenizer(text, truncation=False, verbose=False)["input_ids"])
    return float(probs[1]), n_tokens   # probability of class 1 = phishing


def set_example(name):
    st.session_state["email_text"] = EXAMPLES[name]


tokenizer, model, device = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("BERT (`bert-base-uncased`) fine-tuned to classify emails as phishing or legitimate.")
    st.write(f"Running on: **{device}**")
    threshold = st.slider("Phishing threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Flag as phishing when P(phishing) is at or above this value. "
                               "Lower = catches more phishing but raises false alarms.")
    normalize = st.checkbox("Normalize text like the training data", value=True,
                            help="Lowercase and strip punctuation, which matches how the training emails look.")
    st.caption("Held-out test set of the training data: accuracy 99.47%, F1 0.9949.")

# ---------- Main ----------
st.title("🛡️ Phishing Email Detector")
st.write("Paste an email (subject + body) below and click **Analyze**.")

c1, c2 = st.columns(2)
c1.button("Load phishing example", on_click=set_example, args=("Phishing example",), use_container_width=True)
c2.button("Load legitimate example", on_click=set_example, args=("Legitimate example",), use_container_width=True)

email_text = st.text_area("Email text", key="email_text", height=260,
                          placeholder="Paste the email subject and body here...")

if st.button("Analyze", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please paste some email text first.")
    else:
        text = normalize_like_training(email_text) if normalize else email_text
        if not text:
            st.warning("Nothing left to analyze after cleaning the text.")
            st.stop()

        with st.spinner("Analyzing..."):
            p_phish, n_tokens = predict(text, tokenizer, model, device)

        is_phish = p_phish >= threshold
        if is_phish:
            st.error(f"⚠️ Likely PHISHING  (P = {p_phish:.1%})")
        else:
            st.success(f"✅ Likely legitimate  (P(phishing) = {p_phish:.1%})")

        st.progress(min(max(p_phish, 0.0), 1.0), text=f"P(phishing) = {p_phish:.1%}")

        m1, m2 = st.columns(2)
        m1.metric("P(phishing)", f"{p_phish:.1%}")
        m2.metric("P(legitimate)", f"{1 - p_phish:.1%}")

        if n_tokens > MAX_LEN:
            st.info(f"This email has {n_tokens} tokens. Only the first {MAX_LEN} were analyzed, "
                    "which is the same limit used in training.")

st.divider()
st.caption(
    "Research prototype. The model was trained on one public dataset (the legitimate class is largely "
    "corporate Enron-style mail), so it can be wrong on emails that look different, and it only reads the text, "
    "not sender authentication, headers, or link targets. Do not rely on it as your only protection."
)