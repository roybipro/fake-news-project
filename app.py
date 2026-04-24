import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Bangla Fake News Detector", page_icon="📰")

MODEL_PATH = "model/"

# ---------------------------
# LOAD MODEL (cached)
# ---------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    label = "Fake ❌" if pred == 1 else "Real ✅"
    return label, confidence, probs.tolist()[0]

# ---------------------------
# UI
# ---------------------------
st.title("📰 Bangla Fake News Detector")
st.write("Enter a Bangla news text (headline + content works best).")

# Input box
text = st.text_area("✍️ Paste news text here", height=200)

col1, col2 = st.columns(2)

with col1:
    check_btn = st.button("🔍 Check News")

with col2:
    example_btn = st.button("📌 Try Example")

# Example text
if example_btn:
    text = """এই ভিডিওতে দাবি করা হচ্ছে যে একটি ওষুধ খেলে ২৪ ঘন্টার মধ্যে ক্যান্সার সম্পূর্ণ ভালো হয়ে যাবে।
বিশেষজ্ঞরা বলছেন এটি সম্পূর্ণ ভুয়া এবং বিভ্রান্তিকর তথ্য।"""
    st.text_area("Example Loaded:", value=text, height=150)

# Prediction
if check_btn:
    if len(text.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        label, confidence, probs = predict(text)

        st.subheader("📊 Result")
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence:.2f}")

        # Confidence bar
        st.progress(float(confidence))

        # Detailed probabilities
        st.write("🔎 Class Probabilities:")
        st.write({
            "Real": round(probs[0], 3),
            "Fake": round(probs[1], 3)
        })

# Footer
st.markdown("---")
st.caption("Built with Transformers + Streamlit")