from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "model/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    outputs = model(**inputs)
    logits = outputs.logits
    
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs).item()
    
    confidence = probs[0][pred].item()
    label = "Fake" if pred == 1 else "Real"

    return label, confidence


# TEST MULTIPLE INPUTS

samples = [
    "সরকার নতুন শিক্ষা নীতি ঘোষণা করেছে",
    "এই খবরটি সম্পূর্ণ ভুয়া এবং ভিত্তিহীন",
    "বাংলাদেশ ক্রিকেট দল বিশ্বকাপে দুর্দান্ত জয় পেয়েছে",
    "এই ওষুধ খেলে ১ দিনে ক্যান্সার ভালো হয়ে যাবে",
    "আজ ঢাকায় ভারী বৃষ্টির সম্ভাবনা রয়েছে",
    "এই ভিডিওটি শেয়ার করলে আপনি ধনী হয়ে যাবেন"
]

print("\n🔍 Predictions:\n")

for text in samples:
    label, conf = predict(text)
    print(f"📰 Text: {text}")
    print(f"👉 Prediction: {label} | Confidence: {conf:.2f}")
    print("-" * 50)