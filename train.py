import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# SETTINGS

MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LEN = 128
SAMPLE_SIZE = 4000


# LOAD DATA

df = pd.read_csv("data/final_dataset.csv")

print("Original dataset:")
print(df['label'].value_counts())


# BASIC VALIDATION

required_cols = ["headline", "content", "label"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# Ensure label is numeric 0/1
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)


# BUILD TEXT (SAFE)

df['headline'] = df['headline'].astype(str)
df['content'] = df['content'].astype(str)
df['text'] = (df['headline'].str.strip() + " " + df['content'].str.strip()).str.strip()

# Drop rows where text is empty
df = df[df['text'].str.len() > 0]

print("\nAfter cleaning:")
print(df['label'].value_counts())


# SAFE BALANCING

real_df = df[df['label'] == 0]
fake_df = df[df['label'] == 1]

print(f"\nReal count: {len(real_df)}")
print(f"Fake count: {len(fake_df)}")

if len(real_df) == 0 or len(fake_df) == 0:
    print("⚠️ One class missing after cleaning. Skipping balancing.")
else:
    min_size = min(len(real_df), len(fake_df))
    real_df = real_df.sample(min_size, random_state=42)
    fake_df = fake_df.sample(min_size, random_state=42)
    df = pd.concat([real_df, fake_df]).sample(frac=1).reset_index(drop=True)

print("\nDataset used for training:")
print(df['label'].value_counts())


# OPTIONAL SAMPLING

df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)


# SPLIT

texts = df['text'].tolist()
labels = df['label'].tolist()

if len(texts) == 0:
    raise ValueError("❌ Dataset is empty after preprocessing!")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)


# TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LEN)


# DATASET CLASS

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)


# MODEL

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)


# METRICS

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# TRAINING CONFIG
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    gradient_accumulation_steps=2,
    logging_steps=50,
    dataloader_pin_memory=False)

# TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# TRAIN

trainer.train()


# EVALUATE

results = trainer.evaluate()
print("\nEvaluation Results:")
print(results)


# SAVE

model.save_pretrained("model/")
tokenizer.save_pretrained("model/")

print("\n✅ Training Complete!")