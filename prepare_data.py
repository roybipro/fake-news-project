import pandas as pd

# Load datasets
real = pd.read_csv("data/LabeledAuthentic-7K.csv")
fake = pd.read_csv("data/LabeledFake-1K.csv")

print("Real columns:", real.columns)
print("Fake columns:", fake.columns)

# 🔥 IMPORTANT: Use correct label column OR create new one

# Remove old label if exists (to avoid confusion)
if 'label' in real.columns:
    real = real.drop(columns=['label'])

if 'label' in fake.columns:
    fake = fake.drop(columns=['label'])

# Add correct labels
real['label'] = 0
fake['label'] = 1

# Combine
df = pd.concat([real, fake])

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Save
df.to_csv("data/final_dataset.csv", index=False)

print("✅ Dataset created successfully!")
print(df['label'].value_counts())