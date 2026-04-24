import pandas as pd

df = pd.read_csv("data/final_dataset.csv")
print(df['label'].value_counts())