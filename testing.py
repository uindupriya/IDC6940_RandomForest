import pandas as pd

train_df = pd.read_csv("wesad_wrist_train.csv")

print("Original labels:")
print(train_df["label"].value_counts())

label_mapping = {1: 0, 2: 1, 3: 0}
train_df["label"] = train_df["label"].map(label_mapping)

print("\nMapped labels:")
print(train_df["label"].value_counts())