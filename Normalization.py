import pandas as pd
from sklearn.preprocessing import StandardScaler

print("=== Starting Block-Stratified Time Splitting & Normalization ===")

# 1. LOAD THE CLEAN DATASETS
print("Loading clean datasets...")
chest_data = pd.read_csv("wesad_chest_clean.csv")
wrist_data = pd.read_csv("wesad_wrist_clean.csv")

chest_features = ['ECG', 'EDA', 'TEMP', 'RESP']
wrist_features = ['EDA', 'TEMP']

chest_train_list, chest_test_list = [], []
wrist_train_list, wrist_test_list = [], []

# ====================================================
# 2. CHEST PIPELINE: SPLIT 80/20 WITHIN EACH CONDITION BLOCK
# ====================================================
print("\nProcessing Chest Streams...")
for subject, subject_group in chest_data.groupby('subject'):
    subject_group = subject_group.sort_index()
    
    # Store train/test parts for this specific subject across all labels
    sub_train_parts = []
    sub_test_parts = []
    
    # Split chronologically within each condition block to preserve label variety
    for label, label_group in subject_group.groupby('label'):
        split_idx = int(len(label_group) * 0.8)
        sub_train_parts.append(label_group.iloc[:split_idx])
        sub_test_parts.append(label_group.iloc[split_idx:])
        
    # Combine back into a unified train/test split for this subject
    train_split = pd.concat(sub_train_parts).sort_index().copy()
    test_split = pd.concat(sub_test_parts).sort_index().copy()
    
    # Local scaling: fit ONLY on training metrics to avoid leakage
    scaler = StandardScaler()
    train_split[chest_features] = scaler.fit_transform(train_split[chest_features])
    test_split[chest_features] = scaler.transform(test_split[chest_features])
    
    chest_train_list.append(train_split)
    chest_test_list.append(test_split)
    print(f" -> {subject} Chest arrays split stratified & standardized.")

# ====================================================
# 3. WRIST PIPELINE: SPLIT 80/20 WITHIN EACH CONDITION BLOCK
# ====================================================
print("\nProcessing Wrist Streams...")
for subject, subject_group in wrist_data.groupby('subject'):
    subject_group = subject_group.sort_index()
    
    sub_train_parts = []
    sub_test_parts = []
    
    # Split chronologically within each condition block to preserve label variety
    for label, label_group in subject_group.groupby('label'):
        split_idx = int(len(label_group) * 0.8)
        sub_train_parts.append(label_group.iloc[:split_idx])
        sub_test_parts.append(label_group.iloc[split_idx:])
        
    train_split = pd.concat(sub_train_parts).sort_index().copy()
    test_split = pd.concat(sub_test_parts).sort_index().copy()
    
    scaler = StandardScaler()
    train_split[wrist_features] = scaler.fit_transform(train_split[wrist_features])
    test_split[wrist_features] = scaler.transform(test_split[wrist_features])
    
    wrist_train_list.append(train_split)
    wrist_test_list.append(test_split)
    print(f" -> {subject} Wrist arrays split stratified & standardized.")

# ====================================================
# 4. EXPORT HANDOFF FILES
# ====================================================
print("\nRecombining and saving stratified data splits...")
pd.concat(chest_train_list, ignore_index=True).to_csv("wesad_chest_train.csv", index=False)
pd.concat(chest_test_list, ignore_index=True).to_csv("wesad_chest_test.csv", index=False)
pd.concat(wrist_train_list, ignore_index=True).to_csv("wesad_wrist_train.csv", index=False)
pd.concat(wrist_test_list, ignore_index=True).to_csv("wesad_wrist_test.csv", index=False)

print("\nSuccess! Block-stratified handoff files saved with zero leakage.")
