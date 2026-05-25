import pickle
import numpy as np
import pandas as pd
import os

subjects = ['S2','S3','S4','S5','S6','S7','S8','S9',
            'S10','S11','S13','S14','S15','S16','S17']

chest_dfs = []
wrist_dfs = []

for subject in subjects:
    path = f'/home/iu6/IDC6940_RandomForest/Dataset_WESAD/{subject}/{subject}.pkl'
    
    if not os.path.exists(path):
        print(f"{subject} not found, skipping...")
        continue

    print(f"Loading {subject}...")

    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    labels = data['label'].flatten()
    mask = np.isin(labels, [1, 2, 3])

    # ---------------- CHEST ----------------
    chest_df = pd.DataFrame({
        'subject': subject,
        'ECG': data['signal']['chest']['ECG'].flatten()[mask],
        'EDA': data['signal']['chest']['EDA'].flatten()[mask],
        'TEMP': data['signal']['chest']['Temp'].flatten()[mask],
        'RESP': data['signal']['chest']['Resp'].flatten()[mask],
        'label': labels[mask]
    })

    # only basic cleaning (NO spike removal)
    chest_df = chest_df.dropna()
    chest_df = chest_df[(chest_df['TEMP'] > 20) & (chest_df['TEMP'] < 45)]
    chest_df = chest_df[chest_df['EDA'] >= 0]

    chest_dfs.append(chest_df)

    # ---------------- WRIST ----------------
    wrist_label = labels
    label_4hz = wrist_label[::175]

    eda_wrist = data['signal']['wrist']['EDA'].flatten()
    temp_wrist = data['signal']['wrist']['TEMP'].flatten()

    min_len = min(len(label_4hz), len(eda_wrist), len(temp_wrist))

    wrist_df = pd.DataFrame({
        'subject': subject,
        'EDA': eda_wrist[:min_len],
        'TEMP': temp_wrist[:min_len],
        'label': label_4hz[:min_len]
    })

    wrist_df = wrist_df[wrist_df['label'].isin([1, 2, 3])]
    wrist_dfs.append(wrist_df)

# ---------------- FINAL COMBINE ----------------
chest_final = pd.concat(chest_dfs, ignore_index=True)
wrist_final = pd.concat(wrist_dfs, ignore_index=True)

print("Chest shape:", chest_final.shape)
print("Wrist shape:", wrist_final.shape)

print(chest_final['label'].value_counts())
print(wrist_final['label'].value_counts())

chest_final.to_csv('wesad_chest_clean.csv', index=False)
wrist_final.to_csv('wesad_wrist_clean.csv', index=False)

print("Saved cleaned datasets.")