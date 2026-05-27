import pickle
import numpy as np
import pandas as pd
import os

subjects = [
    'S2','S3','S4','S5','S6','S7','S8','S9',
    'S10','S11','S13','S14','S15','S16','S17'
]

VALID_LABELS = [1, 2, 3]

chest_dfs = []
wrist_dfs = []

print("Starting WESAD preprocessing pipeline...")


# ----------------------------------------------------
# Helper: filter labels properly
# ----------------------------------------------------
def filter_valid_labels(labels):
    labels = np.array(labels).flatten()
    mask = np.isin(labels, VALID_LABELS)
    return labels[mask], mask


# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
for subject in subjects:

    path = f'/home/iu6/IDC6940_RandomForest/Dataset_WESAD/{subject}/{subject}.pkl'

    if not os.path.exists(path):
        print(f"[SKIP] {subject} not found")
        continue

    print(f"\nProcessing {subject}...")

    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # -------------------------
    # LABELS
    # -------------------------
    labels = np.array(data['label']).flatten()
    labels, mask = filter_valid_labels(labels)

    # -------------------------
    # CHEST SIGNALS
    # -------------------------
    chest_ecg = np.array(data['signal']['chest']['ECG']).flatten()[:len(labels)]
    chest_eda = np.array(data['signal']['chest']['EDA']).flatten()[:len(labels)]
    chest_temp = np.array(data['signal']['chest']['Temp']).flatten()[:len(labels)]
    chest_resp = np.array(data['signal']['chest']['Resp']).flatten()[:len(labels)]

    chest_df = pd.DataFrame({
        'subject': subject,
        'ECG': chest_ecg,
        'EDA': chest_eda,
        'TEMP': chest_temp,
        'RESP': chest_resp,
        'label': labels
    })

    chest_df = chest_df.dropna()
    chest_df = chest_df[(chest_df['TEMP'] > 20) & (chest_df['TEMP'] < 45)]
    chest_df = chest_df[chest_df['EDA'] >= 0]

    chest_dfs.append(chest_df)

    print(f"[CHEST] {subject}: {chest_df.shape}")

    # -------------------------
    # WRIST SIGNALS
    # -------------------------
    wrist_labels = np.array(data['label']).flatten()

    wrist_eda = np.array(data['signal']['wrist']['EDA']).flatten()
    wrist_temp = np.array(data['signal']['wrist']['TEMP']).flatten()

    min_len = min(len(wrist_eda), len(wrist_temp), len(wrist_labels))

    wrist_df = pd.DataFrame({
        'subject': subject,
        'EDA': wrist_eda[:min_len],
        'TEMP': wrist_temp[:min_len],
        'label': wrist_labels[:min_len]
    })

    wrist_df = wrist_df[wrist_df['label'].isin(VALID_LABELS)]
    wrist_df = wrist_df.dropna()

    wrist_dfs.append(wrist_df)

    print(f"[WRIST] {subject}: {wrist_df.shape}")


# ----------------------------------------------------
# FINAL DATASET
# ----------------------------------------------------
chest_final = pd.concat(chest_dfs, ignore_index=True)
wrist_final = pd.concat(wrist_dfs, ignore_index=True)

print("\nChest shape:", chest_final.shape)
print("Wrist shape:", wrist_final.shape)

chest_final.to_csv("wesad_chest_clean.csv", index=False)
wrist_final.to_csv("wesad_wrist_clean.csv", index=False)

print("\nSaved cleaned datasets successfully.")