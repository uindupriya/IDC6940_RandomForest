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
        
    # Get the raw, unbroken 700Hz timeline bounds
    raw_labels = np.array(data['label']).flatten()
    raw_ecg = np.array(data['signal']['chest']['ECG']).flatten()
    raw_eda = np.array(data['signal']['chest']['EDA']).flatten()
    raw_temp = np.array(data['signal']['chest']['Temp']).flatten()
    raw_resp = np.array(data['signal']['chest']['Resp']).flatten()
    
    # Track minimum length to trim trailing-end sensor shutoff mismatches safely
    min_len = min(len(raw_labels), len(raw_ecg), len(raw_eda), len(raw_temp), len(raw_resp))

    # -------------------------
    # CHEST SIGNALS (100% continuous for NeuroKit2 windowing)
    # -------------------------
    chest_df = pd.DataFrame({
        'subject': subject,
        'ECG': raw_ecg[:min_len],
        'EDA': raw_eda[:min_len],
        'TEMP': raw_temp[:min_len],
        'RESP': raw_resp[:min_len],
        'label': raw_labels[:min_len]  # Keeps ALL labels (0,1,2,3,4...) so data stays unbroken
    })
    
    # Handle artifacts without changing the shape or deleting rows
    chest_df.loc[(chest_df['TEMP'] <= 20) | (chest_df['TEMP'] >= 45), 'TEMP'] = np.nan
    chest_df.loc[chest_df['EDA'] < 0, 'EDA'] = np.nan
    chest_df = chest_df.ffill().bfill()  # Cleans sensor errors seamlessly
    
    chest_dfs.append(chest_df)
    print(f"[CHEST CONTINUOUS] {subject}: {chest_df.shape}")

    # -------------------------
    # WRIST SIGNALS (Downsampled labels to match native 4Hz sensors)
    # -------------------------
    wrist_eda = np.array(data['signal']['wrist']['EDA']).flatten()
    wrist_temp = np.array(data['signal']['wrist']['TEMP']).flatten()
    
    wrist_len = min(len(wrist_eda), len(wrist_temp))
    # Pick every 175th label to match the slow 4Hz wrist sensors perfectly by time
    wrist_labels = raw_labels[::175][:wrist_len]
    
    wrist_df = pd.DataFrame({
        'subject': subject,
        'EDA': wrist_eda[:wrist_len],
        'TEMP': wrist_temp[:wrist_len],
        'label': wrist_labels
    })
    
    # Wrist can be filtered row-by-row because her NeuroKit pipeline isn't analyzing the wrist
    wrist_df = wrist_df[wrist_df['label'].isin(VALID_LABELS)]
    wrist_df = wrist_df.dropna()
    wrist_dfs.append(wrist_df)
    print(f"[WRIST CLEAN] {subject}: {wrist_df.shape}")

# ----------------------------------------------------
# FINAL DATASET EXPORT
# ----------------------------------------------------
print("\nCombining datasets...")
chest_final = pd.concat(chest_dfs, ignore_index=True)
wrist_final = pd.concat(wrist_dfs, ignore_index=True)

print("\nFinal Chest shape (Continuous):", chest_final.shape)
print("Final Wrist shape (Clean Tabular):", wrist_final.shape)

chest_final.to_csv("wesad_chest_clean.csv", index=False)
wrist_final.to_csv("wesad_wrist_clean.csv", index=False)
print("\nSaved cleaned datasets successfully. Ready for handoff!")
