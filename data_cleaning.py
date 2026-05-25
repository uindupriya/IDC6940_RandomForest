import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
            'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

chest_dfs = []
wrist_dfs = []

def remove_spikes(series, threshold=5):
    rolling_median = series.rolling(window=10, center=True).median()
    diff = np.abs(series - rolling_median)
    mad = diff.median()
    spike_mask = diff > (threshold * mad)
    series[spike_mask] = rolling_median[spike_mask]
    return series

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
    
    # --- CHEST ---
    chest_df = pd.DataFrame({
        'subject': subject,
        'ECG': data['signal']['chest']['ECG'].flatten()[mask],
        'EDA': data['signal']['chest']['EDA'].flatten()[mask],
        'TEMP': data['signal']['chest']['Temp'].flatten()[mask],
        'RESP': data['signal']['chest']['Resp'].flatten()[mask],
        'label': labels[mask]
    })
    
    # handle nulls
    if chest_df.isnull().sum().any():
        chest_df = chest_df.fillna(method='ffill').fillna(method='bfill')
    
    # remove impossible values
    chest_df = chest_df[(chest_df['TEMP'] > 20) & (chest_df['TEMP'] < 45)]
    chest_df = chest_df[chest_df['EDA'] >= 0]
    
    # remove spikes
    for col in ['ECG', 'EDA', 'TEMP', 'RESP']:
        chest_df[col] = remove_spikes(chest_df[col].copy())
    
    chest_dfs.append(chest_df)
    print(f"{subject} chest shape: {chest_df.shape}")
    
    # --- WRIST ---
    wrist_df = pd.DataFrame({
        'subject': subject,
        'BVP': data['signal']['wrist']['BVP'].flatten(),
        'EDA': data['signal']['wrist']['EDA'].flatten(),
        'TEMP': data['signal']['wrist']['TEMP'].flatten(),
        'label': data['label'].flatten()
    })
    
    # wrist has different sampling rate so resample labels
    # keep only labels 1, 2, 3
    # --- WRIST ---
    # wrist signals have different sampling rates
    # BVP = 64Hz, EDA = 4Hz, TEMP = 4Hz, label = 700Hz
    # we need to use each signal with its own label
    
    wrist_label = data['label'].flatten()
    
    # get wrist EDA and TEMP (both at 4Hz)
    # label needs to be downsampled from 700Hz to 4Hz
    # ratio = 700/4 = 175, take every 175th label
    label_4hz = wrist_label[::175]
    eda_wrist = data['signal']['wrist']['EDA'].flatten()
    temp_wrist = data['signal']['wrist']['TEMP'].flatten()
    
    # match lengths
    min_len = min(len(label_4hz), len(eda_wrist), len(temp_wrist))
    
    wrist_df = pd.DataFrame({
        'subject': subject,
        'EDA': eda_wrist[:min_len],
        'TEMP': temp_wrist[:min_len],
        'label': label_4hz[:min_len]
    })
    
    # keep only labels 1, 2, 3
    wrist_df = wrist_df[wrist_df['label'].isin([1, 2, 3])]
# combine all subjects
chest_final = pd.concat(chest_df, ignore_index=True)
wrist_final = pd.concat(wrist_df, ignore_index=True)

# normalize
scaler = StandardScaler()
chest_features = ['ECG', 'EDA', 'TEMP', 'RESP']
wrist_features = ['BVP', 'EDA', 'TEMP']

chest_final[chest_features] = scaler.fit_transform(chest_final[chest_features])
wrist_final[wrist_features] = scaler.fit_transform(wrist_final[wrist_features])

print("\nChest dataset shape:", chest_final.shape)
print("Wrist dataset shape:", wrist_final.shape)

print("\nChest label distribution:")
print(chest_final['label'].value_counts())

print("\nWrist label distribution:")
print(wrist_final['label'].value_counts())

# save both
chest_final.to_csv('/home/iu6/IDC6940_RandomForest/wesad_chest_clean.csv', index=False)
wrist_final.to_csv('/home/iu6/IDC6940_RandomForest/wesad_wrist_clean.csv', index=False)
print("\nSaved wesad_chest_clean.csv and wesad_wrist_clean.csv!")