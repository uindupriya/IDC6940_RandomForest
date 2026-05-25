import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats

# list of available subjects - add more as you upload them
subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 
            'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']

all_dfs = []

for subject in subjects:
    path = f'/home/iu6/IDC6940_RandomForest/Dataset_WESAD/{subject}/{subject}.pkl'
    
    # check if file exists
    if not os.path.exists(path):
        print(f"{subject} not found, skipping...")
        continue
    
    print(f"Loading {subject}...")
    
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # extract signals
    chest_eda = data['signal']['chest']['EDA'].flatten()
    chest_temp = data['signal']['chest']['Temp'].flatten()
    chest_resp = data['signal']['chest']['Resp'].flatten()
    chest_ecg = data['signal']['chest']['ECG'].flatten()
    labels = data['label'].flatten()
    
    # keep only labels 1, 2, 3
    mask = np.isin(labels, [1, 2, 3])
    
    df = pd.DataFrame({
        'subject': subject,
        'ECG': chest_ecg[mask],
        'EDA': chest_eda[mask],
        'TEMP': chest_temp[mask],
        'RESP': chest_resp[mask],
        'label': labels[mask]
    })
    
    # handle null values
    if df.isnull().sum().any():
        print(f"{subject} has nulls — filling with forward fill...")
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
    
    # remove physically impossible TEMP values
    before = len(df)
    df = df[df['TEMP'] > 20]  # skin temp cannot be below 20°C
    df = df[df['TEMP'] < 45]  # skin temp cannot be above 45°C
    after = len(df)
    if before != after:
        print(f"{subject} removed {before - after} impossible TEMP values")
    
    # remove extreme EDA values (negative EDA is impossible)
    before = len(df)
    df = df[df['EDA'] >= 0]
    after = len(df)
    if before != after:
        print(f"{subject} removed {before - after} impossible EDA values")
    
    all_dfs.append(df)
    print(f"{subject} loaded — shape: {df.shape}")

# combine all subjects
final_df = pd.concat(all_dfs, ignore_index=True)

print("\nFinal dataset shape:", final_df.shape)
print(final_df['label'].value_counts())
print("\nNull check:", final_df.isnull().sum().tolist())

# normalize signals
scaler = StandardScaler()
features = ['ECG', 'EDA', 'TEMP', 'RESP']
final_df[features] = scaler.fit_transform(final_df[features])

print("\nAfter normalization:")
print(final_df[features].describe().round(4))

# save to CSV
final_df.to_csv('/home/iu6/IDC6940_RandomForest/wesad_clean.csv', index=False)
print("\nSaved to wesad_clean.csv!")