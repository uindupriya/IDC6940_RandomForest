import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==================================================
# LOAD DATA
# ==================================================

chest_df = pd.read_csv('/home/iu6/IDC6940_RandomForest/wesad_chest_clean.csv')
wrist_df = pd.read_csv('/home/iu6/IDC6940_RandomForest/wesad_wrist_clean.csv')

# ==================================================
# 1. CONFUSION MATRICES (PER SUBJECT)
# ==================================================

subjects = ["S2", "S3", "S4"]

print("\n=== Generating Per-Subject Confusion Matrices ===")

for s in subjects:
    file_path = f"rf_predictions_{s}.csv"
    print(f"\nLoading {file_path}")

    try:
        pred = pd.read_csv(file_path)
        y_true = pred["y_true"]
        y_pred = pred["y_pred"]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Non-Stress", "Stress"]
        )


        plt.figure(figsize=(6,5))
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Random Forest Confusion Matrix - {s}")
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.savefig(f"{s}_RF_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {s}_RF_confusion_matrix.png")

    except FileNotFoundError:
        print(f"Missing file: {file_path}")

# ==================================================
# 2. CHEST DATA EXPLORATION TABLE
# ==================================================

print("\n=== Generating Chest Data Exploration Table ===\n")

chest_exploration = {
    'Statistic': [
        'Total observations (S2, S3, S4)',
        'Number of subjects',
        'Non-Stress observations (label 0)',
        'Stress observations (label 1)',
        'Missing values (any signal)',
        'Sampling rate',
        'Signals used',
    ],
    'Value': [
        f"{len(chest_df):,}",
        chest_df['subject'].nunique(),
        f"{(chest_df['label']==0).sum():,}",
        f"{(chest_df['label']==1).sum():,}",
        chest_df[['ECG','EDA','TEMP','RESP']].isnull().sum().sum(),
        '700 Hz',
        'ECG, EDA, TEMP, RESP',
    ]
}

chest_exploration_df = pd.DataFrame(chest_exploration)
print(chest_exploration_df.to_string(index=False))
chest_exploration_df.to_csv('/home/iu6/IDC6940_RandomForest/Chest_Data_Exploration.csv', index=False)
print("\nSaved: Chest_Data_Exploration.csv")

# ==================================================
# 3. WRIST DATA EXPLORATION TABLE
# ==================================================

print("\n=== Generating Wrist Data Exploration Table ===\n")

wrist_exploration = {
    'Statistic': [
        'Total observations (S2, S3, S4)',
        'Number of subjects',
        'Non-Stress observations (label 0)',
        'Stress observations (label 1)',
        'Missing values (any signal)',
        'Sampling rate',
        'Signals used',
    ],
    'Value': [
        f"{len(wrist_df):,}",
        wrist_df['subject'].nunique(),
        f"{(wrist_df['label']==0).sum():,}",
        f"{(wrist_df['label']==1).sum():,}",
        wrist_df[['EDA','TEMP']].isnull().sum().sum(),
        '4 Hz',
        'EDA, TEMP',
    ]
}

wrist_exploration_df = pd.DataFrame(wrist_exploration)
print(wrist_exploration_df.to_string(index=False))
wrist_exploration_df.to_csv('/home/iu6/IDC6940_RandomForest/Wrist_Data_Exploration.csv', index=False)
print("\nSaved: Wrist_Data_Exploration.csv")

# ==================================================
# 4. CHEST DESCRIPTIVE STATISTICS TABLE (BY CONDITION)
# ==================================================

print("\n=== Generating Chest Descriptive Statistics Table by Condition ===\n")

# Group by label to isolate Non-Stress (0) and Stress (1) performance
chest_desc = chest_df.groupby('label')[['ECG','EDA','TEMP','RESP']].describe().T
chest_desc = chest_desc.rename(index={
    'count': 'N',
    'mean': 'Mean',
    'std': 'SD',
    '25%': 'Q1',
    '50%': 'Median',
    '75%': 'Q3',
    'min': 'Min',
    'max': 'Max'
})
chest_desc = chest_desc.round(4)
chest_desc.index.names = ['Variable', 'Statistic']
chest_desc = chest_desc.reset_index()

# Filter out and format specifically for your Mean and SD table
chest_summary = chest_desc[chest_desc['Statistic'].isin(['Mean', 'SD'])].copy()
print(chest_summary.to_string(index=False))
chest_summary.to_csv('/home/iu6/IDC6940_RandomForest/Chest_Descriptive_Stats_Per_Condition.csv', index=False)
print("\nSaved: Chest_Descriptive_Stats_Per_Condition.csv")

# ==================================================
# 5. WRIST DESCRIPTIVE STATISTICS TABLE (BY CONDITION)
# ==================================================

print("\n=== Generating Wrist Descriptive Statistics Table by Condition ===\n")

wrist_desc = wrist_df.groupby('label')[['EDA','TEMP']].describe().T
wrist_desc = wrist_desc.rename(index={
    'count': 'N',
    'mean': 'Mean',
    'std': 'SD',
    '25%': 'Q1',
    '50%': 'Median',
    '75%': 'Q3',
    'min': 'Min',
    'max': 'Max'
})
wrist_desc = wrist_desc.round(4)
wrist_desc.index.names = ['Variable', 'Statistic']
wrist_desc = wrist_desc.reset_index()

wrist_summary = wrist_desc[wrist_desc['Statistic'].isin(['Mean', 'SD'])].copy()
print(wrist_summary.to_string(index=False))
wrist_summary.to_csv('/home/iu6/IDC6940_RandomForest/Wrist_Descriptive_Stats_Per_Condition.csv', index=False)
print("\nSaved: Wrist_Descriptive_Stats_Per_Condition.csv")

# ==================================================
# 6. CHEST PARTICIPANT FEATURE TABLE
# ==================================================

print("\n=== Generating Chest Participant Feature Table ===\n")

chest_feature = (
    chest_df.groupby("subject")[["ECG","EDA","TEMP","RESP"]]
    .agg(["mean","std"])
)
chest_feature.columns = [f"{col}_{stat}" for col, stat in chest_feature.columns]
chest_feature = chest_feature.reset_index()
print(chest_feature.to_string(index=False))
chest_feature.to_csv('/home/iu6/IDC6940_RandomForest/Chest_Participant_Feature_Table.csv', index=False)
print("\nSaved: Chest_Participant_Feature_Table.csv")

# ==================================================
# 7. WRIST PARTICIPANT FEATURE TABLE
# ==================================================

print("\n=== Generating Wrist Participant Feature Table ===\n")

wrist_feature = (
    wrist_df.groupby("subject")[["EDA","TEMP"]]
    .agg(["mean","std"])
)
wrist_feature.columns = [f"{col}_{stat}" for col, stat in wrist_feature.columns]
wrist_feature = wrist_feature.reset_index()
print(wrist_feature.to_string(index=False))
wrist_feature.to_csv('/home/iu6/IDC6940_RandomForest/Wrist_Participant_Feature_Table.csv', index=False)
print("\nSaved: Wrist_Participant_Feature_Table.csv")

print("\n==============================")
print(" ALL TABLES GENERATED ")
print("==============================")
print("""
Chest_Data_Exploration.csv
Wrist_Data_Exploration.csv
Chest_Descriptive_Stats.csv
Wrist_Descriptive_Stats.csv
Chest_Participant_Feature_Table.csv
Wrist_Participant_Feature_Table.csv
""")
print("\nDONE")