import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv('/home/iu6/IDC6940_RandomForest/wesad_chest_clean.csv')

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

        cm = confusion_matrix(y_true, y_pred, labels=[1,2,3])

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Neutral", "Stress", "Amusement"]
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
# 2. DATA EXPLORATION TABLE
# ==================================================

print("\n=== Generating Data Exploration Table ===\n")

exploration = {
    'Statistic': [
        'Total observations (S2, S3, S4)',
        'Number of subjects',
        'Baseline observations (label 1)',
        'Stress observations (label 2)',
        'Amusement observations (label 3)',
        'Missing values (any signal)',
        'Sampling rate (chest signals)',
        'Signals used',
    ],
    'Value': [
        f"{len(df):,}",
        df['subject'].nunique(),
        f"{(df['label']==1).sum():,}",
        f"{(df['label']==2).sum():,}",
        f"{(df['label']==3).sum():,}",
        df[['ECG','EDA','TEMP','RESP']].isnull().sum().sum(),
        '700 Hz',
        'ECG, EDA, TEMP, RESP',
    ]
}

exploration_df = pd.DataFrame(exploration)
print(exploration_df.to_string(index=False))
exploration_df.to_csv('/home/iu6/IDC6940_RandomForest/WESAD_Data_Exploration.csv', index=False)
print("\nSaved: WESAD_Data_Exploration.csv")

# ==================================================
# 3. DESCRIPTIVE STATISTICS TABLE
# ==================================================

print("\n=== Generating Descriptive Statistics Table ===\n")

desc = df[['ECG','EDA','TEMP','RESP']].describe().T
desc = desc.rename(columns={
    'count': 'N',
    'mean': 'Mean',
    'std': 'SD',
    '25%': 'Q1',
    '50%': 'Median',
    '75%': 'Q3',
    'min': 'Min',
    'max': 'Max'
})
desc = desc.round(4)
desc.index.name = 'Variable'
desc = desc.reset_index()
print(desc.to_string(index=False))
desc.to_csv('/home/iu6/IDC6940_RandomForest/WESAD_Descriptive_Stats.csv', index=False)
print("\nSaved: WESAD_Descriptive_Stats.csv")

# ==================================================
# 4. PARTICIPANT FEATURE TABLE
# ==================================================

print("\n=== Generating Participant Feature Table ===\n")

feature_table = (
    df.groupby("subject")[["ECG","EDA","TEMP","RESP"]]
      .agg(["mean","std"])
)

feature_table.columns = [
    f"{col}_{stat}" for col, stat in feature_table.columns
]

feature_table = feature_table.reset_index()
print(feature_table.to_string(index=False))
feature_table.to_csv('/home/iu6/IDC6940_RandomForest/Participant_Feature_Table.csv', index=False)
print("\nSaved: Participant_Feature_Table.csv")

print("\n==============================")
print(" ALL TABLES GENERATED ")
print("==============================")
print("""
WESAD_Data_Exploration.csv
WESAD_Descriptive_Stats.csv
Participant_Feature_Table.csv
""")
print("\nDONE")