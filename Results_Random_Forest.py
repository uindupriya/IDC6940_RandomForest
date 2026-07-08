import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

print("\n====================================")
print(" RANDOM FOREST FINAL RESULTS REPORT ")
print("====================================\n")

# ==================================================
# 1. LOAD PREDICTIONS (PER SUBJECT FILES)
# ==================================================

subjects = ["S2", "S3", "S4"]

all_results = []

for s in subjects:
    file_path = f"rf_predictions_{s}.csv"

    print(f"Loading: {file_path}")

    df = pd.read_csv(file_path)
    df["subject"] = s
    all_results.append(df)

# Combine all subjects
pred = pd.concat(all_results, ignore_index=True)

print("\nCombined prediction shape:", pred.shape)

# ==================================================
# 2. CONFUSION MATRIX (PER SUBJECT)
# ==================================================

print("\nGenerating Confusion Matrices per Subject...\n")

for s in subjects:

    sub_df = pred[pred["subject"] == s]

    y_true = sub_df["y_true"]
    y_pred = sub_df["y_pred"]

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Stress", "Stress"]
    )

    plt.figure(figsize=(6,5))
    disp.plot(cmap="Blues", values_format="d")

    plt.title(f"Random Forest Confusion Matrix - {s}")

    # FIX CUT-OFF ISSUE
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)

    out_file = f"{s}_RF_confusion_matrix.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_file}")

# ==================================================
# 3. GLOBAL CONFUSION MATRIX (ALL SUBJECTS)
# ==================================================

print("\nGenerating Global Confusion Matrix...\n")

cm_global = confusion_matrix(pred["y_true"], pred["y_pred"], labels=[0,1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_global,
    display_labels=["Non-Stress", "Stress"]
)

plt.figure(figsize=(6,5))
disp.plot(cmap="Blues", values_format="d")

plt.title("Random Forest Confusion Matrix (S2–S4)")

plt.tight_layout()
plt.subplots_adjust(left=0.25)

plt.savefig("RF_global_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: RF_global_confusion_matrix.png")

# ==================================================
# 4. CLASSIFICATION REPORT (GLOBAL)
# ==================================================

print("\n==============================")
print(" CLASSIFICATION REPORT (RF) ")
print("==============================\n")

print(classification_report(
    pred["y_true"],
    pred["y_pred"],
    target_names=["Non-Stress", "Stress"]
))

# ==================================================
# 5. FEATURE TABLE (YOUR REQUEST)
# ==================================================

print("\nGenerating Subject Feature Table...\n")

df = pd.read_csv("wesad_chest_clean.csv")

label_mapping = {1: 0, 2: 1, 3: 0}
df['label'] = df['label'].map(label_mapping)

feature_table = (
    df.groupby("subject")[["ECG","EDA","TEMP","RESP"]]
      .agg(["mean","std"])
)

# flatten columns
feature_table.columns = [
    f"{col}_{stat}" for col, stat in feature_table.columns
]

feature_table = feature_table.reset_index()

print(feature_table)

feature_table.to_csv("Participant_Feature_Table.csv", index=False)

print("\nSaved: Participant_Feature_Table.csv")

# ==================================================
# 6. FINAL SUMMARY
# ==================================================

print("\n==============================")
print(" FINAL OUTPUT FILES GENERATED ")
print("==============================")

print("""
S2_RF_confusion_matrix.png
S3_RF_confusion_matrix.png
S4_RF_confusion_matrix.png
RF_global_confusion_matrix.png
Participant_Feature_Table.csv
""")

print("\nDONE ✔")