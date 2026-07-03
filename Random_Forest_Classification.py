import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("========================================")
print(" RANDOM FOREST MODEL EVALUATION ")
print("========================================")

# ---------------------------------------
# Load Test Dataset
# ---------------------------------------
test_df = pd.read_csv("wesad_wrist_test.csv")


# ---------------------------------------
# Create Lag Features
# ---------------------------------------
def apply_time_lags(df, features=['EDA', 'TEMP']):
    valid_features = [f for f in features if f in df.columns]

    lagged_groups = []

    for subject, group in df.groupby('subject'):
        group = group.sort_index().copy()

        for col in valid_features:
            for lag in range(1, 5):
                group[f'{col}_lag_{lag}'] = group[col].shift(lag)

        group = group.dropna()
        lagged_groups.append(group)

    return pd.concat(lagged_groups, ignore_index=True)


test_lagged = apply_time_lags(test_df)

feature_columns = [
    c for c in test_lagged.columns
    if c not in ['subject', 'label']
]

subjects = ["ALL"]

results = []

os.makedirs("Confusion_Matrices", exist_ok=True)

for subject in subjects:

    print("\n========================================")
    print(f"Evaluating {subject}")
    print("========================================")

    model_file = f"rf_model_{subject}.pkl"

    if not os.path.exists(model_file):
        print(f"{model_file} not found.")
        continue

    rf = joblib.load(model_file)

    subject_data = test_lagged[
        test_lagged['subject'] == subject
    ]

    X_test = subject_data[feature_columns]
    y_test = subject_data['label']

    y_pred = rf.predict(X_test)

    # ------------------------
    # Metrics
    # ------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[1, 2, 3],
            target_names=[
                "Neutral",
                "Stress",
                "Amusement"
            ],
            zero_division=0
        )
    )

    # ------------------------
    # Confusion Matrix
    # ------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[1, 2, 3]
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Neutral",
            "Stress",
            "Amusement"
        ]
    )

    fig, ax = plt.subplots(figsize=(6,6))

    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.title(f"Confusion Matrix - {subject}")

    plt.tight_layout()

    plt.savefig(
        f"Confusion_Matrices/{subject}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    results.append({
        "Subject": subject,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

# ---------------------------------------
# Save Overall Results
# ---------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "RF_Evaluation_Results.csv",
    index=False
)

print("\n========================================")
print("Evaluation Complete")
print("========================================")

print(results_df)

print("\nSaved:")
print("• RF_Evaluation_Results.csv")
print("• Confusion_Matrices/")