import joblib
import pandas as pd
from sklearn.metrics import classification_report

print("=== Loading Test Dataset ===")

test_df = pd.read_csv("wesad_wrist_test.csv")

label_mapping = {1: 0, 2: 1, 3: 0}
test_df['label'] = test_df['label'].map(label_mapping)

# =========================================================
# LAG FEATURE FUNCTION
# =========================================================

def apply_time_lags(df, features=['ECG', 'EDA', 'TEMP']):
    valid_features = [f for f in features if f in df.columns]
    lagged_groups = []

    for subject, group in df.groupby('subject'):
        group = group.sort_index().copy()

        for col in valid_features:
            for lag in range(1, 5):
                group[f'{col}_lag_{lag}'] = group[col].shift(lag)

        lagged_groups.append(group.dropna())

    return pd.concat(lagged_groups, ignore_index=True)

# Apply lag features
test_lagged = apply_time_lags(test_df)

feature_columns = [
    c for c in test_lagged.columns
    if c not in ['subject', 'label']
]

unique_subjects = test_lagged['subject'].unique()

print(f"\nSubjects Found: {list(unique_subjects)}")

# =========================================================
# STORE ALL PREDICTIONS FOR CONFUSION MATRIX
# =========================================================

all_y_true = []
all_y_pred = []

# =========================================================
# SUBJECT-WISE MODEL EVALUATION
# =========================================================

for subject in unique_subjects:

    model_filename = f"rf_model_{subject}.pkl"

    print("\n----------------------------------------")
    print(f"Evaluating Subject: {subject}")

    try:
        model = joblib.load(model_filename)

        subject_data = test_lagged[test_lagged['subject'] == subject]

        X_test = subject_data[feature_columns]
        y_true = subject_data['label']

        y_pred = model.predict(X_test)

        # Store for global confusion matrix
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=['Non-Stress', 'Stress'],
            zero_division=0
        ))


        # Save per-subject predictions (optional but good)
        pd.DataFrame({
            "subject": subject,
            "y_true": y_true,
            "y_pred": y_pred
        }).to_csv(f"rf_predictions_{subject}.csv", index=False)

    except FileNotFoundError:
        print(f"[WARNING] Model not found: {model_filename}")

# =========================================================
# SAVE COMBINED PREDICTIONS (IMPORTANT FOR CONFUSION MATRIX)
# =========================================================

pd.DataFrame({
    "y_true": all_y_true,
    "y_pred": all_y_pred
}).to_csv("rf_predictions.csv", index=False)

print("\n=== Saved: rf_predictions.csv (for confusion matrix) ===")
print("=== Multi-Model Evaluation Complete ===")