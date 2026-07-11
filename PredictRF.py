import joblib
import pandas as pd
from sklearn.metrics import classification_report

print("=== Loading Test Dataset ===")

test_df = pd.read_csv("wesad_wrist_test.csv")

#label_mapping = {1: 0, 2: 1, 3: 0}
#test_df['label'] = test_df['label'].map(label_mapping)

# =========================================================
# LAG FEATURE FUNCTION
# =========================================================

def engineer_features(df, fs=4):

    valid_features = [f for f in ['EDA', 'TEMP'] if f in df.columns]

    win_1min = fs * 60
    win_3min = fs * 60 * 3

    groups = []

    for subject, group in df.groupby('subject'):

        g = group.sort_index().copy()

        for col in valid_features:

            for lag in range(1,5):
                g[f'{col}_lag_{lag}'] = g[col].shift(lag)

            g[f'{col}_roll1min_mean'] = (
                g[col].rolling(window=win_1min, min_periods=1).mean()
            )

            g[f'{col}_roll1min_std'] = (
                g[col].rolling(window=win_1min, min_periods=2).std()
            )

            g[f'{col}_roll3min_mean'] = (
                g[col].rolling(window=win_3min, min_periods=1).mean()
            )

            g[f'{col}_roll3min_std'] = (
                g[col].rolling(window=win_3min, min_periods=2).std()
            )

        groups.append(g.dropna())

    return pd.concat(groups, ignore_index=True)

# Apply lag features
test_lagged = engineer_features(test_df)

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

    model_filename = f"rf_model_{subject}_rolling.pkl"

    print("\n----------------------------------------")
    print(f"Evaluating Subject: {subject}")

    try:
        model = joblib.load(model_filename)

        subject_data = test_lagged[test_lagged['subject'] == subject]

        X_test = subject_data[feature_columns]
        y_true = subject_data['label']

        y_pred = model.predict(X_test)
        print("\nPrediction distribution:")
        print(pd.Series(y_pred).value_counts().sort_index())

        print("\nTrue distribution:")
        print(y_true.value_counts().sort_index())

        from sklearn.metrics import confusion_matrix

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred, labels=[0,1]))

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