import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

print("=== Chest RF Pipeline (Lags + 1-min + 3-min Rolling Windows) ===")

# -------------------------------------------------------
# 1. LOAD PRE-SPLIT, PRE-NORMALIZED CHEST FILES
# -------------------------------------------------------
print("Loading chest data splits...")
train_df = pd.read_csv("wesad_chest_train.csv")
test_df  = pd.read_csv("wesad_chest_test.csv")



FS = 700  # chest sensor Hz

# -------------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------------
def engineer_chest_features(df, fs=700):
    """
    Per-subject feature engineering for chest signals (no leakage):
      - Lag features  : t-1 to t-4        (raw signal history)
      - 1-min rolling : mean + std         (42,000 samples @ 700Hz)
      - 3-min rolling : mean + std         (126,000 samples @ 700Hz)

    Same 1-min and 3-min window durations as wrist RF for consistent
    cross-sensor comparison. Sample counts differ due to sampling rate:
      Wrist 1-min = 240 samples   | Chest 1-min = 42,000 samples
      Wrist 3-min = 720 samples   | Chest 3-min = 126,000 samples
    """
    valid_features = [f for f in ['ECG', 'EDA', 'TEMP', 'RESP'] if f in df.columns]
    print(f" -> Chest channels found: {valid_features}")

    win_1min = fs * 60           # 42,000 samples
    win_3min = fs * 60 * 3      # 126,000 samples

    print(f" -> 1-min window: {win_1min:,} samples")
    print(f" -> 3-min window: {win_3min:,} samples")
    print(f" -> Note: Large windows at 700Hz — may take 15-30 mins...")

    groups = []
    for subject, group in df.groupby('subject'):
        print(f"    Engineering features for {subject}...")
        g = group.sort_index().copy()

        for col in valid_features:
            # Lag features
            for lag in range(1, 5):
                g[f'{col}_lag_{lag}'] = g[col].shift(lag)

            # 1-min rolling (acute stress onset)
            g[f'{col}_roll1min_mean'] = (
                g[col].rolling(window=win_1min, min_periods=1).mean()
            )
            g[f'{col}_roll1min_std'] = (
                g[col].rolling(window=win_1min, min_periods=2).std()
            )

            # 3-min rolling (sustained physiological arousal)
            g[f'{col}_roll3min_mean'] = (
                g[col].rolling(window=win_3min, min_periods=1).mean()
            )
            g[f'{col}_roll3min_std'] = (
                g[col].rolling(window=win_3min, min_periods=2).std()
            )

        groups.append(g.dropna())
        print(f"    {subject} done.")

    return pd.concat(groups, ignore_index=True)


print("\nEngineering chest features (this will take a while)...")
train_engineered = engineer_chest_features(train_df)
test_engineered  = engineer_chest_features(test_df)

feature_columns = [
    c for c in train_engineered.columns
    if c not in ['subject', 'label']
]
print(f"\n -> Total feature columns: {len(feature_columns)}")
print(f"    {feature_columns}")

# -------------------------------------------------------
# 3. PER-SUBJECT TRAINING LOOP
# -------------------------------------------------------
results_summary = []
all_importances = []

for subject, group in train_engineered.groupby('subject'):
    print(f"\n{'='*50}")
    print(f"Training Chest RF for {subject}...")

    X_train = group[feature_columns]
    y_train = group['label']

    test_group = test_engineered[test_engineered['subject'] == subject]
    if test_group.empty:
        print(f" [SKIP] No test data found for {subject}")
        continue

    X_test = test_group[feature_columns]
    y_test = test_group['label']

    print(f" -> Train: {X_train.shape} | Test: {X_test.shape}")
    print(f" -> Train labels: {y_train.value_counts().to_dict()}")
    print(f" -> Test  labels: {y_test.value_counts().to_dict()}")

    rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    macro_acc = accuracy_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1   = f1_score(y_test, y_pred, average='macro')
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n[{subject}] Accuracy : {macro_acc:.4f}")
    print(f"\n[{subject}] Precision : {macro_prec:.4f}")
    print(f"[{subject}] Recall    : {macro_rec:.4f}")
    print(f"[{subject}] F1-Score  : {macro_f1:.4f}")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=['Non-Stress', 'Stress'],
        zero_division=0
    ))

    # Feature importance
    importances = pd.DataFrame({
        'subject':    subject,
        'feature':    feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"Top 10 features for {subject}:")
    print(importances.head(10).to_string(index=False))
    importances.to_csv(f"chest_feature_importance_{subject}.csv", index=False)
    all_importances.append(importances)

    model_filename = f"rf_chest_{subject}_rolling.pkl"
    joblib.dump(rf_model, model_filename)
    print(f" Saved: {model_filename}")

    results_summary.append({
        'subject':   subject,
        'sensor':    'chest',
        'accuracy': round(macro_acc, 4),
        'precision': round(macro_prec, 4),
        'recall':    round(macro_rec, 4),
        'f1_macro':  round(macro_f1, 4)
    })

# -------------------------------------------------------
# 4. SUMMARY TABLE
# -------------------------------------------------------
print(f"\n{'='*50}")
print("=== Chest RF Results Summary ===")
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))
summary_df.to_csv("rf_chest_rolling_results.csv", index=False)
print("Saved: rf_chest_rolling_results.csv")

# -------------------------------------------------------
# 5. AGGREGATE FEATURE IMPORTANCE
# -------------------------------------------------------
print("\n=== Chest Aggregate Feature Importance ===")
all_imp_df = pd.concat(all_importances, ignore_index=True)
agg_imp = (
    all_imp_df.groupby('feature')['importance']
    .mean()
    .reset_index()
    .sort_values('importance', ascending=False)
)
print(agg_imp.to_string(index=False))
agg_imp.to_csv("chest_feature_importance_aggregate.csv", index=False)
print("Saved: chest_feature_importance_aggregate.csv")

print("\n=== Chest RF Complete ===")


