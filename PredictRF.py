import joblib
import pandas as pd
from sklearn.metrics import classification_report

print("=== Loading Test Dataset ===")
test_df = pd.read_csv("wesad_wrist_test.csv")

# 1. Re-create the lag function so your test shapes match perfectly
def apply_time_lags(df, features=['ECG', 'EDA', 'TEMP']):
    valid_features = [f for f in features if f in df.columns]
    lagged_groups = []
    for subject, group in df.groupby('subject'):
        group_copy = group.sort_index().copy()
        for col in valid_features:
            for lag in range(1, 5):
                group_copy[f'{col}_lag_{lag}'] = group_copy[col].shift(lag)
        lagged_groups.append(group_copy.dropna())
    return pd.concat(lagged_groups, ignore_index=True)

test_lagged = apply_time_lags(test_df)
feature_columns = [c for c in test_lagged.columns if c not in ['subject', 'label']]

# 2. Automatically find and loop through all unique subjects in the test data
unique_subjects = test_lagged['subject'].unique()

print(f"\n=== Found Subjects in Test Data: {list(unique_subjects)} ===")

for subject in unique_subjects:
    model_filename = f"rf_model_{subject}.pkl"
    print(f"\n--------------------------------------------------")
    print(f"Loading saved model file: {model_filename}")
    
    try:
        # Load the frozen model brain
        loaded_rf = joblib.load(model_filename)
        
        # Isolate this specific subject's test rows
        subject_test = test_lagged[test_lagged['subject'] == subject]
        X_test = subject_test[feature_columns]
        y_true = subject_test['label']
        
        # Generate predictions using the loaded file
        y_pred = loaded_rf.predict(X_test)
        
        # Output evaluation metrics
        print(f"Evaluation Results for {subject} using LOADED MODEL:")
        print(classification_report(y_true, y_pred, labels=[1, 2, 3], 
                                    target_names=['Neutral', 'Stress', 'Amusement'], 
                                    zero_division=0))
    except FileNotFoundError:
        print(f"[Warning] Could not find a saved model file named {model_filename}")

print("\n=== Multi-Model Evaluation Script Complete ===")
