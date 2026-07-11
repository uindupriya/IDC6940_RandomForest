import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("wesad_chest_train.csv")
test_df = pd.read_csv("wesad_chest_test.csv")

signals = ['ECG', 'EDA', 'TEMP', 'RESP']
subjects = ['S2', 'S3', 'S4']

results = []

for signal in signals:
    for subject in subjects:
        train_sub = train_df[train_df['subject'] == subject]
        test_sub = test_df[test_df['subject'] == subject]

        X_train = train_sub[[signal]]
        y_train = train_sub['label']
        X_test = test_sub[[signal]]
        y_test = test_sub['label']

        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        results.append({
            'Signal': signal,
            'Subject': subject,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4)
        })
        print(f"{signal} | {subject} | P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

df_results = pd.DataFrame(results)
print("\n=== RF Per Variable Results ===")
print(df_results.to_string(index=False))
df_results.to_csv("rf_per_variable_results.csv", index=False)