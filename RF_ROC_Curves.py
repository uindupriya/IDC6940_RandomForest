import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

subjects = ['S2', 'S3', 'S4']

sensors = {
    'Chest': {
        'train': 'wesad_chest_train.csv',
        'test': 'wesad_chest_test.csv',
        'signals': ['ECG', 'EDA', 'TEMP', 'RESP']
    },
    'Wrist': {
        'train': 'wesad_wrist_train.csv',
        'test': 'wesad_wrist_test.csv',
        'signals': ['EDA', 'TEMP']
    }
}

all_results = []

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('ROC Curves — Random Forest by Sensor and Subject', fontsize=13)

for row_idx, (sensor_name, config) in enumerate(sensors.items()):
    train_df = pd.read_csv(config['train'])
    test_df = pd.read_csv(config['test'])
    signals = config['signals']

    print(f"\n{'='*50}")
    print(f"Sensor: {sensor_name} | Signals: {signals}")
    print(f"{'='*50}")

    for col_idx, subject in enumerate(subjects):
        train_sub = train_df[train_df['subject'] == subject]
        test_sub = test_df[test_df['subject'] == subject]

        X_train = train_sub[signals]
        y_train = train_sub['label']
        X_test = test_sub[signals]
        y_test = test_sub['label']

        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        accuracy = (y_pred == y_test).mean()

        print(f"{subject} | AUC={auc:.4f} | P={precision:.4f} | R={recall:.4f} | F1={f1:.4f} | Acc={accuracy:.4f}")

        all_results.append({
            'Sensor': sensor_name,
            'Subject': subject,
            'AUC': round(auc, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1': round(f1, 4),
            'Accuracy': round(accuracy, 4)
        })

        ax = axes[row_idx][col_idx]
        ax.plot(fpr, tpr, color='#D85A30', lw=2, label=f'AUC = {auc:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f'{sensor_name} — {subject}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig("rf_roc_curves_combined.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: rf_roc_curves_combined.png")

df_results = pd.DataFrame(all_results)
print("\n=== Combined Results Summary ===")
print(df_results.to_string(index=False))
df_results.to_csv("rf_combined_results.csv", index=False)
print("Saved: rf_combined_results.csv")