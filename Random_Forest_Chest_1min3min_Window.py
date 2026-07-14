import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

WINDOW_DIR = Path("windowed_outputs")

OUTPUT_DIR = Path("rf_window_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ROC_DIR = OUTPUT_DIR / "roc_curves"
ROC_DIR.mkdir(exist_ok=True)

CM_DIR = OUTPUT_DIR / "confusion_matrices"
CM_DIR.mkdir(exist_ok=True)

WINDOWS = ["1min", "3min"]

SIGNALS = ["ECG", "EDA", "TEMP", "RESP"]

# -------------------------------------------------------
# Find Subjects
# -------------------------------------------------------

subjects = sorted({
    f.stem.split("_")[-1]
    for f in WINDOW_DIR.glob("train_*_S*.csv")
})

print("Subjects:", subjects)

results = []

# -------------------------------------------------------
# Loop over windows
# -------------------------------------------------------

for window in WINDOWS:

    print(f"\n{'='*60}")
    print(f"Processing {window}")
    print(f"{'='*60}")

    for subject in subjects:

        train_file = WINDOW_DIR / f"train_{window}_{subject}.csv"
        test_file = WINDOW_DIR / f"test_{window}_{subject}.csv"

        if not train_file.exists() or not test_file.exists():
            print(f"Skipping {subject}")
            continue



        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)

        # Keep only Baseline (1) and Stress (2)
        train = train[train["label"].isin([1, 2])].copy()
        test = test[test["label"].isin([1, 2])].copy()

        # Convert to binary classification
        # Baseline = 0, Stress = 1
        train["label"] = train["label"].map({1: 0, 2: 1})
        test["label"] = test["label"].map({1: 0, 2: 1})

        for signal in SIGNALS:

            feature_cols = [
                f"{signal}_mean",
                f"{signal}_std",
                f"{signal}_min",
                f"{signal}_max"
            ]

            X_train = train[feature_cols]
            y_train = train["label"]

            X_test = test[feature_cols]
            y_test = test["label"]

            # -------------------------------------------------------
            # Train RF
            # -------------------------------------------------------

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1]

            # -------------------------------------------------------
            # Metrics
            # -------------------------------------------------------

            acc = accuracy_score(y_test, y_pred)

            prec = precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            )

            rec = recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="macro"
            )

            auc = roc_auc_score(y_test, y_prob)

            # -------------------------------------------------------
            # ROC Curve
            # -------------------------------------------------------

            fpr, tpr, thresholds = roc_curve(
                y_test,
                y_prob
            )

            roc_df = pd.DataFrame({
                "FPR": fpr,
                "TPR": tpr,
                "Threshold": thresholds
            })

            roc_df.to_csv(
                ROC_DIR /
                f"roc_{signal}_{window}_{subject}.csv",
                index=False
            )

            # -------------------------------------------------------
            # Confusion Matrix
            # -------------------------------------------------------

            cm = confusion_matrix(y_test, y_pred)

            cm_df = pd.DataFrame(
                cm,
                index=["Actual_0", "Actual_1"],
                columns=["Pred_0", "Pred_1"]
            )

            cm_df.to_csv(
                CM_DIR /
                f"confusion_{signal}_{window}_{subject}.csv"
            )

            # -------------------------------------------------------
            # Results Summary
            # -------------------------------------------------------

            results.append({
                "signal": signal,
                "subject": subject,
                "window": window,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4)
            })

            print(
                f"{subject:>3} | "
                f"{signal:<4} | "
                f"{window:<4} | "
                f"AUC={auc:.3f} "
                f"F1={f1:.3f}"
            )

# -------------------------------------------------------
# Save Summary
# -------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df[
    [
        "signal",
        "subject",
        "window",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]
]

results_df.to_csv(
    OUTPUT_DIR / "rf_window_results.csv",
    index=False
)

print("\nFinished.")
print(results_df)