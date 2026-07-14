import pandas as pd

# ----------------------------------------------------
# Load results
# ----------------------------------------------------

arima = pd.read_csv("model_pipeline/all_classification_metrics.csv")
rf = pd.read_csv("rf_window_results/rf_window_results.csv")

# ----------------------------------------------------
# Standardize column names
# ----------------------------------------------------

# participant -> subject
arima = arima.rename(columns={
    "participant": "subject"
})

# ECG_mean -> ECG
arima["signal"] = arima["variable"].str.replace(
    "_mean",
    "",
    regex=False
)

# ----------------------------------------------------
# Keep only required columns
# ----------------------------------------------------

arima = arima[
    [
        "signal",
        "subject",
        "window",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
]

rf = rf[
    [
        "signal",
        "subject",
        "window",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]
]

# ----------------------------------------------------
# Rename metric columns
# ----------------------------------------------------

arima = arima.rename(columns={
    "accuracy": "ARIMA Accuracy",
    "precision": "ARIMA Precision",
    "recall": "ARIMA Recall",
    "f1": "ARIMA F1"
})

rf = rf.rename(columns={
    "accuracy": "RF Accuracy",
    "precision": "RF Precision",
    "recall": "RF Recall",
    "f1": "RF F1"
})

# ----------------------------------------------------
# Merge RF and ARIMA
# ----------------------------------------------------

comparison = pd.merge(
    rf,
    arima,
    on=["signal", "subject", "window"],
    how="inner"
)

# ----------------------------------------------------
# Arrange columns
# ----------------------------------------------------

comparison = comparison[
    [
        "signal",
        "subject",
        "window",
        "RF Accuracy",
        "ARIMA Accuracy",
        "RF Precision",
        "ARIMA Precision",
        "RF Recall",
        "ARIMA Recall",
        "RF F1",
        "ARIMA F1"
    ]
]

comparison = comparison.rename(columns={
    "signal": "Signal",
    "subject": "Subject",
    "window": "Window"
})

# ----------------------------------------------------
# Sort for readability
# ----------------------------------------------------

comparison = comparison.sort_values(
    by=["Signal", "Subject", "Window"]
).reset_index(drop=True)

# ----------------------------------------------------
# Split by window
# ----------------------------------------------------

comparison_1min = comparison[
    comparison["Window"] == "1min"
]

comparison_3min = comparison[
    comparison["Window"] == "3min"
]

# ----------------------------------------------------
# Save outputs
# ----------------------------------------------------

comparison_1min.to_csv(
    "arima_vs_rf_1min_results.csv",
    index=False
)

comparison_3min.to_csv(
    "arima_vs_rf_3min_results.csv",
    index=False
)

print("\nSaved:")
print("  arima_vs_rf_1min_results.csv")
print("  arima_vs_rf_3min_results.csv")

print("\n=== 1-Minute Comparison ===")
print(comparison_1min)

print("\n=== 3-Minute Comparison ===")
print(comparison_3min)