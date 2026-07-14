from pathlib import Path
import pickle
import pandas as pd

results_dir = Path("")

rows = []

for pkl_file in results_dir.glob("*min.pkl"):

    with open(pkl_file, "rb") as f:
        result = pickle.load(f)

    scores = result["scores"]

    parts = pkl_file.stem.split("_")

    participant = parts[0]
    variable = "_".join(parts[1:-1])
    window = parts[-1]



    rows.append({
        "participant": participant,
        "variable": variable,
        "window": window,
        "MAE": round(result["scores"]["MAE"], 3),
        "MSE": round(result["scores"]["MSE"], 3),
        "RMSE": round(result["scores"]["RMSE"], 3),
        "MAPE": round(result["scores"]["MAPE"], 3),
    })

metrics_df = pd.DataFrame(rows)
import re

# Fix participant, variable columns
metrics_df["participant"] = metrics_df["variable"].str.extract(r"(S\d+)")
metrics_df["variable"] = metrics_df["variable"].str.replace(
    r"results_S\d+_", "", regex=True
)

# Round metrics
metrics_df[["MAE", "MSE", "RMSE", "MAPE"]] = metrics_df[
    ["MAE", "MSE", "RMSE", "MAPE"]
].round(3)

# Reorder columns
metrics_df = metrics_df[
    ["participant", "variable", "window", "MAE", "MSE", "RMSE", "MAPE"]
]

metrics_df.to_csv("arima_metrics_summary.csv", index=False)

print(metrics_df)