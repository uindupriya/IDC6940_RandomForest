import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc
)


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

RF_DIR = Path("rf_window_results")

CONF_DIR = RF_DIR / "confusion_matrices"
ROC_DIR = RF_DIR / "roc_curves"

OUTPUT_DIR = Path("rf_plots")
OUTPUT_DIR.mkdir(exist_ok=True)



# -------------------------------------------------------
# RF Confusion Matrix (ARIMA style)
# -------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, auc

# ---------------------------------------------------------
# RF Confusion Matrix
# ---------------------------------------------------------
def save_rf_confusion_matrix(csv_path, output_path):

    cm_df = pd.read_csv(csv_path, index_col=0)

    fig, ax = plt.subplots(figsize=(5, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm_df.values,
        display_labels=["Not stressed", "Stress"]
    ).plot(ax=ax, colorbar=False)

    ax.set_title("Random Forest Stress Classification")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------
# RF ROC Curve
# ---------------------------------------------------------
def save_rf_roc_curve(csv_path, output_path):

    roc_df = pd.read_csv(csv_path)

    roc_auc = auc(
        roc_df["FPR"],
        roc_df["TPR"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(
        roc_df["FPR"],
        roc_df["TPR"],
        linewidth=2,
        label=f"Random Forest (AUC = {roc_auc:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        color="gray"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return roc_auc

# -------------------------------------------------------
# Generate plots
# -------------------------------------------------------

subjects = [
    "S2",
    "S3",
    "S4"
]

signals = [
    "ECG",
    "EDA",
    "RESP",
    "TEMP"
]

windows = [
    "1min",
    "3min"
]


for signal in signals:

    for window in windows:

        for subject in subjects:


            name = (
                f"{signal}_{window}_{subject}"
            )


            conf_file = (
                CONF_DIR /
                f"confusion_{name}.csv"
            )


            roc_file = (
                ROC_DIR /
                f"roc_{name}.csv"
            )


            if conf_file.exists():

                save_rf_confusion_matrix(
                    conf_file,
                    f"{OUTPUT_DIR / name}_conf_matrix"
                )


            if roc_file.exists():

                save_rf_roc_curve(
                    roc_file,
                    f"{OUTPUT_DIR / name}_roc_curve"
                )


print("RF plots generated in ARIMA-compatible style.")

