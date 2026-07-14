import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt


LABEL_NAMES = {
    1: "Non-Stress",
    2: "Stress"
}

LABEL_COLORS = {
    1: "#378ADD",
    2: "#D85A30"
}


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

MODEL_DIR = Path("../model_pipeline")
DATA_DIR = Path("../windowed_outputs")
PLOT_DIR = Path("arima_prediction_plots")

PLOT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------
# Plot Function
# -------------------------------------------------------

def plot_arima_prediction(
        arima_results,
        target_col="ECG",
        title="ARIMA Prediction"):


    plot_df = arima_results["predictions"]


    plt.figure(figsize=(20, 6))


    # Actual signal
    plt.plot(
        plot_df.index,
        plot_df["actual"],
        color="black",
        linestyle="-",
        linewidth=1.2,
        label="Actual"
    )


    # Predictions
    plt.plot(
        plot_df.index,
        plot_df["predicted"],
        color="#D85A30",
        linestyle="--",
        linewidth=1.2,
        label="ARIMA Prediction"
    )


    # Add stress regions
    for label, group in plot_df.groupby("label"):

        if int(label) == 2:

            plt.fill_between(
                group.index,
                plot_df["actual"].min(),
                plot_df["actual"].max(),
                alpha=0.15,
                color="#D85A30",
                label="Stress"
            )


    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel(target_col)

    plt.legend()

    plt.tight_layout()


    filename = (
        PLOT_DIR /
        f"{title}.png"
    )

    plt.savefig(
        filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")



# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":


    model_files = sorted(
        MODEL_DIR.glob("*.pkl")
    )


    print(
        f"Found {len(model_files)} ARIMA models"
    )


    for model_file in model_files:


        print("\nProcessing:", model_file.name)


        # ---------------------------------------
        # Parse filename
        # Expected:
        # arima_ECG_S2_1min.pkl
        # ---------------------------------------

        parts = model_file.stem.split("_")

        signal = parts[3]
        subject = parts[2]
        window = parts[5]


        # ---------------------------------------
        # Load model results
        # ---------------------------------------

        arima_results = joblib.load(
            model_file
        )


        # ---------------------------------------
        # Select correct signal
        # ---------------------------------------

        target_col = signal


        # ---------------------------------------
        # Plot
        # ---------------------------------------

        title = (
            f"ARIMA_{signal}_{subject}_{window}"
        )


        plot_arima_prediction(
            arima_results,
            target_col=target_col,
            title=title
        )


    print("\nFinished all ARIMA prediction plots.")