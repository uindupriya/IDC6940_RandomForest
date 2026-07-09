import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LABEL_COLORS = {
    1: "blue",      # Neutral
    2: "red",       # Stress
    3: "green"      # Amusement
}

LABEL_NAMES = {
    1: "Neutral",
    2: "Stress",
    3: "Amusement"
}


def plot_arima_prediction(test_df, arima_results, target_col="ECG", title="ARIMA Prediction"):
    # Adjust this key if your run_arima output names predictions differently
    plot_df = test_df.copy().reset_index(drop=True)

    plot_df["prediction"] = arima_results["predictions"]["predicted"]
    plot_df["actual"] = arima_results["predictions"]["actual"]
    plot_df["label"] = test_df["label"]

    plt.figure(figsize=(14, 6))

    for label, group in plot_df.groupby("label"):
        label = int(label)
        color = LABEL_COLORS.get(label, "gray")
        label_name = LABEL_NAMES.get(label, f"Label {label}")

        # Actual line = solid
        plt.plot(
            plot_df.index,
            plot_df["actual"],
            color="black",
            linestyle="-",
            linewidth=1,
            label=f"Actual - {label_name}"
        )

        # Predicted line = dashed
        plt.plot(
            plot_df.index,
            plot_df["prediction"],
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"Predicted - {label_name}"
        )

    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{title}.png")