import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

WINDOW_DIR = Path("windowed_outputs")

WINDOWS = ["1min", "3min"]

label_map = {
    1: "Baseline",
    2: "Stress"
}

palette = {
    "Baseline": "#378ADD",
    "Stress": "#D85A30"
}


# -------------------------------------------------------
# LOAD WINDOWED DATA
# -------------------------------------------------------

def load_window_data(window):

    files = list(
        WINDOW_DIR.glob(f"train_{window}_S*.csv")
    )

    dfs = []

    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    data = pd.concat(
        dfs,
        ignore_index=True
    )

    data["Condition"] = data["label"].map(label_map)

    return data



# -------------------------------------------------------
# 1. WINDOWED CHEST SIGNAL PAIRPLOT
# -------------------------------------------------------

for window in WINDOWS:

    wrist = load_window_data(window)

    sample = pd.concat([
        grp.sample(
            min(1000, len(grp)),
            random_state=42
        )
        for _, grp in wrist.groupby("label")
    ]).reset_index(drop=True)


    pair_cols = [
        "ECG_mean",
        "EDA_mean",
        "TEMP_mean",
        "RESP_mean",
        "Condition"
    ]


    g = sns.pairplot(
        sample[pair_cols],
        hue="Condition",
        palette=palette,
        plot_kws={
            "alpha":0.4,
            "s":12
        },
        diag_kind="kde",
        corner=True
    )

    g.figure.suptitle(
        f"WESAD Wrist Windowed Signals — {window}",
        y=1.02,
        fontsize=13
    )

    plt.savefig(
        f"wesad_wrist_window_pairplot_{window}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Saved wesad_wrist_window_pairplot_{window}.png"
    )



# -------------------------------------------------------
# 2. EDA vs TEMP WINDOWED SCATTER
# -------------------------------------------------------

for window in WINDOWS:

    wrist = load_window_data(window)

    sample = pd.concat([
        grp.sample(
            min(3000, len(grp)),
            random_state=42
        )
        for _, grp in wrist.groupby("label")
    ]).reset_index(drop=True)


    plt.figure(figsize=(6,5))

    sns.scatterplot(
        data=sample,
        x="TEMP_mean",
        y="EDA_mean",
        hue="Condition",
        palette=palette,
        alpha=0.4,
        s=15
    )

    plt.title(
        f"Wrist Windowed EDA vs TEMP — {window}"
    )

    plt.xlabel(
        "TEMP Mean"
    )

    plt.ylabel(
        "EDA Mean"
    )

    plt.tight_layout()

    plt.savefig(
        f"wesad_wrist_EDA_TEMP_{window}.png",
        dpi=150
    )

    plt.show()

    print(
        f"Saved wesad_wrist_EDA_TEMP_{window}.png"
    )



# -------------------------------------------------------
# 3. WINDOW FEATURE DISTRIBUTIONS
# -------------------------------------------------------

for window in WINDOWS:

    wrist = load_window_data(window)


    sample = pd.concat([
        grp.sample(
            min(3000, len(grp)),
            random_state=42
        )
        for _, grp in wrist.groupby("label")
    ]).reset_index(drop=True)


    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14,10)
    )

    fig.suptitle(
        f"Wrist Window Features — {window}",
        fontsize=13
    )


    plots = [

        (
            "EDA_mean",
            "EDA_std",
            "EDA Mean vs Std"
        ),

        (
            "TEMP_mean",
            "TEMP_std",
            "TEMP Mean vs Std"
        ),

        (
            "RESP_mean",
            "RESP_std",
            "RESP Mean vs Std"
        ),

        (
            "ECG_mean",
            "ECG_std",
            "ECG Mean vs Std"
        )

    ]


    for ax, (x,y,title) in zip(
        axes.flatten(),
        plots
    ):

        sns.scatterplot(
            data=sample,
            x=x,
            y=y,
            hue="Condition",
            palette=palette,
            alpha=0.4,
            s=12,
            ax=ax
        )

        ax.set_title(title)

        ax.spines[
            ["top","right"]
        ].set_visible(False)


    plt.tight_layout()

    plt.savefig(
        f"wesad_wrist_window_features_{window}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Saved wesad_wrist_window_features_{window}.png"
    )