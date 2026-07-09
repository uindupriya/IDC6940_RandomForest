import pickle

import pandas as pd
import yaml


def get_classification_rules(df, subject, activity_label, variables):
    # Compute participant-specific classification rules for selected variables.

    subset = df[
        (df["subject"] == subject) &
        (df["label"] == activity_label)
    ]

    rules = []

    for var in variables:
        mean = subset[var].mean()
        std = subset[var].std()

        rules.append({
            "subject": subject,
            "activity_label": activity_label,
            "variable": var,
            "mean": mean,
            "std": std,
            "min": subset[var].min(),
            "max": subset[var].max(),
            "lower_1sd": mean - std,
            "upper_1sd": mean + std
        })

    return pd.DataFrame(rules)

def load_arima_results(path, subject, variable, window):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data[subject][variable][window]

def main(config_path):
    # 1. Read in config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)


    # 2. Read in participant .pkl
    from pathlib import Path
    import pickle

    for participant in config["participants"]:

        for variable in config["variables"]:

            for window in config["windows"]:
                print(f"\n--- {participant} | {variable} | {window} ---")

                # Read ARIMA pickle
                pkl_path = find_arima_pickle(
                    output_dir=config["dataset_path"],
                    subject=participant,
                    variable=variable,
                    p=config["p"],
                    d=config["d"],
                    q=config["q"]
                )

                with open(pkl_path, "rb") as f:
                    participant_splits = pickle.load(f)

                arima_results = participant_splits[participant][variable][window]

                # Read matching train/test csv
                train_df = pd.read_csv(
                    Path(config["dataset_path"]) / f"train_{window}_{participant}.csv"
                )

                test_df = pd.read_csv(
                    Path(config["dataset_path"]) / f"test_{window}_{participant}.csv"
                )

                # Build rules
                rules = pd.concat([
                    get_classification_rules(
                        train_df,
                        participant,
                        activity_label=1,
                        variables=[variable]
                    ),
                    get_classification_rules(
                        train_df,
                        participant,
                        activity_label=2,
                        variables=[variable]
                    )
                ])

                # Classify predictions
                classified = classify_arima_results(
                    arima_results=arima_results,
                    test_df=test_df,
                    rules_df=rules,
                    subject=participant,
                    variable=variable
                )

                # Metrics
                metrics = compute_metrics(
                    classified["true_binary"],
                    classified["predicted_binary"]
                )

                print(metrics)
    #3. Classify ARIMA

    # 4. Compute metrics/confusion matrix


if __name__ == '__main__':
    output = main('config.yaml')