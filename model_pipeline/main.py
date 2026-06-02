import numpy as np
import yaml
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import os

from ARIMA_model import run_arima
from health_feature_helpers import FEATURE_MAP, FEATURE_HELPERS


def build_dataset(config: dict) -> pd.DataFrame:
    dataset_path = Path(config["dataset_path"])
    participants = config["participants"]
    variables = config["variables"]

    all_data = []

    for participant in participants:
        data = pd.read_csv(dataset_path / f"{participant}_wesad_chest_clean.csv")

        for variable in variables:
            signal = data[variable]

            # get feature variables
            if variable not in FEATURE_MAP:
                print("Variable not found in FEATURE_MAP: ", variable)
                continue
            derived_features = FEATURE_MAP[variable]
            for feature in derived_features:
                if feature not in FEATURE_HELPERS:
                    print("Feature not found in FEATURE_HELPERS: ", feature)
                    continue
                helper = FEATURE_HELPERS[feature]
                result = helper(signal)

                # Series output (HR)
                if hasattr(result, "__len__") and not isinstance(result, str):
                    data[feature] = result

                # Scalar output (HRV summary stats)
                else:
                    data[feature] = result


        all_data.append(data)

    return pd.concat(all_data, ignore_index=True)



def main(config_path):
    # Read in config params

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    df = build_dataset(config)

    p = config.get("p")
    d = config.get("d")
    q = config.get("q")

    # YAML "None" may load as string, so clean it
    p = None if p in ["None", "none", None] else int(p)
    d = None if d in ["None", "none", None] else int(d)
    q = None if q in ["None", "none", None] else int(q)


    if config["composite_participants"]:
        # Uses all participant datapoints together
        # Preserves a 70/30 train/test split across the full combined dataset
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            shuffle=False
        )

        arima_results = run_arima(
            train_df=train_df,
            test_df=test_df,
            target_col="HR", # TODO: change to use all feature vars
            p=p,
            d=d,
            q=q
        )

        print("Composite participant mode enabled")
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(arima_results["scores"])

        return arima_results

    else:
        # Individual participant mode
        participant_splits = {}

        for participant in config["participants"]:
            participant_df = df[df["subject"] == participant]

            train_df, test_df = train_test_split(
                participant_df,
                test_size=0.30,
                shuffle=False
            )

            arima_results = run_arima(
                train_df=train_df,
                test_df=test_df,
                target_col="HR",
                p=p,
                d=d,
                q=q
            )

            participant_splits[participant] = arima_results

            print(f"\n{participant}")
            print(f"Train shape: {train_df.shape}")
            print(f"Test shape: {test_df.shape}")
            print(arima_results["scores"])

        return participant_splits


if __name__ == '__main__':
    main('config.yaml')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
