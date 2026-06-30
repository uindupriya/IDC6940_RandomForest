import datetime

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
from plotting_utils.arima_plotting_helpers import plot_arima_prediction


# def build_dataset(config: dict) -> pd.DataFrame:
#     dataset_path = Path(config["dataset_path"])
#     participants = config["participants"]
#     variables = config["variables"]
#
#     all_data = []
#
#     for participant in participants:
#         data = pd.read_csv(dataset_path / f"test_1min_{participant}.csv")
#
#         # for variable in variables:
#         #     signal = data[variable]
#         #
#         #     # get feature variables
#         #     if variable not in FEATURE_MAP:
#         #         print("Variable not found in FEATURE_MAP: ", variable)
#         #         continue
#         #     derived_features = FEATURE_MAP[variable]
#         #     for feature in derived_features:
#         #         if feature not in FEATURE_HELPERS:
#         #             print("Feature not found in FEATURE_HELPERS: ", feature)
#         #             continue
#         #         helper = FEATURE_HELPERS[feature]
#         #         result = helper(signal)
#         #
#         #         # Series output (HR)
#         #         if hasattr(result, "__len__") and not isinstance(result, str):
#         #             data[feature] = result
#         #
#         #         # Scalar output (HRV summary stats)
#         #         else:
#         #             data[feature] = result
#
#
#         all_data.append(data)
#
#     return pd.concat(all_data, ignore_index=True)



def main(config_path):
    # Read in config params

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # df = build_dataset(config)

    rolling_window_dir = Path(config["dataset_path"])

    p = config.get("p")
    d = config.get("d")
    q = config.get("q")

    # YAML "None" may load as string, so clean it
    p = None if p in ["None", "none", None] else int(p)
    d = None if d in ["None", "none", None] else int(d)
    q = None if q in ["None", "none", None] else int(q)

    target_col = "ECG_mean"


    if config["composite_participants"]:

        train_1min_df = pd.read_csv(rolling_window_dir / f"train_1min_all.csv")
        train_3min_df = pd.read_csv(rolling_window_dir / f"train_3min_all.csv")

        test_1min_df = pd.read_csv(rolling_window_dir / f"test_1min_all.csv")
        test_3min_df = pd.read_csv(rolling_window_dir / f"test_3min_all.csv")

        print(f"[--Plotting ARIMA Prediction at 1 Min Windows--]")
        arima_results_1min = run_arima(
            train_df=train_1min_df,
            test_df=test_1min_df,
            target_col=target_col, # TODO: change to use all feature vars
            p=p,
            d=d,
            q=q
        )
        print(f"[--Plotting ARIMA Prediction at 1 Min Windows--]")
        plot_arima_prediction(
            test_df=test_1min_df,
            arima_results=arima_results_1min,
            target_col=target_col,
            title="ARIMA ECG Prediction - 1 Minute Window"
        )
        print(f"[--Running ARIMA Prediction at 3 Min Windows--]")
        arima_results_3min = run_arima(
            train_df=train_3min_df,
            test_df=test_3min_df,
            target_col=target_col,  # TODO: change to use all feature vars
            p=p,
            d=d,
            q=q
        )
        print(f"[--Plotting ARIMA Prediction at 3 Min Windows--]")
        plot_arima_prediction(
            test_df=test_3min_df,
            arima_results=arima_results_3min,
            target_col=target_col,
            title="ARIMA ECG Prediction - 3 Minute Window"
        )

        print("Composite participant mode enabled")
        print("[--1 MINUTE ROLLING WINDOW--]")
        print(f"Train shape 1min: {train_1min_df.shape}")
        print(f"Test shape 1min: {test_1min_df.shape}")
        print(arima_results_1min["scores"])
        print()
        print("[--3 MINUTE ROLLING WINDOW--]")
        print(f"Train shape 3min: {train_1min_df.shape}")
        print(f"Test shape 3min: {test_1min_df.shape}")
        print(arima_results_1min["scores"])


        print(arima_results_1min["scores"])
        output = {}
        output[target_col]["1min"]= arima_results_1min
        output[target_col]["3min"] = arima_results_3min
        return output

    else:
        # Individual participant mode
        participant_splits = {}


        for participant in config["participants"]:
            participant_splits[participant] = {
                target_col: {}
            }
            participant_train_1min_df = pd.read_csv(rolling_window_dir / f"train_1min_{participant}.csv")
            participant_test_1min_df = pd.read_csv(rolling_window_dir / f"test_1min_{participant}.csv")


            print(f"[--Running {participant} ARIMA Prediction at 1 Min Windows--]")
            arima_results_1min = run_arima(
                train_df=participant_train_1min_df,
                test_df=participant_test_1min_df,
                target_col=target_col,
                p=p,
                d=d,
                q=q
            )

            print(f"[--Plotting {participant} ARIMA Prediction at 1 Min Windows--]")
            plot_arima_prediction(
                test_df=participant_test_1min_df,
                arima_results=arima_results_1min,
                target_col=target_col,
                title="ARIMA ECG Prediction - 1 Minute Window"
            )

            participant_splits[participant][target_col]["1min"] = arima_results_1min

            participant_train_3min_df = pd.read_csv(rolling_window_dir / f"train_3min_{participant}.csv")
            participant_test_3min_df = pd.read_csv(rolling_window_dir / f"test_3min_{participant}.csv")

            print(f"[--Running {participant} ARIMA Prediction at 3 Min Windows--]")
            arima_results_3min = run_arima(
                train_df=participant_train_3min_df,
                test_df=participant_test_3min_df,
                target_col=target_col,
                p=p,
                d=d,
                q=q
            )

            print(f"[--Plotting {participant} ARIMA Prediction at 3 Min Windows--]")
            plot_arima_prediction(
                test_df=participant_test_3min_df,
                arima_results=arima_results_3min,
                target_col=target_col,
                title="ARIMA ECG Prediction - 3 Minute Window"
            )

            participant_splits[participant][target_col]["3min"] = arima_results_3min

            print("[--1 MINUTE ROLLING WINDOW--]")
            print(f"Train shape 1min: {participant_train_1min_df.shape}")
            print(f"Test shape 1min: {participant_test_1min_df.shape}")
            print(arima_results_1min["scores"])
            print()
            print("[--3 MINUTE ROLLING WINDOW--]")
            print(f"Train shape 3min: {participant_train_3min_df.shape}")
            print(f"Test shape 3min: {participant_train_3min_df.shape}")
            print(arima_results_3min["scores"])



        return participant_splits


if __name__ == '__main__':
    output = main('config.yaml')
    import pickle

    with open("arima_results_S4_ECG_mean_5_0_0_6-25_10-35.pkl", "wb") as f:
        pickle.dump(output, f)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
