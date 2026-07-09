import datetime

import numpy as np
import yaml
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import os

from ARIMA_model import run_arima
from health_feature_helpers import FEATURE_MAP, FEATURE_HELPERS
from model_pipeline.ARIMA_classification import get_classification_rules
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

    # target_col = "ECG_mean"

    rules_df = pd.DataFrame({})


    if config["composite_participants"]:
        for target_col in config["variables"]:

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



            with open(f"arima_results_all_{target_col}_5_0_0_7-5_9-39.pkl", "wb") as f:
                pickle.dump(output, f)
            return output

    else:
        # Individual participant mode
        # participant_splits = {}


        for participant in config["participants"]:
            for target_col in config["variables"]:
                # participant_splits[participant] = {
                #     target_col: {}
                # }
                participant_train_1min_df = pd.read_csv(rolling_window_dir / f"train_1min_{participant}.csv")
                participant_test_1min_df = pd.read_csv(rolling_window_dir / f"test_1min_{participant}.csv")


                print(f"[--Running {participant} ARIMA Prediction for {target_col} at 1 Min Windows--]")
                arima_results_1min = run_arima(
                    train_df=participant_train_1min_df,
                    test_df=participant_test_1min_df,
                    target_col=target_col,
                    p=p,
                    d=d,
                    q=q
                )

                print(f"[--Plotting {participant} ARIMA Prediction for {target_col} at 1 Min Windows--]")
                plot_arima_prediction(
                    test_df=participant_test_1min_df,
                    arima_results=arima_results_1min,
                    target_col=target_col,
                    title="ARIMA ECG Prediction - 1 Minute Window"
                )

                with open(f"arima_results_{participant}_{target_col}_1min.pkl", "wb") as f:
                    pickle.dump(arima_results_1min, f)

                participant_rules = pd.concat([
                    get_classification_rules(
                        participant_train_1min_df,
                        participant,
                        activity_label=1,
                        variables=[target_col]
                    ),
                    get_classification_rules(
                        participant_train_1min_df,
                        participant,
                        activity_label=2,
                        variables=[target_col]
                    )
                ], ignore_index=True)

                participant_rules["window"] = "1min"

                rules_df = pd.concat(
                    [rules_df, participant_rules],
                    ignore_index=True
                )


                participant_train_3min_df = pd.read_csv(rolling_window_dir / f"train_3min_{participant}.csv")
                participant_test_3min_df = pd.read_csv(rolling_window_dir / f"test_3min_{participant}.csv")

                print(f"[--Running {participant} ARIMA Prediction for {target_col} at 3 Min Windows--]")
                arima_results_3min = run_arima(
                    train_df=participant_train_3min_df,
                    test_df=participant_test_3min_df,
                    target_col=target_col,
                    p=p,
                    d=d,
                    q=q
                )

                print(f"[--Plotting {participant} ARIMA Prediction for {target_col} at 3 Min Windows--]")
                plot_arima_prediction(
                    test_df=participant_test_3min_df,
                    arima_results=arima_results_3min,
                    target_col=target_col,
                    title="ARIMA ECG Prediction - 3 Minute Window"
                )

                with open(f"arima_results_{participant}_{target_col}_3min.pkl", "wb") as f:
                    pickle.dump(arima_results_3min, f)

                participant_rules = pd.concat([
                    get_classification_rules(
                        participant_train_3min_df,
                        participant,
                        activity_label=1,
                        variables=[target_col]
                    ),
                    get_classification_rules(
                        participant_train_3min_df,
                        participant,
                        activity_label=2,
                        variables=[target_col]
                    )
                ], ignore_index=True)

                participant_rules["window"] = "3min"

                rules_df = pd.concat(
                    [rules_df, participant_rules],
                    ignore_index=True
                )

                print("[--1 MINUTE ROLLING WINDOW--]")
                print(f"Train shape 1min: {participant_train_1min_df.shape}")
                print(f"Test shape 1min: {participant_test_1min_df.shape}")
                print(arima_results_1min["scores"])
                print()
                print("[--3 MINUTE ROLLING WINDOW--]")
                print(f"Train shape 3min: {participant_train_3min_df.shape}")
                print(f"Test shape 3min: {participant_train_3min_df.shape}")
                print(arima_results_3min["scores"])

    rules_df.to_csv("classification_rules.csv")





if __name__ == '__main__':
    main('config.yaml')





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
