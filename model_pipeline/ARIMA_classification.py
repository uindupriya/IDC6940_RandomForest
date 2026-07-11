import os
import pickle

import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix, precision_score, accuracy_score, \
    recall_score, ConfusionMatrixDisplay, classification_report


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

def classify_arima_results(arima_results, stress, baseline):
    results = arima_results["predictions"].copy()
    results = results[results["label"].isin([1, 2])].copy()

    baseline_mean = float(baseline["mean"])
    stress_mean = float(stress["mean"])

    def classify(value):
        d_baseline = abs(value - baseline_mean)
        d_stress = abs(value - stress_mean)

        return 2 if d_stress < d_baseline else 1


    results["predicted_label"] = results["predicted"].apply(classify)

    return results

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=2,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label=2,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label=2,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[1, 2]
        ).tolist()
    }


def save_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, 2]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not stressed", "Stress"]
    )

    display.plot()
    plt.title("ARIMA Stress Classification")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_curve(y_true, stress_scores, output_path):
    # Convert WESAD labels to binary for ROC:
    # baseline/not stressed = 0, stress = 1
    y_true_binary = (pd.Series(y_true) == 2).astype(int)

    # ROC cannot be calculated when only one class is present
    if y_true_binary.nunique() < 2:
        print(
            f"ROC curve not generated for {output_path}: "
            "test data contains only one class."
        )
        return None

    auc_score = roc_auc_score(y_true_binary, stress_scores)
    fpr, tpr, _ = roc_curve(y_true_binary, stress_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ARIMA heuristic, AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return auc_score


def load_arima_results(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main(config_path):
    # 1. Read in config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Read in classification rules
    rules_df = pd.read_csv("classification_rules.csv")

    os.makedirs("plots", exist_ok=True)
    output_dir="plots"

    all_metrics = []


    # 2. Read in participant .pkl

    for participant in config["participants"]:

        for variable in config["variables"]:

            for window in ["1min", "3min"]:
                print(f"\n--- {participant} | {variable} | {window} ---")

                rules = rules_df[
                    (rules_df["subject"] == participant) &
                    (rules_df["variable"] == variable) &
                    (rules_df["window"] == window)
                    ]

                baseline = rules[rules["activity_label"] == 1].iloc[0]
                stress = rules[rules["activity_label"] == 2].iloc[0]

                print(f"Stress Rules: 1 - {baseline} 2 - {stress}")

                # Read ARIMA pickle
                pkl_path = f"arima_results_{participant}_{variable}_{window}.pkl"

                print("Classifying...")
                data = load_arima_results(pkl_path)

                # Classify predictions
                classified = classify_arima_results(
                    arima_results=data,
                    stress = stress,
                    baseline = baseline
                )

                print("Computing metrics...")
                # Metrics
                metric_results = compute_metrics(
                    classified["label"],
                    classified["predicted_label"]
                )

                auc_roc_path = f"{output_dir}/{participant}_{variable}_{window}_roc_auc.png"


                classified_path = f"{output_dir}/{participant}_{variable}_{window}_classified.pkl"


                confusion_matrix_path = f"{output_dir}/{participant}_{variable}_{window}_conf_matrix.png"


                metrics_path = f"{output_dir}/{participant}_{variable}_{window}_metrics.pkl"

                print("Computing AUC ROC...")
                auc_score = save_roc_curve(
                    y_true=classified["label"],
                    stress_scores=classified["predicted"],
                    output_path=auc_roc_path
                )

                metric_results["roc_auc"] = auc_score

                print("Computing Confusion Matrix...")
                save_confusion_matrix(
                    y_true=classified["label"],
                    y_pred=classified["predicted_label"],
                    output_path=confusion_matrix_path
                )

                classified.to_pickle(classified_path)

                with open(metrics_path, "wb") as file:
                    pickle.dump(metric_results, file)

                all_metrics.append({
                    "participant": participant,
                    "variable": variable,
                    "window": window,
                    "accuracy": metric_results["accuracy"],
                    "precision": metric_results["precision"],
                    "recall": metric_results["recall"],
                    "f1": metric_results["f1"],
                    "roc_auc": metric_results["roc_auc"]
                })

                print(metric_results)

                print(
                    classification_report(
                        classified["label"],
                        classified["predicted_label"],
                        labels=[1, 2],
                        target_names=["Not stressed", "Stress"],
                        zero_division=0
                    )
                )

            all_metrics_df = pd.DataFrame(all_metrics)

            all_metrics_df.to_csv(
                "all_classification_metrics.csv",
                index=False
            )

if __name__ == '__main__':
    output = main('config.yaml')