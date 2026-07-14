import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

try:
    from pmdarima import auto_arima
except ImportError:
    auto_arima = None


def compute_continuous_scores(y_true, y_pred) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Avoid divide-by-zero issues
    nonzero_mask = y_true != 0
    if nonzero_mask.any():
        mape = np.mean(
            np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
        ) * 100
    else:
        mape = np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }

def run_arima(train_df, test_df, target_col: str, p=None, d=None, q=None):
    y_train = train_df[target_col].dropna()
    y_test = test_df[target_col].dropna()

    order = (p, d, q)

    print(f"Using ARIMA order: {order}")

    model = ARIMA(y_train, order=order)
    fitted_model = model.fit()

    predictions = []

    for actual in y_test:
        pred = fitted_model.forecast(steps=1).iloc[0]
        predictions.append(pred)

        fitted_model = fitted_model.append(
            [actual],
            refit=False
        )

    predictions = pd.Series(predictions, index=y_test.index)

    scores = compute_continuous_scores(
        y_true=y_test,
        y_pred=predictions
    )

    results_df = pd.DataFrame({
        "subject": test_df.loc[y_test.index, "subject"].values,
        "label": test_df.loc[y_test.index, "label"].values,
        "actual": y_test.values,
        "predicted": predictions.values
    })


    return {
        "model": fitted_model,
        "order": order,
        "predictions": results_df,
        "scores": scores
    }
