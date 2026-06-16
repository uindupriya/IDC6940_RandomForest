import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# --------------------------------
# Config
# --------------------------------
ALL_PARTICIPANTS = False

TRAIN_PATH = "wesad_chest_train.csv"
TEST_PATH = "wesad_chest_test.csv"

OUTPUT_DIR = Path("windowed_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SIGNAL_COLS = ["ECG", "EDA", "TEMP", "RESP"]
SUBJECT_COL = "subject"
LABEL_COL = "label"

SAMPLING_RATE = 1

WINDOWS = {
    "1min": 60,
    "3min": 180
}

STRIDE = 30

MAX_WORKERS = max(1, os.cpu_count() - 1)


# --------------------------------
# Helpers
# --------------------------------
def compute_zscores(train_df, test_df, signal_cols):
    print("Calculating z-scores from train set...")

    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in signal_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()

        if std == 0 or pd.isna(std):
            print(f"Warning: {col} has std=0 or NaN. Skipping z-score.")
            train_df[f"{col}_z"] = train_df[col]
            test_df[f"{col}_z"] = test_df[col]
        else:
            train_df[f"{col}_z"] = (train_df[col] - mean) / std
            test_df[f"{col}_z"] = (test_df[col] - mean) / std

    print("Finished z-scores.")
    return train_df, test_df


def create_subject_windows(subject, group, window_seconds, stride):
    window_size = window_seconds * SAMPLING_RATE
    rows = []

    group = group.reset_index(drop=True)

    if len(group) < window_size:
        return pd.DataFrame()

    for start in range(0, len(group) - window_size + 1, stride):
        end = start + window_size
        window = group.iloc[start:end]

        row = {
            SUBJECT_COL: subject,
            "window_size_sec": window_seconds,
            "start_idx": start,
            "end_idx": end - 1
        }

        for col in SIGNAL_COLS:
            use_col = f"{col}_z" if f"{col}_z" in window.columns else col

            row[f"{col}_mean"] = window[use_col].mean()
            row[f"{col}_std"] = window[use_col].std()
            row[f"{col}_min"] = window[use_col].min()
            row[f"{col}_max"] = window[use_col].max()

        row[LABEL_COL] = window[LABEL_COL].mode().iloc[0]

        rows.append(row)

    return pd.DataFrame(rows)


def create_windows_parallel(df, window_name, window_seconds, split_name):
    print(f"\nStarting {split_name} {window_name} windowing...")

    grouped = list(df.groupby(SUBJECT_COL))
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                create_subject_windows,
                subject,
                group,
                window_seconds,
                STRIDE
            ): subject
            for subject, group in grouped
        }

        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            subject = futures[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)

                print(
                    f"[{split_name} {window_name}] "
                    f"Finished {subject} "
                    f"({completed}/{total}) - {len(result)} windows"
                )

            except Exception as e:
                print(f"Error processing {subject}: {e}")

    if results:
        output = pd.concat(results, ignore_index=True)
    else:
        output = pd.DataFrame()

    print(
        f"Finished {split_name} {window_name}: "
        f"{len(output)} total windows"
    )

    return output


def save_outputs(train_windows, test_windows, window_name):
    if ALL_PARTICIPANTS:
        train_path = OUTPUT_DIR / f"train_{window_name}_all.csv"
        test_path = OUTPUT_DIR / f"test_{window_name}_all.csv"

        train_windows.to_csv(train_path, index=False)
        test_windows.to_csv(test_path, index=False)

        print(f"Saved {train_path}")
        print(f"Saved {test_path}")

    else:
        for subject in train_windows[SUBJECT_COL].unique():
            train_subject = train_windows[
                train_windows[SUBJECT_COL] == subject
            ]

            test_subject = test_windows[
                test_windows[SUBJECT_COL] == subject
            ]

            train_path = OUTPUT_DIR / f"train_{window_name}_{subject}.csv"
            test_path = OUTPUT_DIR / f"test_{window_name}_{subject}.csv"

            train_subject.to_csv(train_path, index=False)
            test_subject.to_csv(test_path, index=False)

            print(f"Saved {train_path}")
            print(f"Saved {test_path}")


# --------------------------------
# Main
# --------------------------------
print("Reading train/test CSVs...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Using {MAX_WORKERS} worker threads.")

if ALL_PARTICIPANTS:
    train, test = compute_zscores(train, test, SIGNAL_COLS)

for window_name, window_seconds in WINDOWS.items():
    train_windows = create_windows_parallel(
        train,
        window_name,
        window_seconds,
        split_name="train"
    )

    test_windows = create_windows_parallel(
        test,
        window_name,
        window_seconds,
        split_name="test"
    )

    save_outputs(train_windows, test_windows, window_name)

print("\nFinished all windowing.")