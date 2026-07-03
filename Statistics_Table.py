import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

subjects = ["S2", "S3", "S4"]

print("\n=== Generating Per-Subject Confusion Matrices ===")

for s in subjects:

    file_path = f"rf_predictions_{s}.csv"
    print(f"\nLoading {file_path}")

    try:
        pred = pd.read_csv(file_path)

        y_true = pred["y_true"]
        y_pred = pred["y_pred"]

        cm = confusion_matrix(y_true, y_pred, labels=[1,2,3])

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Neutral", "Stress", "Amusement"]
        )

        plt.figure(figsize=(6,5))

        disp.plot(cmap="Blues", values_format="d")

        plt.title(f"Random Forest Confusion Matrix - {s}")

        plt.tight_layout()
        plt.subplots_adjust(left=0.25)

        plt.savefig(f"{s}_RF_confusion_matrix.png", dpi=300, bbox_inches="tight")

        plt.close()

        

    except FileNotFoundError:
        print(f"Missing file: {file_path}")