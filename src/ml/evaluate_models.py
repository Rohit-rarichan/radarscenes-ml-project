from pathlib import Path
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report


def load_data(parquet_path: Path, eval_samples = 50000):
    df = pd.read_parquet(parquet_path)
    if len(df) > eval_samples:
        df = df.sample(n = eval_samples, random_state = 42)
    drop_cols = [c for c in ["label_id", "sequence"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label_id"]
    return X, y


def plot_confusion(cm, labels, name):
    Path("results/confusion_matrices").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix – {name.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    out_path = f"results/confusion_matrices/{name}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix → {out_path}")


def main():
    parquet_path = Path("data/test_balanced.parquet")

    X, y = load_data(parquet_path)

    model_names = ["logreg","svm","rf","knn"]
    labels = sorted(y.unique())

    for name in model_names:
        print(f"\n----- Evaluating {name.upper()} -----")
        model = joblib.load(f"models/{name}.joblib")

        preds = model.predict(X)

        print(classification_report(y, preds))

        cm = confusion_matrix(y, preds)
        plot_confusion(cm, labels, name)


if __name__ == "__main__":
    main()
