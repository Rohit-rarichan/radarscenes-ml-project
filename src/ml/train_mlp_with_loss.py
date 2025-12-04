from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score


def load_data(path):
    df = pd.read_parquet(path)

    # sample for speed — adjust as needed
    df = df.sample(n=20000, random_state=42)

    X = df.drop(columns=["label_id","sequence"])
    y = df["label_id"]
    return X, y


def plot_rf_learning_curve(model, X, y, name="rf"):
    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    # Compute means
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # Dark theme for slide
    plt.style.use("dark_background")
    sns.set_palette("colorblind")

    plt.figure(figsize=(12, 7))

    plt.plot(train_sizes, train_mean, "o-", linewidth=2,
             label="Training Accuracy")
    plt.plot(train_sizes, val_mean, "o-", linewidth=2,
             label="Validation Accuracy")

    plt.title("Learning Curve – Random Forest", fontsize=18, weight="bold")
    plt.xlabel("Training Samples", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    Path("results/learning_curves").mkdir(parents=True, exist_ok=True)
    out_path = "results/learning_curves/rf_learning_curve.png"

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")


def main():
    # Load your best RF model
    model = joblib.load("models/rf.joblib")

    # Load data (use your engineered_features parquet)
    X, y = load_data(
        Path("/home/rraricha/RadarScenes-ml-project/data/engineered_features.parquet")
    )

    plot_rf_learning_curve(model, X, y)


if __name__ == "__main__":
    main()
