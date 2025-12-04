from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier


def load_data(path):
    df = pd.read_parquet(path)

    # sample a subset for speed
    df = df.sample(n=20000, random_state=42)

    X = df.drop(columns=["label_id","sequence"])
    y = df["label_id"]
    return X, y


def compute_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    return (
        train_sizes,
        train_scores.mean(axis=1),
        val_scores.mean(axis=1)
    )


def plot_depth_comparison(X, y):
    # Two random forest models
    rf_full = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    rf_limited = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    print("Computing learning curves...")

    ts_full, train_full, val_full = compute_learning_curve(rf_full, X, y)
    ts_lim,  train_lim,  val_lim  = compute_learning_curve(rf_limited, X, y)

    # Dark slide theme
    plt.style.use("dark_background")
    sns.set_palette("colorblind")

    plt.figure(figsize=(12, 7))

    # fully grown RF
    plt.plot(ts_full, train_full, "o-", linewidth=2, label="Train (depth=None)")
    plt.plot(ts_full, val_full, "o-", linewidth=2, label="Val (depth=None)")

    # depth-limited RF
    plt.plot(ts_lim, train_lim, "o--", linewidth=2, label="Train (depth=20)")
    plt.plot(ts_lim, val_lim, "o--", linewidth=2, label="Val (depth=20)")

    plt.title("Random Forest Depth Comparison", fontsize=18, weight="bold")
    plt.xlabel("Training Samples", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    Path("results/depth_comparison").mkdir(parents=True, exist_ok=True)
    out_path = "results/depth_comparison/rf_depth_comparison.png"

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    X, y = load_data(
        Path("/home/rraricha/RadarScenes-ml-project/data/engineered_features.parquet")
    )
    plot_depth_comparison(X, y)


if __name__ == "__main__":
    main()
