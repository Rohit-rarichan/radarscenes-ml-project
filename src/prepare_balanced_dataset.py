from pathlib import Path
import pandas as pd

def load_engineered():
    path = Path("data/engineered_features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Engineered parquet not found: {path}")
    print(f"Loading engineered dataset from: {path}")
    return pd.read_parquet(path)


def make_balanced(df, per_class=5000):
    """
    Build a balanced dataset with equal representation
    of all classes (label_id).
    """
    groups = []
    for label, g in df.groupby("label_id"):
        print(f" - Label {label}: {len(g)} rows available")

        # Sample exact per_class rows (replace=True if too small)
        if len(g) >= per_class:
            sampled = g.sample(n=per_class, random_state=42)
        else:
            sampled = g.sample(n=per_class, replace=True, random_state=42)

        groups.append(sampled)

    df_bal = pd.concat(groups).reset_index(drop=True)
    print(f"\nBalanced dataset created with {len(df_bal)} total rows.")
    return df_bal


def save_splits(df_bal):
    train = df_bal.sample(frac=0.7, random_state=42)
    test = df_bal.drop(train.index)

    Path("data").mkdir(parents=True, exist_ok=True)

    train.to_parquet("data/train_balanced.parquet", index=False)
    test.to_parquet("data/test_balanced.parquet", index=False)

    print("\nSaved balanced datasets:")
    print(" ✔ data/train_balanced.parquet")
    print(" ✔ data/test_balanced.parquet")


def main():
    print("Loading dataset...")
    df = load_engineered()

    print("\nBalancing dataset...")
    df_bal = make_balanced(df, per_class=5000)

    print("\nSaving splits...")
    save_splits(df_bal)

    print("\nDone! Balanced data is ready for training & evaluation.")


if __name__ == "__main__":
    main()
