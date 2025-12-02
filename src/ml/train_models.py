from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    #df = df.sample(n=100000, random_state=42)   # 300k rows instead of 21 million
    X = df.drop(columns=["label_id", "sequence"])
    y = df["label_id"]
    return X, y


def train_and_tune_models(X, y):
    models = {
        "logreg": (
            LogisticRegression(max_iter=500),
            {"clf__C":[0.1,1,10]}
        ),
        "svm": (
            SVC(),
            {    "clf__C":[0.1,1,10],
             "clf__gamma":["scale","auto"] }
        ),
        "rf": (
            RandomForestClassifier(),
            {"clf__n_estimators":[100,200],
             "clf__max_depth":[10,20,None]}
        ),
        "knn": (
            KNeighborsClassifier(),
            {"clf__n_neighbors":[3,5,7]}
        ),
    }

    results = {}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    Path("models").mkdir(exist_ok=True)

    for name,(model,params) in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])

        grid = GridSearchCV(pipe, params, cv=kfold,
                            scoring="f1_macro",
                            n_jobs=-1)

        print(f"\nTraining {name.upper()}...")
        grid.fit(X, y)

        print(f"Best {name}: {grid.best_score_:.4f}")
        print("Best params:", grid.best_params_)

        results[name] = grid.best_estimator_

        joblib.dump(grid.best_estimator_, f"models/{name}.joblib")

    return results


def main():
    parquet_path = Path("data/train_balanced.parquet")

    X, y = load_data(parquet_path)
    train_and_tune_models(X, y)

    print("\nAll models trained and saved to /models/")


if __name__ == "__main__":
    main()
