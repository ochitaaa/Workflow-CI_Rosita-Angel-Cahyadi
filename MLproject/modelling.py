import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow


def main(args):
    # Setup MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns")))
    mlflow.set_experiment(args.experiment_name)
    mlflow.autolog()

    # Load dataset
    df = pd.read_csv(args.data_path)

    target = args.target_col
    is_train_col = args.is_train_col

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    # Split data using is_train column if available
    if is_train_col in df.columns:
        train_mask = df[is_train_col] == 1
        test_mask = df[is_train_col] == 0

        X_train = df.loc[train_mask].drop(columns=[target, is_train_col])
        y_train = df.loc[train_mask][target]

        X_test = df.loc[test_mask].drop(columns=[target, is_train_col])
        y_test = df.loc[test_mask][target]

    else:
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state
        )

    # Model WITHOUT mlflow.start_run()  (important!!)
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        class_weight=(args.class_weight if args.class_weight.lower() != "none" else None)
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--target-col", type=str, default="Diabetic")
    parser.add_argument("--is-train-col", type=str, default="is_train")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--experiment-name", type=str, default="Random Forest Diabetes Classification")

    args = parser.parse_args()
    main(args)
