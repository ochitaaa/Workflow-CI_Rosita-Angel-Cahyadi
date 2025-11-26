import pandas as pd
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import mlflow

def main(args):
    # MLflow autolog
    mlflow.autolog()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(args.experiment_name)

    # Baca dataset dari argumen
    df = pd.read_csv(args.data_path)

    X = df.drop(columns=[args.target_col, args.is_train_col])
    y = df[args.target_col]

    train_mask = df[args.is_train_col] == 1
    test_mask = df[args.is_train_col] == 0

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    print("Training class distribution:\n", y_train.value_counts())
    print("\nTest class distribution:\n", y_test.value_counts())

    with mlflow.start_run(run_name="RandomForest_Diabetes_Autolog"):
        rf_model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight=args.class_weight
        )
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy (test):", acc)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

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
