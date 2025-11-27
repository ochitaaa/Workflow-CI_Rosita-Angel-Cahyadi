import argparse
import os
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

def main(args):
    # Setup MLflow
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Random Forest Diabetes Classification")
    mlflow.autolog()

    # Load dataset
    df = pd.read_csv(args.data_path)
    target = "Diabetic"
    is_train_col = "is_train"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    # Split data using is_train column
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
            X, y, test_size=0.2, random_state=42
        )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Simpan model ke MLflow (untuk build-docker)
    mlflow.sklearn.log_model(model, "model")
    
    # Salin model ke Docker
    import shutil
    run_id = mlflow.active_run().info.run_id
    model_artifact_path = os.path.join("mlruns", "0", run_id, "model")
    os.makedirs(model_artifact_path, exist_ok=True)
    
    # Salin model.pkl ke folder model
    shutil.copy("artifacts/model.pkl", os.path.join(model_artifact_path, "model.pkl"))
    
    # Buat file MLmodel minimal
    with open(os.path.join(model_artifact_path, "MLmodel"), "w") as f:
        f.write("""
artifact_path: model
flavors:
  python_function:
    loader_module: mlflow.sklearn
    python_version: 3.10
  sklearn:
    pickled_model: model.pkl
    sklearn_version: 1.6.1
""")
