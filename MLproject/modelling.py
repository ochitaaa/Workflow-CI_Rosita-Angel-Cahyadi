import argparse
import os
import pandas as pd
import pickle
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve


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

    # Model WITHOUT mlflow.start_run()
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
    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:\n", cm)

    # ===== SAVE ARTIFACTS =====
    os.makedirs("artifacts", exist_ok=True)

    # Save model
    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save dataset copy
    df.to_csv("artifacts/diabetes_dataset_2019_preprocessing.csv", index=False)

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/training_confusion_matrix.png")
    plt.close()

    # ROC Curve
    if len(set(y_test)) == 2:  # binary
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig("artifacts/training_roc_curve.png")
        plt.close()

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("artifacts/training_precision_recall_curve.png")
        plt.close()


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
