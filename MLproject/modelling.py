import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
import numpy as np
import os
import sys
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Validasi jumlah argumen
    if len(sys.argv) < 4:
        raise ValueError("Harus menyediakan 3 argumen: n_estimators, max_depth, dataset")
    
    n_estimators = int(sys.argv[1])
    max_depth_str = sys.argv[2]
    file_path = sys.argv[3]

    # Proses max_depth
    if max_depth_str.lower() == "null":
        max_depth = None
    else:
        max_depth = int(max_depth_str)

    # Baca data
    data = pd.read_csv(file_path)

    # Pisahkan fitur dan target
    data['is_train'] = data['is_train'].astype(int)
    train_mask = data['is_train'] == 1
    test_mask = data['is_train'] == 0

    X_train = data[train_mask].drop(columns=['Diabetic', 'is_train'])
    X_test = data[test_mask].drop(columns=['Diabetic', 'is_train'])
    y_train = data[train_mask]['Diabetic']
    y_test = data[test_mask]['Diabetic']

    with mlflow.start_run():
        # Latih model dengan class_weight='balanced'
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # probabilitas kelas 'yes'

        # Log metrik
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)

        # Log metrik tambahan
        # 1. Precision-Recall AUC
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_test_enc = le.fit_transform(y_test)
        precision, recall, _ = precision_recall_curve(y_test_enc, y_pred_proba)
        pr_auc = auc(recall, precision)
        mlflow.log_metric("pr_auc", pr_auc)

        # 2. F1-score per class
        f1_per_class = f1_score(y_test, y_pred, average=None, labels=['no', 'yes'])
        mlflow.log_metric("f1_no", f1_per_class[0])
        mlflow.log_metric("f1_yes", f1_per_class[1])

        # Log parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth if max_depth is not None else "None")
        mlflow.log_param("class_weight", "balanced")

        # Log model
        input_example = X_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"Model trained with accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}, PR-AUC: {pr_auc:.4f}")
