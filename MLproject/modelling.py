# -*- coding: utf-8 -*-
"""
modelling.py - Diabetes Classification for CI/CD
Model klasifikasi diabetes dengan MLflow (tanpa autolog)
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve, auc,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import sys
import os

def main():
    print("Memulai training model diabetes...\n")
    
    # Baca dataset dari argumen CLI
    if len(sys.argv) < 2:
        raise ValueError("Harap berikan path dataset sebagai argumen pertama.")
    dataset_path = sys.argv[1]
    
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Persiapan data: gunakan kolom 'is_train' untuk split
    train_mask = df['is_train'] == 1
    test_mask = df['is_train'] == 0

    X_train = df[train_mask].drop(columns=['Diabetic', 'is_train'])
    X_test = df[test_mask].drop(columns=['Diabetic', 'is_train'])
    y_train = df[train_mask]['Diabetic']
    y_test = df[test_mask]['Diabetic']
    
    print(f"Data split berdasarkan 'is_train': Train={len(y_train)}, Test={len(y_test)}")
    
    # Training model
    print("\n🔧 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # PR AUC
    le = LabelEncoder()
    y_test_enc = le.fit_transform(y_test)
    precision, recall, _ = precision_recall_curve(y_test_enc, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Log ke MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("pr_auc", pr_auc)
    
    # Log parameter
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", "None")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", len(y_test) / len(df))
    
    # Log model
    input_example = X_train.head(5)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    
    # Tambahkan tag
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("problem_type", "binary_classification")
    mlflow.set_tag("target", "Diabetic")
    
    print(f"\nHasil Evaluasi:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1-Macro: {f1_macro:.4f}")
    print(f"   PR-AUC   : {pr_auc:.4f}")
    print("\nModel berhasil dilatih dan disimpan di MLflow!")

if __name__ == "__main__":
    main()
