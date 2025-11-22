import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
pd.set_option('display.max_columns', None)

# Autolog
if "MLFLOW_EXPERIMENT_ID" in os.environ:
    del os.environ["MLFLOW_EXPERIMENT_ID"]
    
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Random Forest Credit Score Classification")
mlflow.autolog()

# Load Data
df = pd.read_csv("/Users/Shared/Files From d.localized/Rosita/DICODING ASAH 2025/Proyek Membangun Sistem Machine Learning/Eksperimen_SML_Rosita_Angel_Cahyadi/preprocessing/Credit Score Dataset_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop(columns=['Credit Score', 'is_train'])
y = df['Credit Score']

# Split menjadi train dan test berdasarkan kolom 'is_train'
train_mask = df['is_train'] == 1
test_mask = df['is_train'] == 0

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print("Training class distribution:\n", y_train.value_counts())
print("\nTest class distribution:\n", y_test.value_counts())

# Jalankan autolog
with mlflow.start_run(run_name="RandomForest_Autolog"):
    # Latih model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Hasil prediksi
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy (test):", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
