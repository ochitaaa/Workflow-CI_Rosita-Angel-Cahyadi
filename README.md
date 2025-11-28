## **Diabetes Classification dengan Workflow CI**

Proyek ini mengimplementasikan klasifikasi diabetes menggunakan algoritma Random Forest dengan integrasi MLflow dan GitHub Actions untuk *re-training* otomatis. Seluruh proses dari pelatihan model hingga deployment sebagai Docker image dilakukan secara otomatis melalui *Continuous Integration* (CI).
Dataset

Dataset yang digunakan adalah: `diabetes_dataset_2019_preprocessing.csv`

Struktur Proyek:
- `.github/workflows/main.yml` : workflow CI dan melakukan *re-training* otomatis.
- `MLproject`
  - `MLproject` : Konfigurasi MLflow Project.
  - `conda.yaml` : linkungan Conda untuk dependensi.
  - `diabetes_dataset_2019_preprocessing.csv` : dataset yang digunakan.
  - `modelling.py` : script pelatihan model

Model berhasil dibuat menjadi Docker Image dan diupload otomatis ke Docker Hub melalui GitHub Actions.
https://hub.docker.com/repository/docker/ochitaaa/mlflow-diabetes-model
