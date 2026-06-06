# Project Model Deployment — Model Deployment

Final project Model Deployment.

- **Dataset**: C (Genap, Laki-laki) — `data/data_C.csv` (25.000 nasabah × 28 kolom)
- **Target**: `Credit_Score` (Poor / Standard / Good)
- **Best model**: RandomForest (test F1-macro 0.7245, Accuracy 0.7426)
- **NIM**: 2802526252

## Struktur project

```
NIM_ModelDeployment/
├── notebook/
│   └── 01_eda_modeling.ipynb       # EDA + multi-model experiment
├── pipeline/
│   ├── preprocessing.py            # DataPreprocessor class
│   ├── training.py                 # ModelTrainer class
│   ├── evaluation.py               # ModelEvaluator class
│   └── main_pipeline.py            # Orchestrator + MLflow tracking
├── models/
│   ├── best_model.pkl              # Trained RandomForest (compressed)
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
├── deployment/
│   ├── app.py                      # Streamlit web app
│   ├── inference.py                # CreditScorePredictor class
│   └── requirements.txt
├── aws/
│   ├── pipeline_aws.py             # S3 upload/download + cloud training
│   └── deploy_notes.md             # Step-by-step AWS + Streamlit Cloud guide
├── screenshots/                    # Test case + MLflow screenshots
├── data/
│   └── data_C.csv
├── streamlit_app.py                # Entry point untuk Streamlit Cloud
├── requirements.txt                # Top-level deps (untuk Streamlit Cloud)
├── packages.txt                    # System packages (untuk Streamlit Cloud)
├── .gitignore
└── README.md
```

## Quick Start (Local)

### Setup

```powershell
cd "C:\Model Deployment\NIM_ModelDeployment"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Train pipeline

```powershell
python pipeline\main_pipeline.py
```

Akan train 4 model (logreg, rf, xgb, lgbm), log ke MLflow, save best model ke `models/`.

### Lihat MLflow tracking UI

```powershell
mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db"
```

Buka http://localhost:5000

### Run Streamlit app

```powershell
streamlit run deployment\app.py
```

Buka http://localhost:8501

## Hasil Eksperimen Multi-Model

| Model | Test Accuracy | Test F1-macro |
|---|---|---|
| **RandomForest** | **0.7426** | **0.7245** (best) |
| XGBoost | 0.7332 | 0.7120 |
| LightGBM | 0.7294 | 0.7094 |
| LogisticRegression | 0.6540 | 0.6164 |

## Deployment

- **Local**: `streamlit run deployment\app.py` (Fase 3)
- **Cloud public**: Streamlit Community Cloud (Fase 4 — lihat `aws/deploy_notes.md`)
- **AWS pipeline**: S3 + EC2 / SageMaker Studio (Fase 4 — lihat `aws/deploy_notes.md`)

## Comparison Local vs Cloud

### Local advantages
- **Cepat iterasi** — ga ada network latency, tinggal Ctrl+S, run
- **Gratis** — pakai resource laptop sendiri, ga ada billing
- **Data privacy** — data sensitif (kredit nasabah) ga keluar dari device
- **Full control** — bebas pilih versi library, kernel, hardware
- **Offline ready** — bisa kerja tanpa internet

### Local disadvantages
- **Ga scalable** — terbatas hardware lokal (RAM, CPU, disk)
- **Susah sharing** — orang lain ga bisa akses tanpa setup ulang
- **Single-user** — cuma satu orang yang bisa pakai sekaligus
- **Ga 24/7** — laptop dimatiin = app mati

### Cloud advantages
- **Scalable** — auto-scaling resource sesuai traffic
- **Public access** — siapapun bisa akses lewat URL
- **24/7 uptime** — managed by provider
- **Collaboration** — tim bisa akses concurrent
- **Managed infra** — ga pusing OS update, security patches
- **CI/CD friendly** — push GitHub → auto deploy

### Cloud disadvantages
- **Biaya** — EC2/SageMaker billable per jam, S3 per GB
- **Vendor lock-in** — kode AWS-specific susah pindah ke Azure/GCP
- **Latency** — tergantung region & koneksi user
- **Setup awal ribet** — IAM, VPC, security groups, dll
- **Data egress cost** — download data dari S3 dihitung

## Link Video Penjelasan
https://drive.google.com/file/d/1qvSVPNwzCRP9ddpDieHjKOSTiX8RxSwT/view?usp=sharing

## Catatan

Project ini dikerjakan sebagai final assignment Model Deployment.
Lihat `aws/deploy_notes.md` untuk panduan deployment lengkap.
