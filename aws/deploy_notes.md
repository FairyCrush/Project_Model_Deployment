# AWS Deployment Notes

Panduan lengkap deployment pipeline & web app ke AWS + Streamlit Cloud.

---

## Bagian 1 — Setup GitHub Repo (Wajib sebelum Streamlit Cloud)

### 1.1 Setup git config (cuma sekali per komputer)

```powershell
git config --global user.name "Nama Lo"
git config --global user.email "email-github-lo@example.com"
```

Email harus sama dengan email terverifikasi di GitHub Settings → Emails (biar commit muncul di contribution graph).

### 1.2 Init repo lokal

```powershell
cd "C:\Model Deployment\NIM_ModelDeployment"
git init
git add .
git status
```

Verify `.gitignore` jalan — pastiin `venv/`, `mlruns/`, `screenshots/` ga ke-add.

### 1.3 First commit

```powershell
git commit -m "Initial commit: credit score model deployment (Fase 1-3)"
```

### 1.4 Create GitHub repo + push

1. Buka https://github.com/new
2. Repo name: `credit-score-deployment` (atau sesuai preferensi)
3. Visibility: **Public** (wajib biar Streamlit Cloud bisa akses gratis)
4. **JANGAN** centang "Initialize with README" (karena udah ada README lokal)
5. Klik **Create repository**
6. Copy commands dari halaman "push existing repository", contoh:

```powershell
git remote add origin https://github.com/USERNAME/credit-score-deployment.git
git branch -M main
git push -u origin main
```

7. Cek di browser — semua file harus muncul, kecuali yang di `.gitignore`.

---

## Bagian 2 — Streamlit Community Cloud (Soal 2b)

### 2.1 Login Streamlit Cloud

1. Buka https://share.streamlit.io
2. Login pakai akun GitHub anda (klik "Continue with GitHub")
3. Authorize Streamlit access

### 2.2 Deploy app

1. Klik **"Create app"** → **"Deploy a public app from GitHub"**
2. Fill in:
   - **Repository**: `USERNAME/credit-score-deployment`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL** (custom): `nim-credit-score` (sesuain — bakal jadi `nim-credit-score.streamlit.app`)
3. Klik **"Advanced settings"**:
   - Python version: **3.12**
4. Klik **"Deploy"**

### 2.3 Wait for build

Build pertama makan 3-5 menit (install dependencies). Cek logs di kanan kalo ada error.

### 2.4 Test deployed app

URL: `https://nim-credit-score.streamlit.app` (atau sesuai custom URL lo)

Ulangi 3 test case (Good/Standard/Poor) di sini, screenshot semua, save di:
`screenshots/streamlit_cloud/01_test_good.png`, etc.

---

## Bagian 3 — AWS Pipeline (Soal 2a, 20 poin)

Pendekatan: **S3 untuk storage + EC2/SageMaker Studio untuk training**.

### 3.1 Setup AWS account

1. Login ke https://aws.amazon.com/console
2. Pakai akun AWS Educate / Free Tier kalau ada
3. Pilih region: **ap-southeast-1** (Singapore — paling deket Indonesia)

### 3.2 Install AWS CLI lokal + konfigurasi credentials

```powershell
pip install awscli boto3
aws configure
```

Input:
- AWS Access Key ID: (dari IAM > Users > Security credentials)
- AWS Secret Access Key
- Default region: `ap-southeast-1`
- Default output format: `json`

### 3.3 Create S3 bucket + upload data

```powershell
cd "C:\Model Deployment\NIM_ModelDeployment"
.\venv\Scripts\Activate.ps1
pip install boto3
python aws\pipeline_aws.py upload --bucket nim-credit-score-pipeline --region ap-southeast-1
```

Verify di AWS Console > S3: bucket `nim-credit-score-pipeline` harus berisi `data/data_C.csv` + 4 file di `models/`.

### 3.4 Run pipeline training di cloud

Pilih salah satu approach:

**Option A — SageMaker Studio Notebook (Recommended buat presentasi)**

1. AWS Console > SageMaker > **Studio**
2. Open Studio (create domain kalo belum ada)
3. New Notebook → Python 3 kernel
4. Clone repo GitHub lo lewat terminal:
   ```bash
   git clone https://github.com/USERNAME/credit-score-deployment.git
   cd credit-score-deployment
   pip install -r requirements.txt boto3
   ```
5. Download data dari S3:
   ```bash
   python aws/pipeline_aws.py download --bucket nim-credit-score-pipeline
   ```
6. Run pipeline:
   ```bash
   python pipeline/main_pipeline.py
   ```
7. Upload trained artifacts balik ke S3:
   ```bash
   python aws/pipeline_aws.py upload --bucket nim-credit-score-pipeline
   ```
8. Screenshot SageMaker Studio dengan output log → `screenshots/aws/sagemaker_training.png`

**Option B — EC2 t2.micro (Free tier)**

1. AWS Console > EC2 > Launch Instance
2. AMI: Amazon Linux 2023 / Ubuntu 22.04
3. Instance type: **t2.micro** (Free tier eligible)
4. Key pair: create new, download .pem
5. Security group: allow SSH (port 22) from your IP
6. Launch
7. SSH ke instance:
   ```bash
   ssh -i your-key.pem ec2-user@ec2-public-ip
   ```
8. Inside EC2:
   ```bash
   sudo yum install -y git python3-pip
   git clone https://github.com/USERNAME/credit-score-deployment.git
   cd credit-score-deployment
   pip3 install -r requirements.txt boto3
   aws configure  # input credentials
   python3 aws/pipeline_aws.py download --bucket nim-credit-score-pipeline
   python3 pipeline/main_pipeline.py
   python3 aws/pipeline_aws.py upload --bucket nim-credit-score-pipeline
   ```
9. Screenshot terminal dengan training output → `screenshots/aws/ec2_training.png`

### 3.5 Verify model di S3

AWS Console > S3 > `nim-credit-score-pipeline` > `models/` → cek timestamp updated.

Screenshot S3 console showing bucket contents → `screenshots/aws/s3_artifacts.png`

---

## Bagian 4 — Test Deployed App di Cloud

### 4.1 Buka URL Streamlit Cloud
`https://nim-credit-score.streamlit.app`

### 4.2 Run 3 test case
- Klik **Good** → klik Predict → screenshot full page
- Klik **Standard** → klik Predict → screenshot
- Klik **Poor** → klik Predict → screenshot

Save semua di `screenshots/streamlit_cloud/`.
