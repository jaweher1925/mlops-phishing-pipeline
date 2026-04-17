# MLOps Phishing Email Classification Pipeline

![GitHub Actions](https://github.com/jaweher1925/mlops-phishing-pipeline/actions/workflows/ml-pipeline.yml/badge.svg)

## Overview
An end-to-end MLOps pipeline for phishing email classification built with DVC, MLflow, FastAPI, Docker, Kubernetes, and GitHub Actions.

**Champion model:** Logistic Regression — F1 Score: **0.9822**

## Live Demo
- **Hugging Face Gradio App:** https://jaweher07-phishing-classifier.hf.space
- **GitHub Repository:** https://github.com/jaweher1925/mlops-phishing-pipeline

## Dataset
- **Source:** AVN Phishing Email Classification Dataset (Kaggle)
- **Size:** 60,000 emails — Legitimate (31,122) · Phishing (28,476) · Garbage (402)
- **License:** CC BY 4.0

## Pipeline Stages
| Stage | Tool | Output |
|-------|------|--------|
| Data versioning | DVC | dvc.lock |
| Preprocessing | Python | train.csv / test.csv |
| Training | MLflow + sklearn | champion_model.pkl |
| API serving | FastAPI | POST /predict |
| Containerization | Docker | phishing-classifier:latest |
| Orchestration | Kubernetes | Pod running on :30800 |
| CI/CD | GitHub Actions | Auto pipeline on push |
| Public deployment | Hugging Face | Live HTTPS URLs |

## Model Results
| Model | F1 Score | Status |
|-------|----------|--------|
| Logistic Regression | 0.9822 | Champion |
| Random Forest | 0.9440 | |
| AdaBoost | 0.9138 | |

## How to Run
### 1. Clone and setup
```bash
git clone https://github.com/jaweher1925/mlops-phishing-pipeline.git
cd mlops-phishing-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run pipeline
```bash
dvc repro
```

### 3. Start API locally
```bash
uvicorn api.main:app --reload
```

### 4. Run with Docker
```bash
docker build -t phishing-classifier .
docker run -p 8000:8000 phishing-classifier
```

## Author
**Jaweher Hichri** — Grand Valley State University — MLOps Spring 2026
