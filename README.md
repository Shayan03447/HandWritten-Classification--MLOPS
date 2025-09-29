# ✍️ Handwritten Digits Classification

A machine learning project for **handwritten digit recognition** using a modular pipeline, with **DVC** for reproducibility and **MLflow** for experiment tracking.  
Includes preprocessing, model training, evaluation, logging, and reporting.

---

## 📌 What is this project?

This project implements an **end-to-end ML workflow** to classify handwritten digits (0-9):  
- Load raw image data  
- Preprocess images (scaling, normalization)  
- Train models (e.g., Logistic Regression, Random Forest, or CNNs)  
- Evaluate using metrics (accuracy, confusion matrix)  
- Track experiments with **MLflow**  
- Reproduce the pipeline with **DVC**

---

## ❓ Why this project?

- Handwritten digit classification is a **classic ML problem** for image recognition.  
- Demonstrates a complete **ML pipeline workflow** with reproducibility.  
- Uses **MLflow** to track experiments and **DVC** to version data, models, and pipeline stages.  
- Logs and reports provide **traceability and audit-ready outputs**.

---

## ⚙️ How does it work?

### 🔑 Pipeline Steps:

1. **Data Ingestion & Preprocessing**
   - Load raw image dataset (`data/raw/`)  
   - Preprocess (reshape, normalize) and save to `data/preprocessed/`  

2. **Model Training**
   - Train models on preprocessed data using `src/train.py`  
   - Log metrics and artifacts to MLflow  

3. **Evaluation**
   - Evaluate model performance using `src/evaluate.py`  
   - Generate reports in `reports/`  
   - Save runtime logs in `logs/`

4. **Pipeline Reproduction**
   - Use DVC to reproduce the pipeline stages (`dvc.yaml`)  
   - Track experiments consistently

---

## 📂 Repository Structure

```
Handwritten-Digits-Classification/
│-- src/
│ ├── ingestion.py.py # Model training
│ ├── preprocess.py # Data preprocessing
│ └── model_building.py # Evaluation metrics
| └── evaluation.py
│
│-- data/
│ ├── raw/ 
│ └── preprocessed/
│
│-- mlflow/ # Experiment tracking files
│-- reports/ # Metrics, plots, visualizations
│-- logs/ # Runtime logs
│-- dvc.yaml # DVC pipeline stages
│-- requirements.txt
│-- README.md
│-- .gitignore
```

📊 Logging & Reports

- All logs saved in logs/
- Metrics and visualizations in reports/
- Experiments tracked via MLflow

## 💻Tech Stack

- Python 3.8+
- pandas, numpy
- scikit-learn / PyTorch / TensorFlow
- MLflow for experiment tracking
- DVC for pipeline orchestration
- Matplotlib / Seaborn for visualizations