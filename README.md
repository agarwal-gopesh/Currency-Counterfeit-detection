# 🏦 AuthentiNote – Indian Currency Counterfeit Detection

> An end-to-end Deep Learning and MLOps project for detecting counterfeit Indian currency notes using Computer Vision, OCR, and modern MLOps practices.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-purple)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📌 Overview

**AuthentiNote** is an AI-powered counterfeit currency detection system designed for Indian banknotes.

The application combines **Deep Learning** with **Optical Character Recognition (OCR)** to verify the authenticity of uploaded currency images.

Instead of relying solely on a CNN model, the system first performs OCR-based text validation and then passes the image to a deep learning classifier for final prediction, making the overall detection pipeline more robust.

---

# 🚀 Features

- ✅ Counterfeit Currency Detection
- ✅ OCR-based Text Verification
- ✅ Deep Learning Classification
- ✅ Streamlit Web Interface
- ✅ Modular Project Structure
- ✅ Data Versioning using DVC
- ✅ Experiment Tracking Ready
- ✅ Automated CI using GitHub Actions
- ✅ Docker-ready Project Structure
- ✅ JSON Prediction Reports
- ✅ Industry-style MLOps Pipeline

---

# 🧠 Model

The image classifier is built using **Transfer Learning**.

### Architecture

- MobileNetV2
- TensorFlow / Keras
- Binary Classification
- Image Augmentation
- Fine Tuning

**Model Accuracy:** **97%**

---

# 🔍 OCR Verification

Before classification, the application performs OCR using **EasyOCR**.

The OCR module:

- Extracts text from uploaded notes
- Detects suspicious keywords frequently found on fake/sample notes
- Verifies genuine RBI-related text
- Improves robustness by filtering obviously fake inputs before CNN inference

---

# 🛠️ Tech Stack

## Programming

- Python

## Deep Learning

- TensorFlow
- Keras

## OCR

- EasyOCR
- PyTorch

## Data Processing

- NumPy
- Pandas
- Pillow

## Machine Learning

- Scikit-learn

## Web Application

- Streamlit

## Configuration

- YAML

## Version Control

- Git
- GitHub

## MLOps

- DVC
- GitHub Actions
- MLflow 
- Docker ready

---

# 📂 Project Structure

```text
currency-counterfeit-detection/

│
├── artifacts/
├── data/
├── notebooks/
├── reports/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
│
├── dvc.yaml
├── params.yaml
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ MLOps Pipeline

The project follows a modular MLOps workflow.

```
Data
   │
   ▼
Preprocessing
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Saved Model
   │
   ▼
OCR Validation
   │
   ▼
Prediction
   │
   ▼
Streamlit App
```

---

# 📦 Data Versioning

The project uses **DVC (Data Version Control)** for:

- Dataset versioning
- Model artifact tracking
- Pipeline reproducibility
- Reproducible ML workflow

---

# 📈 Experiment Tracking

The project is designed to support **MLflow** for experiment tracking.

It enables:

- Parameter logging
- Metrics logging
- Model artifact management
- Experiment comparison

---

# 🔄 Continuous Integration

The repository includes a **GitHub Actions** workflow that automatically:

- Installs dependencies
- Runs project tests
- Validates code
- Ensures build consistency

This provides an automated CI pipeline for every push.

---

# 🐳 Docker

The project structure is Docker-ready and can be containerized for deployment.

Due to the combined dependency footprint of **TensorFlow** and **EasyOCR (PyTorch)**, deployment on several free-tier hosting platforms is constrained by memory limitations.

---

# 💻 Web Application

Users can:

- Upload an Indian currency image
- Perform OCR verification
- Run deep learning inference
- Receive authenticity prediction
- View prediction confidence
- Download prediction report

---

# 📚 Libraries Used

- TensorFlow
- Keras
- EasyOCR
- PyTorch
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Pillow
- PyYAML
- DVC

---

# 🎯 Future Improvements

- Multi-denomination classification
- Security thread detection
- Serial number validation
- Cloud deployment
- REST API
- Kubernetes deployment

---

# 👨‍💻 Author

## Gopesh Agarwal

AI / ML Engineer | Deep Learning | MLOps

GitHub:

https://github.com/agarwal-gopesh/Currency-Counterfeit-detection

LinkedIn:

https://www.linkedin.com/in/gopesh-agarwal-81a744378/

---

⭐ If you found this project interesting, consider giving it a star!
