# Cyberbullying Detection for curaJOY Impact Fellowship

## 📌 Project Overview
This repository contains the implementation of a cyberbullying detection pipeline developed for curaJOY's Impact Fellowship Program. The project explores machine learning and AutoML techniques to classify online comments as either **cyberbullying** or **non-cyberbullying**, with special attention to context, sarcasm, and false positives.

## 🧠 Core Technologies
- **Python**
- **scikit-learn**
- **TPOT (AutoML)**
- **NLTK (NLP preprocessing)**
- **Streamlit** (for future annotation tool)

## 🛠️ Project Structure
├── cyberbullying_detection_svm_rbf_tpot_final.py # Main training + evaluation script
├── tpot_cyberbullying_pipeline_rbf.py # Auto-generated TPOT pipeline
├── training_log.txt # Log output from model training
├── raw_data.txt # Source data (not included in repo)
├── report.tex / report.pdf
└── README.md


## 🧪 Model Details
- **Baseline**: SVM with RBF kernel
- **Optimized**: Logistic Regression via TPOT (accuracy: ~88.7%)
- **Vectorization**: TF-IDF with 5000 max features

