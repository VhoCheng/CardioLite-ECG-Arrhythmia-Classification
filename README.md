# CardioLite: A Lightweight and Interpretable ECG Arrhythmia Classification Pipeline on the MIT-BIH Database

## Overview
CardioLite is a lightweight and interpretable machine learning pipeline for ECG arrhythmia classification using the MIT-BIH Arrhythmia Database.

This project focuses on a CPU-friendly and reproducible workflow for heartbeat-level arrhythmia classification. Instead of relying on large deep learning models, CardioLite uses handcrafted time-domain and frequency-domain features together with classical machine learning models to provide an efficient baseline for medical AI research.

The project is especially suitable for:
- lightweight biomedical AI experimentation
- interpretable ECG classification studies
- CPU-only environments
- reproducible student research projects

---

## Project Title
**CardioLite: A Lightweight and Interpretable ECG Arrhythmia Classification Pipeline on the MIT-BIH Database**

---

## Dataset
This project uses the **MIT-BIH Arrhythmia Database**, a widely used public benchmark for ECG arrhythmia classification.

- Source: PhysioNet
- Access link: https://physionet.org/content/mitdb/1.0.0/

The dataset can be automatically downloaded using the `wfdb` Python package during preprocessing.

---

## Methods
The complete workflow includes the following steps:

1. **ECG beat segmentation** using annotated R-peaks  
2. **Feature extraction** from each heartbeat segment  
   - time-domain statistics  
   - frequency-domain energy features  
3. **Classical machine learning training**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
4. **Model evaluation**
   - Accuracy
   - Macro-F1
   - One-vs-Rest ROC-AUC
5. **Interpretability analysis**
   - ROC visualization
   - confusion matrix analysis
   - feature importance analysis

---

## Repository Structure

```text
CardioLite-ECG-Arrhythmia-Classification/
│
├── 01_build_dataset.py
├── 02_train_eval.py
├── 03_plot_main_results.py
├── 04_feature_and_spectral_analysis.py
├── README.md
├── requirements.txt
│
└── outputs/
    ├── figures/
    │   ├── fig_roc.png
    │   └── fig_confusion_matrix.png
    │
    └── tables/
        └── results.csv
