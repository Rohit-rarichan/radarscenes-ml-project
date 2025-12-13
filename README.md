# radarscenes-ml-project
# Project Structure
# Machine Learning Pipeline for Radar Object Classification
# This project builds a complete end-to-end classical machine learning pipeline for the RadarScenes dataset, focusing on:
# Data preprocessing & cleaning
# Feature engineering
# Dataset balancing
# Model training (LogReg, SVM, RF, KNN)
# Evaluation & confusion matrices
# Basic radar-to-camera projection
# Exploratory visualizations (range/azimuth, RCS/Doppler, class distributions)
```bash
RadarScenes-ml-project/
│
├── data/
│   ├── engineered_features.parquet
│   ├── train_balanced.parquet
│   ├── test_balanced.parquet
│
├── models/
│   ├── logreg.joblib
│   ├── svm.joblib
│   ├── rf.joblib
│   ├── knn.joblib
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── BEV.ipynb
│
├── src/
│   ├── loaders/
│   │   ├── load_camera.py
│   │   ├── load_dataset.py
│   ├── ml/
│   │   ├── feature_engineering.py
│   │   ├── prepare_balanced_dataset.py
│   │   ├── train_models.py
│   │   ├── evaluate_models.py
│   ├── projection/
│       ├── load_calibration.py
│       ├── radar_to_camera.py
│
├── results/
│   ├── confusion_matrices/
│       ├── logreg.png
│       ├── svm.png
│       ├── rf.png
│       ├── knn.png
│
├── environment.yml
└── README.md
```
