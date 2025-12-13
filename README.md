# radarscenes-ml-project
# Project Structure
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
