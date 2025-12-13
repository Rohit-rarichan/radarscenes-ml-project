# radarscenes-ml-project
## Project Structure
Machine Learning Pipeline for Radar Object Classification

This project builds a complete end-to-end classical machine learning pipeline for the **RadarScenes dataset**, focusing on:
- Data preprocessing & cleaning
- Feature engineering
- Dataset balancing
- Model training (LogReg, SVM, RF, KNN)
- Evaluation & confusion matrices
- Basic radar-to-camera projection
- Exploratory visualizations (range/azimuth, RCS/Doppler, class distributions)
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
## Environment Setup
Create Conda Environment
<pre>
conda env create -f environment.yml
conda activate radarscenes
</pre>
Install extra dependencies
<pre>
pip install seaborn joblib matplotlib pandas scikit-learn pillow
</pre>
## Dataset Description
RadarScenes contains:
- 21M radar detections
- Multiple sensors (camera + 4 radar units)
- Per-point labels (car, pedestrian, cyclist, truck, background, etc.)
- High-resolution timestamps and odometry data

Where we downloaded the dataset:
[RadarScenes](https://zenodo.org/records/4559821)

## Pipeline Overview

### 1.Data Loading

We parse raw radar detections for each sequence, including:
- range_sc: scaled radar range
- azimuth_sc: scaled angle
- vr: Doppler velocity
- rcs: radar cross-section
- label_id: object category

### 2.Feature Engineering

We compute additional physical features:
<pre>
| Feature      | Description            |
| ------------ | ---------------------- |
| `x, y`       | Cartesian coordinates  |
| `distance`   | sqrt(x² + y²)          |
| `angle_deg`  | human-friendly azimuth |
| `speed_abs`  | absolute speed         |
| `power_norm` | normalized RCS         |
</pre>
These help classical ML models learn geometric structure.

### 3.Data Cleaning

We remove:
- invalid radar readings
- extreme noise (range > 120m, |vr| > 150, RCS outside physical range)

### 4.Class Rebalancing

RadarScenes is extremely imbalanced — e.g.:
- Background ≈ 80%+
- Vehicles ≈ 15%
- All others < 5%

We use undersampling to create:
- train_balanced.parquet (5,000 samples per class)
- test_balanced.parquet (1,000 samples per class)

### 5.Model Training
We evaluate four classical ML models:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- KNN

All wrapped in Pipeline(StandardScaler → Classifier).

### 6.Evaluation

For each model, we compute:
- Confusion matrix
- Precision, recall, F1-score (macro & weighted)
- Overall accuracy

## Model Performance Summary
### Logistic Regression
- Accuracy: ~33%
- Handles linear decision boundaries, limited expressive power.
### SVM
- Accuracy: ~59%
- Strong separation power but slower on large datasets.
### KNN
- Accuracy: ~74%
- Captures non-linear boundaries but sensitive to noise.
### Random Forest (Best Model)
- Accuracy: ~85%
- F1-macro: ~0.85
- Handles non-linear features & noise very well

Almost perfect for background & large object categories.
```bash
results/confusion_matrices/
├── logreg.png
├── svm.png
├── rf.png
└── knn.png
```
## Key Insights From Data Exploration
### 1.Severe Class Imbalance
Most radar points belong to background (class 11) — over 550,000 detections in sampled data.
#### Distribution
<img width="857" height="397" alt="image" src="https://github.com/user-attachments/assets/66a80497-a02d-4c6c-8836-f9912cc79305" />

### 2.Without Class 11, Cars Still Dominate
Even after removing background: Cars (class 1) massively outnumber smaller objects (pedestrians, cyclists).
#### Distribution
<img width="876" height="470" alt="image" src="https://github.com/user-attachments/assets/366ef2bb-0413-4669-9315-f2a7fcb11470" />

### 3.Spatial Structure of Radar Points
Features like: range, azimuth. vr, rcs show clear separability between classes.
#### Radar Polar View
<img width="684" height="547" alt="image" src="https://github.com/user-attachments/assets/02fadc7c-dbb3-481a-87c9-f29ae3a6e5cf" />

### 4.Doppler vs RCS Visualization
Cars generally have:
- stronger RCS
- smaller Doppler magnitude (relative motion differences)
Pedestrians have:
- weaker RCS
- larger Doppler variance
#### RCS vs Doppler
<img width="717" height="629" alt="image" src="https://github.com/user-attachments/assets/e5ef3540-df37-4c57-9fcd-d21fc80030ff" />

Bird’s-Eye View (BEV) Projection

To visualize radar spatial structure, all radar detections are transformed into a top-down Bird’s-Eye View (BEV) using: 
- range → radial distance from the sensor
- azimuth → angular direction
- 𝑥 = 𝑟cos(𝜃)
- 𝑦 = 𝑟sin(𝜃)

Where:
- r = radar range
- θ = radar azimuth
This produces a 2D map of where objects are located relative to the sensor

Why BEV?
- No camera calibration required
- Preserves geometric structure
- Reveals patterns between object types
- Standard approach in autonomous driving perception
<img width="1589" height="790" alt="image" src="https://github.com/user-attachments/assets/a456b827-1b43-4e89-a76f-f019520b315f" />

## Future Improvements
- Train a lightweight neural network (MLP or PointNet)
- Incorporate temporal features (radar sweeps over time)
- Explore sensor fusion with camera images
- Replace undersampling with SMOTE or class-weighted losses

## Acknowledgments
- RadarScenes Dataset Creators
- UCI (ICS 171) Staff
- OpenRadar and autonomous driving research community
