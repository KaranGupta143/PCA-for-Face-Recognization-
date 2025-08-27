## Implementation of PCA with ANN Algorithm for Face Recognition

This project implements a full pipeline for face recognition using Principal Component Analysis (PCA) for dimensionality reduction and a Multi-Layer Perceptron (ANN) classifier.

### Features
- PCA + ANN pipeline with grid search and cross-validation
- Adaptive handling for small datasets (caps PCA components and CV folds)
- Headless plotting, artifacts saved under `artifacts/`
- Quick-run mode for fast smoke tests
- Simple inference script to predict a single image

### Dataset
This repository expects the LFW deepfunneled dataset under:
`data/lfw-deepfunneled/lfw-deepfunneled/`

You can adjust the path in `pca_ann_faces.py` by changing `DATA_DIR` or by modifying the dataset location to match the above.

### Environment Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### Train
Quick run (faster, subset + smaller search):
```powershell
$env:QUICK_RUN="1"; .\.venv\Scripts\python pca_ann_faces.py
```

Full run (slower, broader search):
```powershell
$env:QUICK_RUN="0"; .\.venv\Scripts\python pca_ann_faces.py
```

Artifacts will be written to `artifacts/`:
- `pca_ann_faces_<timestamp>.joblib`
- `eigenfaces.png`, `originals.png`, `reconstructions.png`

### Inference on a single image
```powershell
.\.venv\Scripts\python predict.py --model artifacts\pca_ann_faces_<timestamp>.joblib --image "path\to\face.jpg"
```

### Notes
- This implementation is designed for clarity, reproducibility, and responsible use. It avoids evasive or deceptive behavior and follows good ML practices.
- Performance on very small per-class sample sizes will be limited; use more images per class or a larger training split for better accuracy.

