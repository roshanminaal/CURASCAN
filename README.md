# 🏥 CURASCAN — INTELLIGENT HEALTH CARE IMAGING

CURASCAN is a Streamlit-based medical imaging platform with AI-powered classification and segmentation. It supports chest X-ray pneumonia detection, MRI brain tumor ("Glioma", "Meningioma", "No Tumor", "Pituitary") classification, and CT scan segmentation.

---

## Project Structure

```
CURASCAN/
├── models/               # DenseNet classifier, UNet segmenter, Grad-CAM
├── data/                 # Training data (X-ray, MRI, CT)
├── train/                # Training scripts
├── app/                  # Streamlit apps 
├── utils/                # Datasets, losses, metrics, augmentations
├── scans/                # Uploaded patient scans (runtime)
├── checkpoints/          # Saved model weights (runtime)
├── curascan.db           # SQLite database (runtime)
├── requirements.txt
├──config.py
├──database.py
├──image_analysis.py
├──report_analysis.py
├── README.md
└── run.sh
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Train models

```bash
# Classification (chest X-ray)
./run.sh train cls --task xray --epochs 30

# Segmentation (CT)
./run.sh train seg --epochs 50

# K-Fold cross-validation
./run.sh train kfold --task xray --k 5
```
Repeat for MRI & CT 
Trained checkpoints are saved to `checkpoints/cls_best.pth` and `checkpoints/seg_best.pth`.

### 3. Run the apps

streamlit run app/app.py 

**Default admin credentials:** `admin` / `admin123`

---

## Data Layout

### X-ray (binary: normal / pneumonia)
```
data/xray/
  train/normal/   *.jpg
  train/pneumonia/
  val/normal/
  val/pneumonia/
  test/normal/
  test/pneumonia/
```

### MRI (4-class: glioma / meningioma / notumor / pituitary)
```
data/mri/
  train/{glioma,meningioma,notumor,pituitary}/
  test/{glioma,meningioma,notumor,pituitary}/
```

### CT Segmentation
```
data/ct/
 train/healthy/   *.jpg
  train/tumor/
  val/healthy/
  val/tumor/
  test/healthy/
  test/tumor/
```

---

## Models

| Model | Architecture | Task |
|-------|-------------|------|
| Classification | DenseNet121 + custom head | X-ray binary  / MRI 4-class |
| Segmentation | UNet (64→512) | CT binary mask |
| Explainability | Grad-CAM | Saliency heatmaps |

---


## Tech Stack

- **Frontend:** Streamlit ,  HTML, CSS
- **Backend:** python
- **ML:** PyTorch (DenseNet Classifier, U-Net Segmentation), scikit-learn
- **Vision:**  OpenCV, Pillow, NumPy, Matplotlib, Grad-CAM
- **Database:** SQLite
- **Base64** → embedding images in HTML
