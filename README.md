# 🍷 Hyperspectral Wine Classifier

A web application for non-destructive wine classification using hyperspectral imaging and machine learning. Built for the **Basler daA2500** camera with a custom 9×9 mosaic spectral filter (81 spectral bands), it allows researchers and producers to identify wine types from raw TIFF images — no lab analysis required.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iknalos-wine-classifier.streamlit.app)

---

## 📌 Overview

Traditional wine classification relies on chemical analysis — expensive, destructive, and time-consuming. This application leverages **hyperspectral imaging** to capture the unique spectral signature of each wine across 81 wavelength bands, then uses a trained machine learning model to classify unknown samples in seconds.

The pipeline is fully guided — from uploading training images to receiving a prediction — with no coding required.

---

## ✨ Features

- **4-step guided workflow** — upload, ROI selection, training, prediction
- **Dual ROI selector** — interactive sliders to select two analysis regions on the wine liquid
- **Patch-based feature extraction** — divides each ROI into sub-patches for richer training data
- **Ensemble classifier** — SVM + Random Forest + XGBoost soft-voting, proven in peer-reviewed hyperspectral wine research
- **Leave-One-Image-Out validation** — honest accuracy with no data leakage
- **Spectral visualisation** — signature plots, distance heatmap, PCA cluster view, confusion matrix
- **Batch prediction** — upload a ZIP of unknown images and classify all at once
- **Google Drive integration** — load training data or prediction ZIPs directly from Drive
- **Model export** — download the trained model as `.pkl` and reload it anytime
- **File management** — add or remove training files individually from the sidebar

---

## 🔬 How It Works

### Camera & Filter
The app is designed for the **Basler daA2500** camera fitted with a custom **9×9 Fabry-Pérot mosaic filter**. Each raw TIFF captured by this camera contains 81 spectral channels interleaved in a repeating tile pattern across the sensor. The app demosaics this raw image to extract each band separately.

### Pipeline

```
Raw TIFF  →  Demosaic (81 bands)  →  ROI extraction  →  Patch features  →  Classifier  →  Prediction
```

1. **Demosaicing** — separates the 9×9 tile pattern into 81 individual spectral band images
2. **ROI selection** — user defines two regions of interest on the wine liquid
3. **Patch extraction** — each ROI is divided into 30×30 px sub-patches; each patch produces a 162-dimensional feature vector (81 normalised band means + 81 band standard deviations)
4. **Training** — ensemble classifier trained with Leave-One-Image-Out cross-validation to prevent data leakage
5. **Prediction** — patches from new images are classified individually; final prediction is the average confidence across all patches

---

## 🧠 Classifier

The app uses a **soft-voting ensemble** of three models:

| Model | Role |
|-------|------|
| **SVM (RBF kernel)** | Best single-model performer for high-dimensional spectral data |
| **Random Forest** | Handles non-linear band interactions; resistant to overfitting |
| **XGBoost** | Gradient boosting; captures patterns the other two may miss |

Each model outputs a confidence probability per class. The final prediction is the weighted average of all three votes.

> **Why not deep learning?**
> CNN and transformer models achieve higher accuracy on large hyperspectral datasets but require thousands of labelled samples. For the small datasets typical in laboratory wine research (6–20 images), the ensemble approach consistently outperforms deep learning while training in seconds rather than hours.

**Research backing:**
- *ScienceDirect (2024)* — EBM-SVM showed exceptional stability in hyperspectral wine grape classification
- *PubMed* — SVM and MLP achieved F1 scores up to 0.99 in grapevine varietal classification using hyperspectral imaging
- *MDPI Remote Sensing (2024)* — SVM and Random Forest among top performers for grape quality grading with HSI

---

## 🚀 Getting Started

### Run locally

```bash
# Clone the repo
git clone https://github.com/iknalos/Wine-Classifier.git
cd Wine-Classifier

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

Opens at `http://localhost:8501`

### Deploy on Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → `main` → `app.py`
5. Click **Deploy**

---

## 📁 File Naming Convention

Training TIFF filenames must contain the wine label. Supported labels:

| Label | Example filename |
|-------|-----------------|
| `Dao` | `Dao_100K_1.tiff` |
| `LN` | `LN_100K_1.tiff` |
| `LO` | `LO_100K_1.tiff` |
| `ODC` | `ODC_100K_1.tiff` |
| `PN` | `PN_100K_1.tiff` |
| `PO` | `PO_100K_1.tiff` |

The label is detected automatically from the filename — no manual tagging needed.

---

## 📦 Project Structure

```
Wine-Classifier/
├── app.py               # Main Streamlit application
└── requirements.txt     # Python dependencies
```

---

## 🛠️ Requirements

```
streamlit
tifffile
numpy
scikit-learn
matplotlib
scipy
joblib
xgboost
```

Python 3.9+ recommended.

---

## 📷 Camera Setup

| Parameter | Value |
|-----------|-------|
| Camera | Basler daA2500 |
| Filter | Custom 9×9 mosaic (Fabry-Pérot) |
| Spectral bands | 81 |
| Image format | TIFF (16-bit, uint16) |
| Recommended exposure | Consistent across all captures |

> **Important:** All training and prediction images should be captured under the same exposure settings and lighting conditions. Exposure differences between training and unknown images will reduce classification accuracy.

---

## 📊 Validation

The app uses **Leave-One-Image-Out (LOIO) cross-validation** — the most honest validation strategy for small datasets. For each fold, all patches from one image are held out as the test set while the model trains on all remaining images. This prevents data leakage that would occur if patches from the same image appeared in both training and test sets.

---

## 🔮 Future Improvements

- [ ] Automatic tile size detection from raw TIFF
- [ ] Wavelength calibration — map band indices to actual nm values
- [ ] Support for other camera/filter configurations
- [ ] Export predictions as CSV report
- [ ] Multi-user session support with persistent storage

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 👤 Author

**iknalos**
Built with [Streamlit](https://streamlit.io) · Powered by scikit-learn, XGBoost & tifffile
