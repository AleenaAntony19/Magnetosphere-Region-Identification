# 🛰️ Automatic Identification of Magnetospheric Regions Using Supervised Machine Learning

> **Deep Learning Project** | Bainty Kaur Chugh (253316) · Aleena Antony (253125)

---

## Overview

Identifying magnetospheric plasma regions is essential for understanding solar wind interactions, magnetic reconnection, and space weather evolution. Traditional manual identification from spacecraft data is slow and difficult to scale. This project implements and compares multiple machine learning approaches — unsupervised and supervised — to automatically classify magnetospheric regions using data from the **MMS** and **MESSENGER** missions.

The final hybrid model (CNN + Random Forest) achieves **~99% accuracy** with an **F1-score of 0.99**.

---

## Regions Classified

| Region | Description |
|---|---|
| **Solar Wind** | Undisturbed plasma flowing from the Sun |
| **Magnetosheath** | Turbulent region between bow shock and magnetopause |
| **Magnetosphere** | Region dominated by Earth's (or planet's) magnetic field |

---

## Approaches

### 1. Unsupervised — GMM Clustering (T24 Model)

A **Gaussian Mixture Model (GMM)** was used to cluster plasma regions without labelled training data. Key engineered features:

```
ratio_max_width = Width of dominant ion peak / Total energy bins
ratio_high_low  = High-energy flux / Low-energy flux
```

These features capture plasma behavior and generalise across different planetary magnetospheres. Additional inputs include normalised magnetic field strength and ion energy peak width.

---

### 2. Unsupervised — DBSCAN

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) was also explored. It performed poorly due to:
- Overlapping boundaries between plasma regions
- Difficulty separating transitional boundary layers (especially magnetosheath ↔ magnetosphere)
- Sensitivity to noise in spacecraft measurements

> **Conclusion:** DBSCAN is less effective than supervised methods for this task.

---

### 3. Supervised — CNN + Random Forest (Hybrid) ✅ Best

The core of this project is a **hybrid supervised framework** combining two complementary models.

#### Convolutional Neural Network (CNN)

- **Input:** Ion energy spectrograms shaped `(32, 40)` — 32 energy bins × 40 temporal steps (3-minute window)
- **Task:** Learn complex spectral patterns from ion energy distributions
- **Loss Function:**

$$L = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

where $y_i$ is the true class label and $\hat{y}_i$ is the predicted probability.

#### Random Forest (RF)

- **Input:** Scalar plasma parameters — magnetic field strength, ion temperature, spacecraft position
- **Task:** Capture threshold-based relationships among plasma parameters

#### Ensemble Prediction

```python
cnn_probs   = cnn_model.predict(spectrogram)
rf_probs    = rf_model.predict_proba(parameters)
final_probs = (cnn_probs + rf_probs) / 2
prediction  = np.argmax(final_probs)
```

Final predictions are made by **averaging the probability outputs** of both models.

---

## Results

| Method | Accuracy | F1-Score | Notes |
|---|---|---|---|
| GMM (T24) | Moderate | — | No labels needed; generalises well |
| DBSCAN | Low | — | Struggles with overlapping boundaries |
| **CNN + Random Forest** | **~99%** | **0.99** | Best overall; handles transitions well |

---

## Key Takeaways

- **Supervised learning significantly outperforms unsupervised clustering** for magnetospheric region classification, especially at plasma boundary transitions.
- The **hybrid CNN + RF architecture** leverages the strengths of both deep learning (spectral pattern recognition) and classical ML (scalar parameter thresholding).
- The framework is **lightweight and scalable**, making it suitable for large-scale analysis in upcoming missions.

---

## Missions & Applications

This framework is designed for — and validated on — data from:

- 🛸 **MMS** (Magnetospheric Multiscale Mission)
- 🪐 **MESSENGER** (Mercury Surface, Space Environment, Geochemistry, and Ranging)
- 🔭 **BepiColombo** (upcoming Mercury mission)

---

## Project Structure

```
├── data/                   # Spacecraft plasma & magnetic field data
│   ├── mms/
│   └── messenger/
├── models/
│   ├── gmm_t24.py          # Unsupervised GMM (T24 model)
│   ├── dbscan.py           # DBSCAN clustering
│   ├── cnn.py              # Convolutional Neural Network
│   └── random_forest.py    # Random Forest classifier
├── features/
│   └── feature_engineering.py   # ratio_max_width, ratio_high_low, etc.
├── ensemble/
│   └── hybrid_predict.py   # CNN + RF ensemble inference
├── evaluation/
│   └── metrics.py          # Accuracy, F1, confusion matrix
└── README.md
```

---

## Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow torch matplotlib
```

---

## Authors

| Name | ID |
|---|---|
| Bainty Kaur Chugh | 253316 |
| Aleena Antony | 253125 |

---

*Deep Learning Project Report — Automatic Identification of Magnetospheric Regions*
