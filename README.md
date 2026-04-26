# 🧠 MavenNet — EEG Seizure Detection

> A deep learning pipeline for automatic epileptic seizure detection from EEG signals using Continuous Wavelet Transform (CWT) scalograms and a custom dual-attention CNN architecture.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Running the Project](#-running-the-project)
- [Experiments](#-experiments)
- [Results](#-results)
- [Visualisations](#-visualisations)
- [File Reference](#-file-reference)
- [Citation](#-citation)

---

## 🔬 Project Overview

MavenNet is an end-to-end EEG seizure detection system built on the **Bonn EEG dataset**. It converts raw 1-D EEG segments into 2-D CWT scalograms and classifies them using a lightweight CNN with a **Dual Attention module** (spatial + channel attention).

**Key features:**
- Continuous Wavelet Transform (Morlet wavelet) for time-frequency representation
- Custom MavenNet architecture with residual connections and dual attention
- 5-fold cross-validation with group-aware splitting (no data leakage)
- Weighted sampling for class imbalance
- Test-Time Augmentation (TTA) for robust inference
- 5-model ensemble with confidence-weighted voting
- Grad-CAM++ visualisation for interpretability
- t-SNE feature space visualisation

---

## 🏗 Architecture

```
Input EEG Segment (178 samples)
        │
        ▼
  CWT Scalogram (1 × T × 32)
        │
        ▼
┌─────────────────────────┐
│        Backbone         │
│  Conv1 → BN → ReLU      │
│  Conv2 → BN → ReLU      │
│  Conv3 + Skip → BN→ReLU │
│  Output: (128, H, W)    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     Dual Attention      │
│  ┌──────────────────┐   │
│  │ Spatial Attention│   │
│  │ Channel Attention│   │
│  │  → Fuse → Add   │   │
│  └──────────────────┘   │
└────────────┬────────────┘
             │
             ▼
   Global Average Pooling
             │
             ▼
       Dropout (0.4)
             │
             ▼
      FC → 2 classes
     (Normal / Seizure)
```

**Architecture is frozen — do not modify `maven_net.py`.**

---

## 📦 Dataset

This project uses the **Bonn EEG Dataset** from the University of Bonn, Germany.

### Download

1. Visit: [https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/)
2. Download all 5 sets: **Z, O, N, F, S**
3. Each set contains 100 single-channel EEG files in `.txt` format

### Dataset Description

| Set | Label | Class    | Description                          |
|-----|-------|----------|--------------------------------------|
| Z   | 0     | Normal   | Eyes open, healthy volunteers        |
| O   | 1     | Normal   | Eyes closed, healthy volunteers      |
| N   | 2     | Normal   | Seizure-free, hippocampal (opposite) |
| F   | 3     | Normal   | Seizure-free, epileptogenic zone     |
| S   | 4     | Seizure  | Ictal (during seizure activity)      |

- **Sampling rate:** 173.61 Hz
- **Segment length:** 178 samples (~1 second)
- **Total files:** 500 (100 per set)

### Place Files Here

```
data/
└── raw/
    ├── Z/    ← Z001.txt … Z100.txt
    ├── O/    ← O001.txt … O100.txt
    ├── N/    ← N001.txt … N100.txt
    ├── F/    ← F001.txt … F001.txt
    └── S/    ← S001.txt … S100.txt
```

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── raw/
│   │   ├── Z/               ← Normal EEG (eyes open)
│   │   ├── O/               ← Normal EEG (eyes closed)
│   │   ├── N/               ← Interictal (opposite hemisphere)
│   │   ├── F/               ← Interictal (epileptogenic zone)
│   │   └── S/               ← Seizure (ictal)
│   └── processed/
│       └── cwt_data.npy     ← Auto-generated CWT cache
│
├── src/
│   ├── models/
│   │   └── maven_net.py     ← MavenNet architecture (FROZEN)
│   ├── data/
│   │   ├── cwt.py           ← CWT scalogram generator
│   │   └── dataset.py       ← PyTorch Dataset + augmentation
│   ├── utils/
│   │   └── metrics.py       ← Evaluation metrics
│   └── interpretability/
│       └── gradcam.py       ← Grad-CAM++ implementation
│
├── checkpoints/             ← Saved model weights (auto-created)
│   ├── ABCD_vs_E_fold_1.pth
│   ├── ABCD_vs_E_fold_2.pth
│   └── ...
│
├── outputs/                 ← Generated plots (auto-created)
│   ├── roc_curve.png
│   ├── tsne.png
│   ├── gradcam.png
│   ├── confusion_matrix.png
│   └── results_summary.csv
│
├── preprocessing_pipeline.py  ← Raw data loader + chunking
├── train.py                   ← Training + cross-validation
├── pipeline.py                ← Visualisation pipeline
├── requirements.txt           ← Python dependencies
└── README.md                  ← This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Step 1 — Clone or Download

```bash
git clone https://github.com/your-username/mavennet-eeg.git
cd mavennet-eeg
```

### Step 2 — Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
pandas>=2.0.0
PyWavelets>=1.4.1
```

---

## 🚀 Quick Start

```bash
# 1. Place Bonn dataset in data/raw/
# 2. Train (generates checkpoints)
python train.py

# 3. Visualise (generates plots in outputs/)
python pipeline.py
```

---

## 🔧 Running the Project

### Step 1 — Verify Dataset

Make sure your folder looks like this before running anything:

```
data/raw/Z/Z001.txt  ← should exist
data/raw/S/S001.txt  ← should exist
```

### Step 2 — Train the Model

```bash
python train.py
```

What this does:
- Loads all EEG files from `data/raw/`
- Generates CWT scalograms and caches to `data/processed/cwt_data.npy`
- Runs 7 binary classification experiments
- For each experiment: 5-fold cross-validation with group-aware splitting
- Saves best checkpoint per fold to `checkpoints/`
- Saves final results to `outputs/results_summary.csv`

Training time estimate:
- GPU (CUDA): ~15–30 minutes for all experiments
- CPU only: ~2–4 hours

### Step 3 — Generate Visualisations

```bash
python pipeline.py
```

What this generates in `outputs/`:

| File | Description |
|------|-------------|
| `roc_curve.png` | ROC curve with AUC and operating point |
| `tsne.png` | t-SNE feature space (Normal vs Seizure) |
| `gradcam.png` | Grad-CAM++ heatmaps on seizure samples |
| `confusion_matrix.png` | Confusion matrix with per-cell stats |

### Configure Which Plots to Run

In `pipeline.py`, toggle these flags at the top:

```python
RUN_TSNE    = True    # t-SNE embedding plot
RUN_ROC     = True    # ROC curve
RUN_GRADCAM = True    # Grad-CAM++ heatmaps
RUN_CM      = True    # Confusion matrix
```

---

## 🧪 Experiments

The project runs 7 binary classification experiments, each comparing seizure (Set S) against different combinations of normal EEG:

| Experiment | Normal Classes | Description |
|------------|---------------|-------------|
| `ABCD_vs_E` | Z, O, N, F | All normal vs seizure (hardest) |
| `A_vs_E` | Z | Eyes-open healthy vs seizure |
| `B_vs_E` | O | Eyes-closed healthy vs seizure |
| `C_vs_E` | N | Interictal (opposite) vs seizure |
| `D_vs_E` | F | Interictal (epileptogenic) vs seizure |
| `AB_vs_E` | Z, O | Healthy vs seizure |
| `CD_vs_E` | N, F | Interictal vs seizure |

`ABCD_vs_E` is the most clinically relevant and the hardest.

---

## 📊 Results

Expected performance on the Bonn dataset:

### ABCD vs E (Hardest — Most Clinically Relevant)

| Metric | Expected Range |
|--------|---------------|
| AUC | 0.97 – 0.99 |
| Sensitivity | 0.95 – 0.98 |
| Specificity | 0.94 – 0.97 |
| F1 Score | 0.95 – 0.97 |
| Balanced Accuracy | 0.95 – 0.97 |

### Easier Splits (A/B vs E)

| Metric | Expected Range |
|--------|---------------|
| AUC | 0.99 – 1.00 |
| Sensitivity | 0.98 – 1.00 |
| Specificity | 0.97 – 1.00 |
| F1 Score | 0.98 – 1.00 |

Results are saved automatically to `outputs/results_summary.csv` after training.

---

## 🖼 Visualisations

### t-SNE Feature Space
Shows how well MavenNet separates seizure from normal EEG in the 128-dimensional embedding space. Clear cluster separation indicates the model has learned meaningful representations.

### ROC Curve
Plots True Positive Rate vs False Positive Rate across all thresholds. The red dot marks the optimal operating point selected via Youden's J statistic.

### Grad-CAM++ Heatmaps
Shows **which time-frequency regions** the model focuses on when predicting seizure. Three columns per sample:
- Left: raw CWT scalogram
- Middle: Grad-CAM++ activation map
- Right: overlay (what the model "sees")

### Confusion Matrix
Colour-coded confusion matrix showing:
- True Negatives (blue) — correctly identified normal
- True Positives (green) — correctly identified seizure
- False Positives (red) — normal misclassified as seizure
- False Negatives (red) — missed seizures ← most critical

---

## 📄 File Reference

| File | Purpose |
|------|---------|
| `preprocessing_pipeline.py` | Loads Bonn `.txt` files, segments with 50% overlap, normalises |
| `cwt.py` | Converts 1-D EEG to 2-D Morlet CWT scalogram (4–35 Hz) |
| `dataset.py` | PyTorch Dataset with SpecAugment-style augmentation |
| `maven_net.py` | MavenNet model: Backbone + DualAttention + FC (**frozen**) |
| `metrics.py` | Accuracy, sensitivity, specificity, AUC, F1, kappa, NPV |
| `gradcam.py` | Grad-CAM++ with zero-gradient fallback and hook cleanup |
| `train.py` | 5-fold GroupKFold training with TTA, ensemble, early stopping |
| `pipeline.py` | Runs all 4 visualisations and saves to `outputs/` |

---

## ⚠️ Important Notes

1. **Run `train.py` before `pipeline.py`** — pipeline loads saved checkpoints
2. **Do not modify `maven_net.py`** — architecture is frozen; weights depend on exact layer structure
3. **CWT cache** — first run generates `data/processed/cwt_data.npy` (~500MB). Subsequent runs load from cache automatically
4. **GPU recommended** — training on CPU is slow but works correctly
5. **Group-aware splits** — chunks from the same source file always stay in the same fold, preventing data leakage

---

## 📚 Citation

If you use this project in your research, please cite the Bonn EEG dataset:

```
Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001).
Indications of nonlinear deterministic and finite dimensional structures
in time series of brain electrical activity: Dependence on recording
region and brain state.
Physical Review E, 64, 061907.
```

---

## 📬 Contact

For questions or issues, open a GitHub issue or contact the project maintainer.

---

*MavenNet — EEG Seizure Detection Pipeline*
