# Air Pollution Forecasting Using Temporal Neural Networks and Machine Learning for PM2.5 Prediction

An end-to-end machine learning and deep learning framework for next-hour particulate matter ($PM_{2.5}$) forecasting using the Beijing Multi-Site Air Quality dataset. This repository implements a hybrid modeling approach that combines a 4-layer Deep Gated Recurrent Unit (GRU) with a high-dimensional LightGBM gradient-boosted regressor via 5-fold purged time-series cross-validation and out-of-fold optimal blending.

---

## Overview

This project models next-hour $PM_{2.5}$ concentrations by formulating complementary sequence-based and tabular feature-driven pipelines:
* **Sequential Representation**: A Deep GRU models 24-hour temporal dynamics and atmospheric transitions.
* **Tabular Feature Learning**: LightGBM captures non-linear tabular interactions across rolling temporal windows, differenced momentum, and cross-station aggregates.
* **Ensemble Blending**: A weighted linear blend optimizes Out-of-Fold (OOF) Root Mean Squared Error (RMSE) to achieve superior test-set generalization.

---

## Architecture & Workflow

```
                     ┌──────────────────────────────────────────┐
                     │   Beijing Multi-Site Air Quality Data    │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │            Data Preprocessing            │
                     │  - Missing Imputation (ffill/bfill)      │
                     │  - Missingness Indicator Flags           │
                     │  - Relative Humidity & Wind Decomposition│
                     │  - Regional Spatial Aggregations         │
                     └─────────────┬──────────────┬─────────────┘
                                   │              │
                   24-Hour Windows │              │ 468 Tabular Features
                                   ▼              ▼
         ┌───────────────────────────┐          ┌───────────────────────────┐
         │     Deep 4-Layer GRU      │          │     LightGBM Regressor    │
         │ - LayerNorm + GELU Head   │          │ - Lag, EWM, Rolling Stats │
         │ - Cosine Annealing AdamW  │          │ - Station-Wise Grouping   │
         └─────────────┬─────────────┘          └─────────────┬─────────────┘
                       │                                      │
                       └──────────────────┬───────────────────┘
                                          ▼
                               ┌──────────────────────┐
                               │  Purged 5-Fold CV    │
                               │  & Optimal Blending  │
                               │ (0.49 LGB + 0.51 GRU)│
                               └──────────┬───────────┘
                                          ▼
                               ┌──────────────────────┐
                               │ Next-Hour Prediction │
                               │ Test RMSE: 16.6269   │
                               │ Test MAE:   8.7970   │
                               └──────────────────────┘
```

---

## Dataset

* **Source**: Beijing Multi-Site Air Quality Dataset (2013–2017).
* **Spatial Resolution**: 12 distinct air-quality monitoring stations across Beijing.
* **Temporal Resolution**: Hourly pollutant readings ($PM_{2.5}$, $PM_{10}$, $SO_2$, $NO_2$, $CO$, $O_3$) and meteorological variables (Temperature, Pressure, Dew Point, Rainfall, Wind Direction, Wind Speed).
* **Task Objective**: Given historical sensory data for the past 24 hours ($t-23$ to $t$), forecast the continuous concentration of $PM_{2.5}$ at $t+1$.

---

## Feature Engineering

### 1. Meteorological & Physical Transformations
* **Wind Vector Decomposition**: Circular 16-point wind compass directions ($wd$) are mapped to angles in radians and decomposed into orthogonal vectors.
* **Relative Humidity Calculation**: Derived via the August-Roche-Magnus approximation using clipped temperature and dew-point measurements to reflect saturation vapor pressure dynamics.
* **Atmospheric Dispersion Index**: Defined as `dispersion_idx = WSPM * PM_2.5` to model wind-driven surface pollutant dispersal.

### 2. Temporal & Regional Spatial Metrics
* **Cyclical Time Encoding**: Sine and cosine components computed for hour and month.
* **Regional Spatial Statistics**: Instantaneous regional mean ($PM_{2.5}$) across all 12 stations at each timestamp, alongside station-level deviation ($\Delta = PM_{2.5}^{\text{station}} - PM_{2.5}^{\text{regional}}$) to capture localized anomaly patterns.
* **Missing Value Indicators**: Explicit binary flags for all continuous pollutants and meteorological sensors to preserve missingness signals before imputation.

### 3. LightGBM Tabular Feature Space (468 Total Features)
* **Lag Sequences**: 24-step hourly historical lags across all 14 core features.
* **Rolling Statistics**: Moving mean, standard deviation, minimum, maximum, and Exponential Weighted Moving Averages (EWMA) computed over sliding windows of 3, 6, 12, and 24 hours per station.
* **Momentum Differences**: First-order and second-order temporal differences for $PM_{2.5}$, $PM_{10}$, and temperature-dew point spread.

---

## Model Architectures

### Deep Residual GRU
* **Input**: 24-step sequence tensor containing continuous engineered predictors and missingness indicators.
* **Recurrent Backbone**: 4-layer Gated Recurrent Unit (hidden size = 256, dropout = 0.2, batch-first).
* **Normalization & Head**: Final hidden state passed through `nn.LayerNorm(256)`, followed by a multi-layer projection head:
  $$\text{Linear}(256 \to 256) \to \text{GELU} \to \text{Dropout}(0.2) \to \text{Linear}(256 \to 128) \to \text{GELU} \to \text{Linear}(128 \to 1)$$
* **Optimization**: Smooth L1 loss ($\beta=1.0$), AdamW optimizer (learning rate = $1 \times 10^{-3}$, weight decay = $1 \times 10^{-4}$), Cosine Annealing learning rate schedule, Mixed Precision training (`torch.cuda.amp`), and early stopping (patience = 6).

### LightGBM Regressor
* **Objective**: L2 regression (`rmse`), gradient boosted decision tree (GBDT).
* **Hyperparameters**: `learning_rate: 0.02`, `num_leaves: 127`, `max_depth: 8`, `feature_fraction: 0.7`, `bagging_fraction: 0.8`, `bagging_freq: 1`, `lambda_l1: 1.5`, `lambda_l2: 3.0`.
* **Categorical Handling**: Native station categorical feature encoding.

---

## Cross-Validation & Blending

A leak-free validation setup ensures unbiased performance estimation on time-dependent sequences:
* **5-Fold Time-Series Split**: Time-ordered contiguous chronological blocks.
* **Purge Gap**: A 24-hour purge interval removed before and after validation blocks to eliminate autocorrelation contamination between folds.
* **Feature Standardization**: Scaler fit exclusively on training folds to prevent lookahead bias.
* **Blending Optimization**: Grid-search optimization over out-of-fold predictions identified optimal weights:
  ensemble = 0.49 * LightGBM + 0.51 * Deep GRU

---

## Performance & Benchmarks

### 5-Fold Cross-Validation Metrics

| Fold | Deep GRU Val RMSE | LightGBM Val RMSE |
| :--- | :---: | :---: |
| **Fold 1** | 18.6678 | 18.5054 |
| **Fold 2** | 19.8603 | 19.8587 |
| **Fold 3** | 13.9149 | 13.8093 |
| **Fold 4** | 17.3112 | 16.8999 |
| **Fold 5** | 19.2422 | 20.0272 |

### Test Set Generalization

| Model Architecture | OOF RMSE | Test RMSE | Test MAE |
| :--- | :---: | :---: | :---: |
| **Standalone LightGBM** | 17.9678 | 17.1098 | — |
| **Standalone Deep GRU** | 17.9248 | 16.7802 | — |
| **Weighted Ensemble ($0.49 \text{ LGB} + 0.51 \text{ GRU}$)** | **17.5075** | **16.6269** | **8.7970** |

---

## Repository Structure

```text
├── data/
│   ├── train_raw.csv          # Multi-station historical training data
│   ├── test.csv               # Testing data given for Kaggle competition
│   └── test_raw.csv           # Multi-station historical testing data
└── Final Notebook.ipynb       # Notebook
```

## Acknowledgments

This project was developed for **CO5420: Artificial Neural Networks and Deep Learning**, Department of Computer Engineering, Faculty of Engineering, **University of Peradeniya**.

### Project Team
* **E/22/001** — H. M. H. N. Aberathna
* **E/22/008** — T. H. Abeywickrama
* **E/22/027** — M. A. N. P. Anawarathne
* **E/22/130** — S. H. S. Hansara
* **E/22/362** — W. A. H. Sathsarani
