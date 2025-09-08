# Random Forest Hyperparameter Tuning Guide

This document explains how to run hyperparameter tuning for the RandomForest model
using the `RFHyperparameterTuner` class.

---

## 📦 Prerequisites

- Python 3.9+
- Dependencies installed (`scikit-learn`, `pandas`, `numpy`)
- Your project structure set up with:
  - `train_model_new.py`
  - `hyperparameter_tuning.py`
  - `run_model_training.py`

---

## 🚀 Running Tuning via CLI

You can launch tuning from the command line using `run_model_training.py`.

```bash
python run_model_training.py --mode tune --tune-strategy <strategy> --tune-validation <method> --tune-metric <metric>
```

### Arguments

- `--mode`  
  Must be set to `tune` to enable hyperparameter tuning.

- `--tune-strategy`  
  Which parameter grid to use:  
  - `coarse` → quick probe (~few minutes)  
  - `fine` → more detailed (~30–60 min)  
  - `random` → random sampling from a wide grid  
  - `progressive` → runs `coarse` first, then `fine`

- `--tune-validation`  
  Validation method:  
  - `static` → single train/validation split (faster)  
  - `walk_forward` → rolling time-window validation (slower, but better for time series)

- `--tune-metric`  
  Primary evaluation metric (used to pick best params):  
  - `roc_auc` (default)  
  - `f1`  
  - `precision`  
  - `recall`  
  - `accuracy`

---

## 🔧 Examples

Run a **coarse grid search** with ROC-AUC:
```bash
python run_model_training.py --mode tune --tune-strategy coarse --tune-validation static --tune-metric roc_auc
```

Run **progressive tuning** (coarse → fine) with F1 score:
```bash
python run_model_training.py --mode tune --tune-strategy progressive --tune-validation static --tune-metric f1
```

Run **walk-forward validation** (time-series):
```bash
python run_model_training.py --mode tune --tune-strategy coarse --tune-validation walk_forward --tune-metric roc_auc
```

Run a **random grid search** (with sampling):
```bash
python run_model_training.py --mode tune --tune-strategy random --tune-validation static --tune-metric f1
```

---

## 📂 Outputs

- Results are saved under:
  ```
  tuning_results/tuning_results_<timestamp>.csv
  ```
- The **best model** can be saved with:
  ```python
  tuner.save_best_model("artifacts")
  ```

---

## ⚡ Tips

- On laptops (limited RAM/CPU), use:
  - `--tune-strategy coarse`
  - `--tune-validation static`
  - Limit `n_jobs=2–4` inside `RandomForestClassifier`
- For larger servers, try:
  - `--tune-strategy progressive`
  - `--tune-validation walk_forward`
  - Higher `n_jobs` (parallel training)

---
