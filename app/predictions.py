# predictions.py — predict-only pipeline (no training)

import os
import sys
import glob
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix  # <-- add



# # Ensure local imports resolve
# HERE = os.path.dirname(os.path.abspath(__file__))
# if HERE not in sys.path:
#     sys.path.append(HERE)

# # File-relative directories
# #DATA_DIR = os.path.join(HERE, "data")
# DATA_DIR = os.path.join(HERE, "..", "data")  # Go up one level

# ARTIFACTS_DIR = os.path.join(HERE, "artifacts")
# RESULTS_DIR = os.path.join(HERE, "results")


# File-relative directories (resolve to parent/<dir>)
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
ARTIFACTS_DIR = (HERE / ".." / "artifacts").resolve()
RESULTS_DIR = (HERE / ".." / "results").resolve()


from train_model_new import TrainModel  # only used to prepare dataframe/splits


# --- Tiny adapter so TrainModel can init from a df we already have ---
class _TransformAdapter:
    def __init__(self, df: pd.DataFrame):
        self.transformed_df = df


# =========================
# Loaders (data + model)
# =========================
# def load_latest_data() -> pd.DataFrame:
#     """
#     Finds the newest combined/complete/batch parquet produced by your pipeline.
#     Returns a DataFrame.
#     """
#     patterns = [
#         "stock_data_combined_*.parquet",  # highest priority
#         "stock_data_complete_*.parquet",
#         "stock_data_batch_*.parquet",
#     ]

#     newest = None
#     newest_mtime = -1

#     for pat in patterns:
#         for path in glob.glob(os.path.join(DATA_DIR, pat)):
#             mtime = os.path.getmtime(path)
#             if mtime > newest_mtime:
#                 newest = path
#                 newest_mtime = mtime

#     if newest is None:
#         raise FileNotFoundError(
#             f"No parquet files found in {DATA_DIR} matching {patterns}."
#         )

#     print(f"[load_latest_data] Using: {os.path.basename(newest)}")
#     df = pd.read_parquet(newest)
#     return df


def load_latest_data() -> pd.DataFrame:
    """
    Finds the newest parquet produced by your pipeline.
    Priorities: combined > complete > batch > small; fallback to any *.parquet.
    """
    patterns = [
        "stock_data_combined_*.parquet",
        "stock_data_complete_*.parquet",
        "stock_data_batch_*.parquet",
        "stock_data_small_*.parquet",   # <-- added
    ]

    # collect candidates in priority order
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(sorted(DATA_DIR.glob(pat), key=lambda p: p.stat().st_mtime))

    # fallback to any parquet in data/
    if not candidates:
        candidates = sorted(DATA_DIR.glob("*.parquet"), key=lambda p: p.stat().st_mtime)

    if not candidates:
        raise FileNotFoundError(
            f"No parquet files found in {DATA_DIR} matching {patterns + ['*.parquet']}."
        )

    latest = candidates[-1]
    print(f"[load_latest_data] Using: {latest.name}  (from {DATA_DIR})")
    return pd.read_parquet(latest)

# Add this to run_predictions.py after imports:
def load_latest_data_fixed():
    """Fixed version that uses correct paths"""
    patterns = [
        "stock_data_combined_*.parquet",
        "stock_data_complete_*.parquet",
        "stock_data_small_*.parquet",  # Add this pattern too
        "stock_data_batch_*.parquet",
    ]
    
    latest_file = None
    latest_time = 0
    
    for pattern in patterns:
        files = list(DATA_DIR.glob(pattern))  # Use your corrected DATA_DIR
        for file in files:
            if file.stat().st_mtime > latest_time:
                latest_time = file.stat().st_mtime
                latest_file = file
    
    if latest_file is None:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")
    
    print(f"[load_latest_data] Using: {latest_file.name}")
    return pd.read_parquet(latest_file)

# Then in your PredictionRunner.load_data_and_model() method, replace:
# self.data = load_latest_data()
# with:
# self.data = load_latest_data_fixed()
def _try_read_feature_sidecar(artifacts_dir: str) -> Optional[List[str]]:
    """
    Attempt to read feature order from a sidecar JSON placed in artifacts_dir.
    Accepts either a dict with a key for features, or a plain list.
    """
    candidates = ("model_meta.json", "features.json", "rf_features.json")
    for name in candidates:
        p = os.path.join(artifacts_dir, name)
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    for key in ("inference_feature_columns", "feature_cols", "features"):
                        if key in meta and isinstance(meta[key], list):
                            return list(meta[key])
                elif isinstance(meta, list):
                    return list(meta)
            except Exception as e:
                print(f"[feature sidecar] Failed reading {name}: {e}")
    return None


def load_model_and_features(artifacts_dir: str = ARTIFACTS_DIR) -> Tuple[object, List[str], Optional[str]]:
    """
    Loads the newest persisted model from artifacts_dir.
    Supports:
      - wrapped TrainModel (has .model and ._inference_feature_columns)
      - bare sklearn estimator (RandomForestClassifier, etc.)
    Resolves feature order via:
      1) model.feature_names_in_ (if fit on DataFrame)
      2) sidecar JSON in artifacts_dir
      3) (fallback) raise with guidance
    Returns: (sk_model, feature_cols, target_col_if_available_or_None)
    """
    import joblib

    paths = glob.glob(os.path.join(artifacts_dir, "*.joblib")) + \
            glob.glob(os.path.join(artifacts_dir, "*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No model artifacts found in {artifacts_dir}")

    model_path = max(paths, key=os.path.getmtime)
    model_obj = joblib.load(model_path)
    print(f"[load_model_and_features] Using model file: {os.path.basename(model_path)}")

    # Case A: wrapper object with metadata
    if hasattr(model_obj, "model"):
        sk_model = model_obj.model
        feat = getattr(model_obj, "_inference_feature_columns", None)
        tgt = getattr(model_obj, "target_col", None)
        if not feat:
            if hasattr(sk_model, "feature_names_in_"):
                feat = list(sk_model.feature_names_in_)
            else:
                feat = _try_read_feature_sidecar(artifacts_dir)
        if not feat:
            raise RuntimeError(
                "Wrapped model found but feature list is missing. "
                "Add a sidecar JSON with the exact training feature order."
            )
        print(f"[load_model_and_features] Features resolved: {len(feat)}")
        return sk_model, feat, tgt

    # Case B: bare sklearn estimator
    sk_model = model_obj
    feat = None
    tgt = None

    if hasattr(sk_model, "feature_names_in_"):
        feat = list(sk_model.feature_names_in_)
        print(f"[load_model_and_features] feature_names_in_: {len(feat)} features")
    else:
        feat = _try_read_feature_sidecar(artifacts_dir)
        if feat:
            print(f"[load_model_and_features] Sidecar features: {len(feat)}")
        else:
            raise RuntimeError(
                "Could not resolve feature list for inference.\n"
                "- If the model was fit on a NumPy array, sklearn won't store names.\n"
                "- Please provide a sidecar JSON in artifacts with the exact training feature order.\n"
                "  (e.g., model_meta.json with key 'inference_feature_columns' or 'feature_cols')"
            )

    return sk_model, feat, tgt


# =========================
# Prediction comparator
# =========================
class PredictionComparator:
    """Compare different prediction strategies against each other"""
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.prediction_cols: List[str] = []

    # Update your PredictionComparator.add_manual_predictions() method

    def add_manual_predictions(self) -> List[str]:
        """Add rule-based manual predictions with CORRECTED macro column names."""
        print("Creating manual rule-based predictions...")

        def _alias_series(df: pd.DataFrame, aliases: List[str]) -> Optional[pd.Series]:
            for name in aliases:
                if name in df.columns:
                    return pd.to_numeric(df[name], errors="coerce")
            print(f"Warning: none of columns {aliases} found; treating rule as unavailable.")
            return None

        def _safe_num(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
            print(f"Warning: Column '{col}' not found, using default value {default}")
            return pd.Series(default, index=df.index, dtype="float64")

        # pred0: CCI > 200 (momentum breakout)
        cci = _safe_num(self.df, "cci", 0.0)
        self.df["pred0_manual_cci"] = (cci > 200).astype(np.int8)

        # pred1: Previous 30d growth > 1 (momentum)
        growth_30d = _safe_num(self.df, "growth_30d", 0.0)
        self.df["pred1_manual_prev_g1"] = (growth_30d > 1).astype(np.int8)

        # pred2: Previous 30d growth > 1 AND S&P500 30d growth > 1
        growth_snp500_30d = _safe_num(self.df, "growth_snp500_30d", 0.0)
        self.df["pred2_manual_prev_g1_and_snp"] = (
            (growth_30d > 1) & (growth_snp500_30d > 1)
        ).astype(np.int8)

        # CORRECTED: Use actual column names from your data
        # 10-Year Treasury YoY growth rate
        dgs10_yoy = _alias_series(self.df, ["dgs10_yoy"])
        # 5-Year Treasury YoY growth rate  
        dgs5_yoy = _alias_series(self.df, ["dgs5_yoy"])
        # Federal Funds Rate YoY growth rate
        fedfunds_yoy = _alias_series(self.df, ["fedfunds_yoy"])

        # pred3: Low rate environment (declining rates YoY)
        if dgs10_yoy is not None and dgs5_yoy is not None:
            # Both 10Y and 5Y rates declining year-over-year
            self.df["pred3_manual_declining_rates"] = (
                (dgs10_yoy < 0) & (dgs5_yoy < 0)
            ).astype(np.int8)
        else:
            self.df["pred3_manual_declining_rates"] = 0

        # pred4: Fed easing cycle (Fed Funds declining YoY) 
        if fedfunds_yoy is not None:
            # Federal Funds Rate declining year-over-year
            self.df["pred4_manual_fed_easing"] = (fedfunds_yoy < -0.1).astype(np.int8)  # >10% decline
        else:
            self.df["pred4_manual_fed_easing"] = 0

        # pred5: NEW - VIX fear contrarian signal
        growth_vix_30d = _safe_num(self.df, "growth_vix_30d", 0.0)
        # Buy when VIX spiked >20% in past 30 days (contrarian)
        self.df["pred5_manual_vix_contrarian"] = (growth_vix_30d > 0.2).astype(np.int8)

        # pred6: NEW - Bitcoin momentum divergence
        growth_btc_30d = _safe_num(self.df, "growth_btc_30d", 0.0)
        # Stock momentum + Bitcoin momentum
        self.df["pred6_manual_stock_btc_momentum"] = (
            (growth_30d > 1.0) & (growth_btc_30d > 1.0)
        ).astype(np.int8)

        manual_preds = [
            "pred0_manual_cci",
            "pred1_manual_prev_g1", 
            "pred2_manual_prev_g1_and_snp",
            "pred3_manual_declining_rates",      # UPDATED
            "pred4_manual_fed_easing",           # UPDATED  
            "pred5_manual_vix_contrarian",       # NEW
            "pred6_manual_stock_btc_momentum",   # NEW
        ]
        self.prediction_cols.extend(manual_preds)

        print("Manual prediction summary:")
        for pred in manual_preds:
            positive_rate = float(self.df[pred].mean())
            print(f"  {pred}: {positive_rate:.1%} positive predictions")

        return manual_preds

    def add_ml_predictions(self, model, feature_cols: List[str],
                           thresholds=(0.21, 0.50, 0.65, 0.80, 0.90)) -> List[str]:
        """
        Build a clean feature matrix and add ML predictions:
          - preserves training feature order
          - coerces to numeric
          - replaces ±inf with NaN, then fills with column medians (fallback 0)
          - mild winsorization
        Returns: list of created binary prediction column names.
        """
        # 1) Build X in one shot (prevents fragmentation)
        X = pd.DataFrame(
            {c: (pd.to_numeric(self.df[c], errors="coerce") if c in self.df.columns else 0.0)
             for c in feature_cols},
            index=self.df.index,
        )

        # 2) Sanitize non-finites
        arr = X.to_numpy()
        if not np.isfinite(arr).all():
            bad_cols = X.columns[~np.isfinite(arr).all(axis=0)].tolist()
            print(f"[add_ml_predictions] Non-finite detected in: {bad_cols[:12]}{'...' if len(bad_cols) > 12 else ''}")
            X = X.replace([np.inf, -np.inf], np.nan)
            med = X.median(numeric_only=True, skipna=True)
            X = X.fillna(med).fillna(0.0)

        # 3) Mild winsorization
        try:
            lo = X.quantile(0.001, numeric_only=True)
            hi = X.quantile(0.999, numeric_only=True)
            X = X.clip(lower=lo, upper=hi, axis=1)
        except Exception:
            pass

        # 4) Final guard + dtype
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X = X.astype(np.float32, copy=False)

        # 5) Predict probabilities
        proba = model.predict_proba(X)[:, 1]
        self.df["rf_prob_30d"] = proba  # legacy-friendly name

        # 6) Thresholds -> columns expected downstream (pred10_, pred11_, ...)
        ml_preds = []
        for i, thr in enumerate(thresholds):
            pct = int(float(thr) * 100)
            col = f"pred{10 + i}_rf_thresh_{pct}"
            self.df[col] = (proba >= float(thr)).astype(np.int8)
            ml_preds.append(col)

        self.prediction_cols.extend(ml_preds)

        print("ML prediction summary:")
        for col in ml_preds:
            positive_rate = self.df[col].mean()
            print(f"  {col}: {positive_rate:.1%} positive predictions")

        return ml_preds

    def add_ml_thresholds_from_validation(self, proba_col: str = "rf_prob_30d",
                                          target_rates=(0.01, 0.03, 0.05)) -> Dict[str, float]:
        """
        Choose thresholds on the VALIDATION split to hit given selection rates,
        then apply to all splits. No model fitting; just quantile lookups.
        Returns a dict {column_name: threshold}.
        """
        if proba_col not in self.df.columns:
            print(f"[add_ml_thresholds_from_validation] Missing proba col: {proba_col}")
            return {}

        val = self.df[self.df["split"] == "validation"]
        if val.empty:
            print("[add_ml_thresholds_from_validation] No validation data; skipping.")
            return {}

        thresholds: Dict[str, float] = {}
        for r in target_rates:
            thr = float(val[proba_col].quantile(1.0 - r))
            col = f"pred15_rf_auto_rate_{int(r*100)}p"
            self.df[col] = (self.df[proba_col] >= thr).astype(np.int8)
            if col not in self.prediction_cols:
                self.prediction_cols.append(col)
            thresholds[col] = thr
            val_rate = float((val[proba_col] >= thr).mean())
            test = self.df[self.df["split"] == "test"]
            test_rate = float((test[proba_col] >= thr).mean()) if not test.empty else float("nan")
            print(f"[auto-threshold] {col}: thr={thr:.3f} | val rate={val_rate:.2%} | test rate={test_rate:.2%}")

        return thresholds

    def add_daily_topn(self, proba_col: str = "rf_prob_30d", n: int = 3,
                       date_col: str = "Date") -> str:
        """
        Mark Top-N probabilities per day across all tickers.
        Evaluated on the 'test' split later, so no leakage in metrics.
        """
        if proba_col not in self.df.columns or date_col not in self.df.columns:
            print(f"[add_daily_topn] Missing {proba_col} or {date_col}; skipping Top-{n}.")
            return ""

        col = f"pred30_top{n}_daily"
        self.df[col] = (
            self.df.groupby(self.df[date_col])[proba_col]
                   .rank(method="first", ascending=False)
                   .le(n)
                   .astype(np.int8)
        )
        if col not in self.prediction_cols:
            self.prediction_cols.append(col)
        rate_test = float(self.df.loc[self.df["split"] == "test", col].mean())
        print(f"[Top-{n} daily] {col}: test prediction rate ~{rate_test:.2%}")
        return col

    def add_ensemble_predictions(self) -> List[str]:
        """Add ensemble predictions combining manual and ML with sensible gates."""
        print("Creating ensemble predictions...")

        manual_preds = [c for c in self.prediction_cols if "manual" in c]
        ml_mid = [c for c in self.prediction_cols if c.startswith("pred11_rf_thresh_50")]
        ml_auto = [c for c in self.prediction_cols if c.startswith("pred15_rf_auto_rate_")]
        topn = [c for c in self.prediction_cols if c.startswith("pred30_top")]

        ensemble_preds: List[str] = []

        # Ensemble A: medium ML confidence AND momentum confirmation
        if ml_mid and "pred1_manual_prev_g1" in manual_preds:
            col = "pred20_ens_ml50_and_momentum"
            self.df[col] = (
                (self.df[ml_mid[0]] == 1) & (self.df["pred1_manual_prev_g1"] == 1)
            ).astype(np.int8)
            ensemble_preds.append(col)

        # Ensemble B: auto 1% rate OR daily Top-3 (captures very selective ideas)
        auto_1p = [c for c in ml_auto if c.endswith("_1p")]
        if auto_1p:
            base = auto_1p[0]
            if topn:
                col = "pred21_ens_auto1p_or_top3"
                self.df[col] = ((self.df[base] == 1) | (self.df[topn[0]] == 1)).astype(np.int8)
                ensemble_preds.append(col)
            else:
                # fallback: just use auto 1%
                if base not in self.prediction_cols:
                    self.prediction_cols.append(base)

        # Ensemble C: majority of manual >=2 plus auto 3% (requires at least 3 manual rules present)
        if len(manual_preds) >= 3:
            man_sum = self.df[manual_preds].sum(axis=1)
            auto_3p = [c for c in ml_auto if c.endswith("_3p")]
            if auto_3p:
                col = "pred22_ens_manual2plus_and_auto3p"
                self.df[col] = ((man_sum >= 2) & (self.df[auto_3p[0]] == 1)).astype(np.int8)
                ensemble_preds.append(col)

        # Register
        for c in ensemble_preds:
            if c not in self.prediction_cols:
                self.prediction_cols.append(c)

        for pred in ensemble_preds:
            positive_rate = float(self.df[pred].mean())
            print(f"  {pred}: {positive_rate:.1%} positive predictions")

        return ensemble_preds

    def _calculate_prediction_metrics(self, y_true, y_pred, eval_df, pred_col) -> dict:
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        from sklearn.metrics import confusion_matrix  # <-- add
        





        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Debug: print confusion matrix (remove after validation)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        print(f"  {pred_col}: tp={tp} fp={fp} tn={tn} fn={fn}")

        n_predictions = int(y_pred.sum())
        prediction_rate = float(y_pred.mean())

        # Daily metrics
        baseline_hit_rate = float(eval_df[self.target_col].mean())
        positive_preds = eval_df[eval_df[pred_col] == 1]

        if len(positive_preds) > 0:
            strategy_hit_rate = float(positive_preds[self.target_col].mean())
            improvement = strategy_hit_rate - baseline_hit_rate

            # Growth lift
            growth_lift = np.nan
            growth_col = "growth_future_30d"
            if growth_col in eval_df.columns:
                baseline_growth = float(eval_df[growth_col].mean())
                strategy_growth = float(positive_preds[growth_col].mean())
                growth_lift = strategy_growth - baseline_growth
        else:
            strategy_hit_rate = 0.0
            improvement = -baseline_hit_rate
            growth_lift = np.nan

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "n_predictions": n_predictions,
            "prediction_rate": prediction_rate,
            "daily_hit_rate": strategy_hit_rate,
            "daily_baseline": baseline_hit_rate,
            "daily_improvement": improvement,
            "avg_growth_lift": growth_lift,
        }

    # def evaluate_all_predictions(self, split: str = "test") -> pd.DataFrame:
    #     """Evaluate all prediction strategies - RANKED BY PRECISION"""
    #     print(f"\nEvaluating all {len(self.prediction_cols)} prediction strategies on {split} set...")
    #     print("PRIMARY METRIC: PRECISION (minimizing false positives)")

    #     eval_df = self.df[self.df["split"] == split].copy()
    #     if len(eval_df) == 0:
    #         print(f"No data found for split '{split}'")
    #         return pd.DataFrame()

    #     results = []
    #     for pred_col in self.prediction_cols:
    #         if pred_col not in eval_df.columns:
    #             continue
    #         y_pred = eval_df[pred_col]
    #         y_true = eval_df[self.target_col]

    #         # Skip strategies that make no predictions
    #         if y_pred.sum() == 0:
    #             print(f"  Skipping {pred_col}: No positive predictions")
    #             continue

    #         metrics = self._calculate_prediction_metrics(y_true, y_pred, eval_df, pred_col)
    #         metrics["strategy"] = pred_col
    #         results.append(metrics)

    #     results_df = pd.DataFrame(results)

    #     # RANK BY PRECISION (primary), then F1 (secondary), then daily_improvement (tertiary)
    #     if not results_df.empty:
    #         results_df = results_df.sort_values(
    #             ["precision", "f1_score", "daily_improvement"],
    #             ascending=[False, False, False],
    #         )

    #     return results_df
    
    def evaluate_all_predictions(self, split: str = "test") -> pd.DataFrame:
        """Evaluate all prediction strategies on a given split — RANKED BY PRECISION.

        - Strictly aligns/cleans labels and predictions (numeric coercion, NaN drop)
        - Skips strategies that end up with no positive predictions after cleaning
        - Sorts by precision (desc), then F1 (desc), then daily_improvement (desc)
        """
        print(f"\nEvaluating all {len(self.prediction_cols)} prediction strategies on {split} set...")
        print("PRIMARY METRIC: PRECISION (minimizing false positives)")

        eval_df = self.df[self.df["split"] == split].copy()
        if eval_df.empty:
            print(f"No data found for split '{split}'")
            return pd.DataFrame()

        results = []
        for pred_col in self.prediction_cols:
            if pred_col not in eval_df.columns:
                continue

            # --- strict alignment + cleaning ---
            pair = eval_df[[self.target_col, pred_col]].copy()
            pair[self.target_col] = pd.to_numeric(pair[self.target_col], errors="coerce")
            pair[pred_col] = pd.to_numeric(pair[pred_col], errors="coerce")
            pair = pair.dropna(subset=[self.target_col, pred_col])

            if pair.empty:
                print(f"  Skipping {pred_col}: no valid rows after cleaning")
                continue

            y_true = pair[self.target_col].astype(int).to_numpy()
            y_pred = pair[pred_col].astype(int).to_numpy()

            # Skip strategies that make no predictions (after cleaning)
            if y_pred.sum() == 0:
                print(f"  Skipping {pred_col}: No positive predictions (after cleaning)")
                continue

            metrics = self._calculate_prediction_metrics(y_true, y_pred, pair, pred_col)
            metrics["strategy"] = pred_col
            results.append(metrics)

        results_df = pd.DataFrame(results)

        # RANK BY PRECISION (primary), then F1 (secondary), then daily_improvement (tertiary)
        if not results_df.empty:
            results_df = results_df.sort_values(
                ["precision", "f1_score", "daily_improvement"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

        return results_df


    def print_comparison_report(self, results_df: pd.DataFrame):
        """Print comprehensive comparison report - PRECISION FOCUSED"""
        print("\n" + "=" * 100)
        print("STRATEGY COMPARISON REPORT - RANKED BY PRECISION")
        print("=" * 100)

        if len(results_df) == 0:
            print("No results to display")
            return

        # Top strategies summary - precision focused
        print("TOP 10 STRATEGIES (ranked by PRECISION, then F1, then daily improvement):")
        print("-" * 100)

        summary_cols = [
            "strategy",
            "precision",
            "f1_score",
            "recall",
            "daily_improvement",
            "avg_growth_lift",
            "prediction_rate",
            "n_predictions",
        ]

        display_df = results_df[summary_cols].head(10).copy()

        # Format for better readability
        display_df["precision"] = (display_df["precision"] * 100).round(1).astype(str) + "%"
        display_df["f1_score"] = display_df["f1_score"].round(3)
        display_df["recall"] = (display_df["recall"] * 100).round(1).astype(str) + "%"
        display_df["daily_improvement"] = (display_df["daily_improvement"] * 100).round(1).astype(str) + "%"
        display_df["prediction_rate"] = (display_df["prediction_rate"] * 100).round(1).astype(str) + "%"

        # Handle growth lift formatting
        if "avg_growth_lift" in display_df.columns:
            display_df["avg_growth_lift"] = display_df["avg_growth_lift"].apply(
                lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A"
            )

        print(display_df.to_string(index=False))

        # Precision tiers analysis
        print(f"\nPRECISION TIER ANALYSIS:")
        print("-" * 60)

        high_precision = results_df[results_df["precision"] >= 0.6]
        medium_precision = results_df[(results_df["precision"] >= 0.4) & (results_df["precision"] < 0.6)]
        low_precision = results_df[results_df["precision"] < 0.4]

        tiers = [
            ("High Precision (≥60%)", high_precision),
            ("Medium Precision (40-60%)", medium_precision),
            ("Low Precision (<40%)", low_precision),
        ]

        for tier_name, tier_df in tiers:
            if len(tier_df) > 0:
                avg_precision = tier_df["precision"].mean()
                avg_improvement = tier_df["daily_improvement"].mean()
                avg_growth = tier_df["avg_growth_lift"].mean() if "avg_growth_lift" in tier_df else np.nan

                print(f"  {tier_name}: {len(tier_df)} strategies")
                print(f"    Avg Precision: {avg_precision:.1%}")
                print(f"    Avg Daily Improvement: {avg_improvement:.1%}")
                if not pd.isna(avg_growth):
                    print(f"    Avg Growth Lift: {avg_growth:.1%}")
                print(f"    Best Strategy: {tier_df.iloc[0]['strategy']}")

        # Category analysis by precision
        print(f"\nSTRATEGY CATEGORY ANALYSIS (by precision):")
        print("-" * 60)

        categories = {
            "Manual Rules": results_df[results_df["strategy"].str.contains("manual")],
            "ML Thresholds": results_df[results_df["strategy"].str.contains("rf_thresh") | results_df["strategy"].str.contains("rf_auto_rate")],
            "Ensemble": results_df[results_df["strategy"].str.contains("ensemble") | results_df["strategy"].str.contains("ens_")],
        }

        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                avg_precision = cat_df["precision"].mean()
                max_precision = cat_df["precision"].max()
                best_strategy = cat_df.iloc[0]["strategy"] if len(cat_df) > 0 else "N/A"
                avg_improvement = cat_df["daily_improvement"].mean()

                print(f"  {cat_name}: Avg Precision={avg_precision:.1%}, Max Precision={max_precision:.1%}")
                print(f"    Best: {best_strategy} | Avg Daily Improvement: {avg_improvement:.1%}")

        # Investment recommendation based on precision
        print(f"\nINVESTMENT STRATEGY RECOMMENDATIONS:")
        print("-" * 60)

        conservative_strategies = results_df[results_df["precision"] >= 0.6]
        if len(conservative_strategies) > 0:
            best_conservative = conservative_strategies.iloc[0]
            print(f"🛡️  CONSERVATIVE (High Precision): {best_conservative['strategy']}")
            print(f"     Precision: {best_conservative['precision']:.1%}")
            print(f"     Hit Rate: {best_conservative['daily_hit_rate']:.1%}")
            print(f"     Frequency: {best_conservative['prediction_rate']:.1%} of time")

        balanced_strategies = results_df[
            (results_df["precision"] >= 0.4) &
            (results_df["f1_score"] >= 0.3) &
            (results_df["prediction_rate"] >= 0.05)
        ]
        if len(balanced_strategies) > 0:
            best_balanced = balanced_strategies.iloc[0]
            print(f"⚖️  BALANCED (Good Precision + Activity): {best_balanced['strategy']}")
            print(f"     Precision: {best_balanced['precision']:.1%}")
            print(f"     F1: {best_balanced['f1_score']:.3f}")
            print(f"     Daily Improvement: {best_balanced['daily_improvement']:.1%}")


# =========================
# Utilities
# =========================
def _print_proba_diagnostics(df: pd.DataFrame, proba_col: str = "rf_prob_30d", split: str = "test"):
    sub = df[df["split"] == split]
    if proba_col not in sub.columns or sub.empty:
        return
    q = sub[proba_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    print("\nProbability diagnostics on", split, "split:")
    for p, v in q.items():
        print(f"  q{int(p*100):02d}: {v:.3f}")


# =========================
# Orchestration
# =========================
def compare_predictions_on_existing_data() -> pd.DataFrame:
    """Run predict-only comparison on existing data & model"""
    print("=" * 80)
    print("PREDICTION COMPARISON PIPELINE (Predict-Only)")
    print("=" * 80)

    # Step 1: Load data
    print("Step 1: Loading existing data...")
    final_data = load_latest_data()
    print(f"✓ Loaded dataset: {final_data.shape}")
    if "Date" in final_data.columns:
        print(f"  Date range: {final_data['Date'].min()} to {final_data['Date'].max()}")
    if "Ticker" in final_data.columns:
        print(f"  Tickers: {final_data['Ticker'].nunique()}")

    # Step 2: Prepare (splits/target only; no training)
    print("\nStep 2: Preparing data for modeling (splits/target)...")
    tm = TrainModel(_TransformAdapter(final_data))
    tm.prepare_dataframe(start_date="2000-01-01")

    # Temporal split info
    if hasattr(tm, "df_full"):
        tgt = tm.target_col if hasattr(tm, "target_col") else "is_positive_growth_30d_future"
        split_summary = tm.df_full.groupby("split").agg({
            "Date": ["min", "max", "count"],
            tgt: "mean"
        }).round(3)
        print("Temporal split summary:")
        print(split_summary)

    # Step 3: Load model + feature order (predict-only)
    print("\nStep 3: Loading model (predict-only)...")
    sk_model, feature_cols, target_col_from_model = load_model_and_features(ARTIFACTS_DIR)
    tm.model = sk_model
    tm._inference_feature_columns = feature_cols
    if target_col_from_model:
        tm.target_col = target_col_from_model
    target_col = getattr(tm, "target_col", "is_positive_growth_30d_future")
    print(f"✓ Loaded model; using {len(feature_cols)} features for inference")
    print(f"  Target column: {target_col}")

    # Leak check: no 'future' features should be in inference list
    leak_feats = [c for c in feature_cols if 'future' in c.lower()]
    if leak_feats:
        print("WARNING: potential leakage features found in model inputs:")
        for lf in leak_feats[:10]:
            print("  -", lf)


    # Step 4: Generate predictions
    print("\nStep 4: Generating predictions (manual, ML, ensemble)...")
    comparator = PredictionComparator(tm.df_full, target_col)
    comparator.add_manual_predictions()
    comparator.add_ml_predictions(tm.model, tm._inference_feature_columns, thresholds=(0.21, 0.50, 0.65, 0.80, 0.90))

    # Optional: inspect where probabilities sit on the test split
    _print_proba_diagnostics(comparator.df, "rf_prob_30d", split="test")

    # Auto-pick thresholds from VALIDATION to hit ~1%, 3%, 5% selection rates
    comparator.add_ml_thresholds_from_validation("rf_prob_30d", target_rates=(0.01, 0.03, 0.05))

    # Add a daily Top-3 strategy
    comparator.add_daily_topn(proba_col="rf_prob_30d", n=3)

    # Build ensembles using the new, tighter signals
    comparator.add_ensemble_predictions()

    # Step 5: Evaluate
    print(f"\nStep 5: Evaluating {len(comparator.prediction_cols)} strategies...")
    test_results = comparator.evaluate_all_predictions(split="test")
    comparator.print_comparison_report(test_results)

    # Step 6: Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(
        RESULTS_DIR,
        f"prediction_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    if test_results is not None and not test_results.empty:
        test_results.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")

    # Save full dataset with predictions (Parquet, fallback to CSV)
    pred_out_file = os.path.join(
        RESULTS_DIR,
        f"predictions_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    try:
        comparator.df.to_parquet(pred_out_file, index=False)
        print(f"💾 Full dataset with predictions saved to: {pred_out_file}")
    except Exception as e:
        csv_out_file = pred_out_file.replace(".parquet", ".csv")
        comparator.df.to_csv(csv_out_file, index=False)
        print(f"⚠️ Could not write Parquet ({e}). CSV saved to: {csv_out_file}")



    print("\n" + "=" * 80)
    print("PREDICTION COMPARISON COMPLETE")
    print("=" * 80)
    return test_results


def main():
    """Entry point (predict-only)"""

    # In your predictions.py, add this at the start of main():
    final_data = load_latest_data()
    target_cols = [c for c in final_data.columns if 'is_positive' in c and 'future' in c]
    print(f"Target columns found: {target_cols}")
    print(f"Will use: is_positive_growth_30d_future (if exists)")
    print("MODE: Predict-only (load data + model; no training)")
    
    results = compare_predictions_on_existing_data()

    if results is not None and not results.empty:
        top = results.iloc[0]
        print(f"\nWINNER (Highest Precision): {top['strategy']}")
        print(f"   Precision: {top['precision']:.1%}")
        print(f"   F1 Score: {top['f1_score']:.3f}")
        print(f"   Daily Improvement: {top['daily_improvement']:.1%}")
        print(f"   Prediction Rate: {top['prediction_rate']:.1%}")

        # Precision spectrum
        print(f"\nPRECISION SPECTRUM:")
        precision_ranges = [
            (0.8, 1.0, "Ultra High"),
            (0.6, 0.8, "High"),
            (0.4, 0.6, "Medium"),
            (0.0, 0.4, "Low"),
        ]
        for min_p, max_p, label in precision_ranges:
            mask = (results["precision"] >= min_p) & (results["precision"] < max_p)
            count = int(mask.sum())
            if count > 0:
                avg_improvement = results.loc[mask, "daily_improvement"].mean()
                print(f"   {label} Precision ({min_p:.0%}-{max_p:.0%}): {count} strategies, avg improvement: {avg_improvement:.1%}")


if __name__ == "__main__":
    main()