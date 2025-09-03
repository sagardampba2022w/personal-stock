import os
import re
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainConfig:
    target_preference: List[str] = None
    rf_max_depth: Optional[int] = None 
    rf_n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self):
        if self.target_preference is None:
            self.target_preference = [
                "is_positive_growth_30d_future",
                "is_positive_future_30d",
                "is_positive_growth_20d_future",
                "is_positive_growth_14d_future",
                "is_positive_growth_7d_future",
            ]

class TrainModel:
    def __init__(self, transformed):
        self.transformed_df = transformed.transformed_df.copy(deep=True)
        
        # Normalize date and ticker columns
        self._normalize_core_columns()
        
        self.model = None
        self.config = TrainConfig()

    def _normalize_core_columns(self):
        """Normalize core columns (Date, Ticker, Volume) with proper typing"""
        # Handle Date column (try multiple possible names)
        date_cols = [c for c in self.transformed_df.columns if c.lower() in ['date', 'date_x', 'date_y']]
        if date_cols:
            self.transformed_df["Date"] = pd.to_datetime(self.transformed_df[date_cols[0]]).dt.tz_localize(None)
        
        # Handle Ticker column
        ticker_cols = [c for c in self.transformed_df.columns if c.lower() in ['ticker', 'ticker_x', 'ticker_y']]
        if ticker_cols:
            self.transformed_df["Ticker"] = self.transformed_df[ticker_cols[0]].astype(str).str.upper()
        
        # Handle Volume and create ln_volume if needed
        volume_cols = [c for c in self.transformed_df.columns if c.lower() in ['volume', 'volume_x', 'volume_y']]
        if volume_cols and 'ln_volume' not in self.transformed_df.columns:
            vol_col = volume_cols[0]
            self.transformed_df["ln_volume"] = self.transformed_df[vol_col].apply(
                lambda x: np.log(x) if pd.notnull(x) and x > 0 else np.nan
            )

        # Create month/weekday if missing
        if "Date" in self.transformed_df.columns:
            if not any(c.lower().startswith('month') for c in self.transformed_df.columns):
                self.transformed_df["month"] = self.transformed_df["Date"].dt.month_name()
            if not any(c.lower().startswith('weekday') for c in self.transformed_df.columns):
                self.transformed_df["weekday"] = self.transformed_df["Date"].dt.day_name()

    def _define_feature_sets(self):
        """Define comprehensive feature sets with better coverage"""
        print("Defining feature sets...")
        
        # 1. GROWTH FEATURES (exclude any with 'future')
        self.GROWTH = [c for c in self.transformed_df.columns 
                      if c.startswith("growth_") and "future" not in c.lower()]
        
        # 2. OHLCV and basic price features
        ohlcv_candidates = ["Open", "High", "Low", "Close", "Volume", "Adj Close",
                           "open", "high", "low", "close", "volume", "adj_close"]
        self.OHLCV = [c for c in ohlcv_candidates if c in self.transformed_df.columns]
        
        # 3. CATEGORICAL (for dummy creation)
        categorical_candidates = ["Month_x", "Month", "month", "Weekday", "weekday", 
                                "Ticker", "ticker", "ticker_type", "year", "wom", "month_wom"]
        self.CATEGORICAL = [c for c in categorical_candidates if c in self.transformed_df.columns]
        
        # 4. TARGET COLUMNS (anything with 'future' that looks like a target)
        self.TO_PREDICT = [c for c in self.transformed_df.columns 
                          if "future" in c.lower() and any(keyword in c.lower() 
                          for keyword in ["is_positive", "growth", "return"])]
        
        # 5. MACRO FEATURES
        # Growth-based macro indicators
        macro_growth_pattern = re.compile(r"^growth_(btc|vix|dax|snp500|dji|epi|gold|brent_oil|crude_oil)_\d+d$", re.I)
        macro_growth = [c for c in self.transformed_df.columns if macro_growth_pattern.match(c)]
        
        # Rate-based macro indicators  
        macro_rates = [c for c in self.transformed_df.columns 
                      if c.endswith(('_yoy', '_qoq', '_mom')) or 
                      c.upper() in ['FEDFUNDS', 'DGS1', 'DGS5', 'DGS10']]
        
        self.MACRO = macro_growth + macro_rates
        
        # 6. CUSTOM NUMERICAL FEATURES
        custom_candidates = [
            # Moving averages (various naming conventions)
            "sma10", "sma20", "sma50", "sma200", "SMA10", "SMA20", "SMA50", "SMA200",
            # Other engineered features
            "growing_moving_average", "volatility", "sharpe", "high_minus_low_relative", 
            "ln_volume"
            # Note: ticker_type is categorical, moved to CATEGORICAL section
        ]
        self.CUSTOM_NUMERICAL = [c for c in custom_candidates if c in self.transformed_df.columns]
        
        # 7. TECHNICAL INDICATORS (comprehensive TA-Lib coverage)
        # Basic momentum/trend indicators
        ta_basic = ["adx", "adxr", "apo", "bop", "cci", "cmo", "dx", "mfi", "mom", "ppo",
                   "roc", "rocp", "rocr", "rocr100", "rsi", "trix", "ultosc", "willr"]
        
        # MACD family (handle naming variations)
        macd_variants = ["macd", "macd_signal", "macd_hist", "macdsignal", "macdhist",
                        "macd_ext", "macd_signal_ext", "macd_hist_ext", 
                        "macd_fix", "macd_signal_fix", "macd_hist_fix"]
        
        # Aroon indicators (handle naming variations)  
        aroon_variants = ["aroon_up", "aroon_down", "aroonosc", "aroon_1", "aroon_2"]
        
        # Stochastic indicators
        stoch_variants = ["stoch_slowk", "stoch_slowd", "stoch_fastk", "stoch_fastd",
                         "slowk", "slowd", "fastk", "fastd",
                         "stochrsi_fastk", "stochrsi_fastd", "fastk_rsi", "fastd_rsi"]
        
        # Volume indicators
        volume_indicators = ["ad", "adosc", "obv"]
        
        # Volatility indicators
        volatility_indicators = ["atr", "natr", "trange"]
        
        # Directional indicators
        directional_indicators = ["plus_di", "minus_di", "plus_dm"]
        
        # Price transform indicators
        price_indicators = ["avgprice", "medprice", "typprice", "wclprice"]
        
        # Hilbert Transform indicators
        ht_indicators = ["ht_dcperiod", "ht_dcphase", "ht_phasor_inphase", "ht_phasor_quadrature",
                        "ht_sine_sine", "ht_sine_leadsine", "ht_trendmode", "ht_trendmod"]
        
        # Combine all TA indicators
        all_ta_candidates = (ta_basic + macd_variants + aroon_variants + stoch_variants + 
                           volume_indicators + volatility_indicators + directional_indicators +
                           price_indicators + ht_indicators)
        
        self.TECHNICAL_INDICATORS = [c for c in all_ta_candidates if c in self.transformed_df.columns]
        
        # 8. CANDLESTICK PATTERNS
        self.TECHNICAL_PATTERNS = [c for c in self.transformed_df.columns if c.startswith("cdl")]
        
        # 9. ALL NUMERICAL FEATURES (for modeling)
        self.NUMERICAL = list(dict.fromkeys(
            self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + 
            self.CUSTOM_NUMERICAL + self.MACRO
        ))
        
        # 10. COLUMNS TO DROP (not used in modeling)
        drop_candidates = ["Date", "date", "Year", "year", "Month_x", "Month_y", "Month", "month",
                          "index", "Quarter", "index_x", "index_y", "split"] + self.CATEGORICAL + self.OHLCV
        self.TO_DROP = [c for c in drop_candidates if c in self.transformed_df.columns]
        
        # 11. DEBUG: Identify unused columns
        used_cols = set(self.NUMERICAL + self.TO_DROP + self.TO_PREDICT + self.CATEGORICAL + self.OHLCV)
        self.OTHER = [c for c in self.transformed_df.columns if c not in used_cols]
        
        # Print summary
        print(f"Feature Set Summary:")
        print(f"  Growth features: {len(self.GROWTH)}")
        print(f"  Technical indicators: {len(self.TECHNICAL_INDICATORS)}")
        print(f"  Technical patterns: {len(self.TECHNICAL_PATTERNS)}")
        print(f"  Custom numerical: {len(self.CUSTOM_NUMERICAL)}")
        print(f"  Macro features: {len(self.MACRO)}")
        print(f"  Categorical (for dummies): {len(self.CATEGORICAL)}")
        print(f"  Target columns: {len(self.TO_PREDICT)}")
        print(f"  Total numerical features: {len(self.NUMERICAL)}")
        print(f"  Unused columns: {len(self.OTHER)}")
        
        if self.OTHER:
            print(f"  Unused sample: {self.OTHER[:10]}")

    def _define_dummies(self):
        """Create dummy variables from categorical columns"""
        print("Creating dummy variables...")
        
        if self.CATEGORICAL:
            # Ensure categorical columns are string type
            for c in self.CATEGORICAL:
                self.transformed_df[c] = self.transformed_df[c].astype(str)
            
            # Create dummies
            dummy_variables = pd.get_dummies(
                self.transformed_df[self.CATEGORICAL], 
                dtype="int32",
                drop_first=False  # Keep all dummies for now
            )
            
            # Concatenate with original dataframe
            self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
            self.DUMMIES = list(dummy_variables.columns)
            
            print(f"Created {len(self.DUMMIES)} dummy variables")
            print(f"Sample dummies: {self.DUMMIES[:5]}")
            
        else:
            self.df_full = self.transformed_df.copy(deep=True)
            self.DUMMIES = []
            print("No categorical variables found for dummy creation")

    def _perform_temporal_split(self, df: pd.DataFrame, min_date, max_date, 
                               train_prop=0.7, val_prop=0.15, test_prop=0.15):
        """Create temporal train/validation/test splits"""
        if "Date" not in df.columns:
            raise ValueError("Expected a 'Date' column for temporal split.")
        
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        total_days = (max_date - min_date).days
        train_end = min_date + pd.Timedelta(days=int(total_days * train_prop))
        val_end = train_end + pd.Timedelta(days=int(total_days * val_prop))

        def assign_bucket(dt):
            if dt <= train_end:
                return "train"
            elif dt <= val_end:
                return "validation"
            else:
                return "test"

        df["split"] = df["Date"].apply(assign_bucket)
        
        # Print split summary
        split_counts = df["split"].value_counts()
        print(f"Temporal split created:")
        for split_name in ["train", "validation", "test"]:
            if split_name in split_counts:
                print(f"  {split_name}: {split_counts[split_name]:,} samples")
        
        return df

    def _choose_target(self) -> str:
        """Choose target column based on preference order"""
        for t in self.config.target_preference:
            if t in self.TO_PREDICT:
                print(f"Selected target: {t}")
                return t
        
        # Fallback to any column starting with is_positive
        alt = [c for c in self.TO_PREDICT if c.lower().startswith("is_positive")]
        if alt:
            print(f"Selected fallback target: {alt[0]}")
            return alt[0]
        
        if self.TO_PREDICT:
            print(f"Selected first available target: {self.TO_PREDICT[0]}")
            return self.TO_PREDICT[0]
        
        raise ValueError(f"No suitable target column found. Available targets: {self.TO_PREDICT}")

    def prepare_dataframe(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Main pipeline to prepare data for modeling"""
        print("Preparing dataframe for modeling...")
        
        # 1. Define feature sets
        self._define_feature_sets()
        
        # 2. Create dummy variables
        self._define_dummies()

        # 3. Validate Date column
        if "Date" not in self.df_full.columns:
            raise ValueError("Missing 'Date' column in dataframe.")

        # 4. Normalize dates
        self.df_full["Date"] = pd.to_datetime(self.df_full["Date"]).dt.tz_localize(None)

        # 5. Optional date filtering
        if start_date is not None:
            start = pd.Timestamp(start_date).tz_localize(None)
            self.df_full = self.df_full[self.df_full["Date"] >= start].copy()
            print(f"Filtered data from {start_date}")

        if end_date is not None:
            end = pd.Timestamp(end_date).tz_localize(None)
            self.df_full = self.df_full[self.df_full["Date"] <= end].copy()
            print(f"Filtered data until {end_date}")

        # 6. Create temporal splits
        min_date_df = self.df_full["Date"].min()
        max_date_df = self.df_full["Date"].max()
        print(f"Date range: {min_date_df} to {max_date_df}")
        
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

        # 7. Create ML datasets
        self._define_dataframes_for_ML()

    def _define_dataframes_for_ML(self):
        """Create train/validation/test datasets with proper feature selection"""
        print("Creating ML datasets...")
        
        # *** KEY FIX: Include dummy variables in features ***
        features_list = list(dict.fromkeys(self.NUMERICAL + self.DUMMIES))
        print(f"Total features before filtering: {len(features_list)}")
        print(f"  - Numerical: {len(self.NUMERICAL)}")
        print(f"  - Dummies: {len(self.DUMMIES)}")
        
        # Filter out any features containing 'future' (avoid data leakage)
        features_list = [c for c in features_list if "future" not in c.lower()]
        print(f"Features after removing 'future': {len(features_list)}")
        
        # *** CRITICAL FIX: Remove any string/categorical columns that shouldn't be in features ***
        # Check for any remaining categorical columns that slipped through
        string_cols = []
        for col in features_list:
            if col in self.df_full.columns:
                if self.df_full[col].dtype == 'object' or self.df_full[col].dtype.name == 'string':
                    string_cols.append(col)
        
        if string_cols:
            print(f"WARNING: Removing string columns from features: {string_cols}")
            features_list = [c for c in features_list if c not in string_cols]
            print(f"Features after removing string columns: {len(features_list)}")
        
        # Choose target
        target = self._choose_target()
        self.target_col = target

        # Create split datasets
        self.train_df = self.df_full[self.df_full["split"] == "train"].copy(deep=True)
        self.valid_df = self.df_full[self.df_full["split"] == "validation"].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full["split"].isin(["train","validation"])].copy(deep=True)
        self.test_df = self.df_full[self.df_full["split"] == "test"].copy(deep=True)

        def build_X_y(df):
            """Build feature matrix and target vector"""
            # Get available features (some might be missing in certain splits)
            available_features = [c for c in features_list if c in df.columns]
            
            if target not in df.columns:
                raise ValueError(f"Target '{target}' not found in {df['split'].iloc[0] if 'split' in df else 'unknown'} split")
            
            X = df[available_features].copy(deep=True)
            y = df[target].copy()
            
            # Final validation: ensure all features are numeric
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'string':
                    print(f"ERROR: Column '{col}' is still string type: {X[col].dtype}")
                    raise ValueError(f"Non-numeric column '{col}' found in feature matrix")
            
            return X, y

        # Create feature matrices and target vectors
        self.X_train, self.y_train = build_X_y(self.train_df)
        self.X_valid, self.y_valid = build_X_y(self.valid_df)
        self.X_train_valid, self.y_train_valid = build_X_y(self.train_valid_df)
        self.X_test, self.y_test = build_X_y(self.test_df)
        self.X_all, self.y_all = build_X_y(self.df_full)

        # Clean datasets (remove inf/nan)
        self.X_train = self._clean_dataframe_from_inf_and_nan(self.X_train)
        self.X_valid = self._clean_dataframe_from_inf_and_nan(self.X_valid)
        self.X_train_valid = self._clean_dataframe_from_inf_and_nan(self.X_train_valid)
        self.X_test = self._clean_dataframe_from_inf_and_nan(self.X_test)
        self.X_all = self._clean_dataframe_from_inf_and_nan(self.X_all)

        # Print summary
        print(f"ML Dataset Summary:")
        print(f"  Features used: {len(self.X_train.columns)}")
        print(f"  Train: {self.X_train.shape[0]:,} samples")
        print(f"  Validation: {self.X_valid.shape[0]:,} samples") 
        print(f"  Test: {self.X_test.shape[0]:,} samples")
        print(f"  Train+Valid: {self.X_train_valid.shape[0]:,} samples")
        print(f"  Target: {self.target_col}")
        print(f"  Target distribution (train): {self.y_train.value_counts().to_dict()}")

    @staticmethod
    def _clean_dataframe_from_inf_and_nan(df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf with NaN, then fill NaN with 0"""
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.fillna(0)

    def debug_feature_coverage(self):
        """Debug which features are used vs unused"""
        all_cols = set(self.df_full.columns) 
        used_in_model = set(self.X_train_valid.columns)
        dropped_cols = set(self.TO_DROP)
        future_cols = {c for c in all_cols if "future" in c.lower()}
        target_col = {self.target_col}
        
        unused_cols = all_cols - used_in_model - dropped_cols - future_cols - target_col
        
        print(f"\nFeature Coverage Debug:")
        print(f"  Total columns in df_full: {len(all_cols)}")
        print(f"  Used in model: {len(used_in_model)}")
        print(f"  Dropped (OHLCV/categorical/etc): {len(dropped_cols)}")
        print(f"  Future columns (targets): {len(future_cols)}")
        print(f"  Target column: 1")
        print(f"  Unused (non-future): {len(unused_cols)}")
        
        if unused_cols:
            unused_list = sorted(list(unused_cols))
            print(f"  Sample unused: {unused_list[:20]}")
            
        # Check for potential missing features
        potential_missing = []
        for col in unused_cols:
            if any(pattern in col.lower() for pattern in ['growth_', 'sma', 'rsi', 'macd', 'volatil']):
                potential_missing.append(col)
        
        if potential_missing:
            print(f"  Potentially useful unused features: {potential_missing[:10]}")

    # [Keep all the existing training, evaluation, and persistence methods unchanged]
    def train_random_forest(self, max_depth: Optional[int] = None, n_estimators: Optional[int] = None,
                             train_on: str = 'train', class_weight: Optional[str] = None):
        from sklearn.ensemble import RandomForestClassifier
        max_depth = self.config.rf_max_depth if max_depth is None else max_depth
        n_estimators = self.config.rf_n_estimators if n_estimators is None else n_estimators

        print(f"Training RandomForestClassifier (max_depth={max_depth}, n_estimators={n_estimators}, train_on={train_on}, class_weight={class_weight})")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            class_weight=class_weight
        )
        if train_on not in ('train', 'train_valid'):
            raise ValueError("train_on must be 'train' or 'train_valid'")
        X_fit, y_fit = (self.X_train, self.y_train) if train_on == 'train' else (self.X_train_valid, self.y_train_valid)
        self.model.fit(X_fit, y_fit)

    def find_best_threshold(self, split: str = "validation", metric: str = "f1", grid: Optional[List[float]] = None) -> float:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
        y_true, proba = self._get_split_arrays(split)
        if grid is None:
            grid = [i/100 for i in range(1,100)]  # 0.01 ... 0.99

        if metric == "youden":
            fpr, tpr, thr = roc_curve(y_true, proba)
            youden = tpr - fpr
            idx = int(np.argmax(youden))
            return float(thr[idx])

        best_thr, best_val = 0.5, -1.0
        for t in grid:
            y_pred = (proba >= t).astype(int)
            if metric == "f1":
                val = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                val = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                val = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError("metric must be one of: 'f1', 'youden', 'precision', 'recall'")
            if val > best_val:
                best_val, best_thr = val, t
        return float(best_thr)

    def _get_split_arrays(self, split: str):
        if self.model is None:
            raise ValueError("Model not trained/loaded.")
        s = split.lower()
        if s == "train":
            X, y = self.X_train, self.y_train
        elif s in ("valid","validation"):
            X, y = self.X_valid, self.y_valid
        elif s == "test":
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("split must be one of: 'train', 'validation', 'test'")
        proba = self.model.predict_proba(X)[:, 1]
        return y.values, proba

    def summarize_performance(self, threshold: float = 0.5, splits: tuple = ("validation","test")) -> pd.DataFrame:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix,
            brier_score_loss
        )
        rows = []
        for s in splits:
            y_true, proba = self._get_split_arrays(s)
            if len(y_true) == 0:
                continue

            y_pred = (proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
                fp = cm[0, 1] if cm.shape[1] > 1 else 0
                fn = cm[1, 0] if cm.shape[0] > 1 else 0
                tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

            # Core metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Extras
            pred_pos_rate = float(y_pred.mean())
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            bal_acc = (rec + specificity) / 2 if not (np.isnan(specificity)) else float("nan")
            roc_auc = float(roc_auc_score(y_true, proba)) if np.unique(y_true).size > 1 else float("nan")
            pr_auc = float(average_precision_score(y_true, proba)) if y_true.sum() > 0 else float("nan")
            brier = float(brier_score_loss(y_true, proba))

            row = {
                "split": s,
                "n_samples": int(len(y_true)),
                "pos_rate": float(np.mean(y_true)),
                "pred_pos_rate": pred_pos_rate,
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "specificity": float(specificity),
                "balanced_accuracy": float(bal_acc),
                "f1": float(f1),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier": brier,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "threshold": threshold,
            }
            rows.append(row)

        cols = ["split","n_samples","pos_rate","pred_pos_rate","accuracy","precision",
                "recall","specificity","balanced_accuracy","f1","roc_auc","pr_auc","brier",
                "tn","fp","fn","tp","threshold"]
        return pd.DataFrame(rows, columns=cols)

    def text_classification_report(self, split: str = "test", threshold: float = 0.5) -> str:
        from sklearn.metrics import classification_report
        y_true, proba = self._get_split_arrays(split)
        y_pred = (proba >= threshold).astype(int)
        return classification_report(y_true, y_pred, digits=4, zero_division=0)

    def refit_on_train_valid(self, class_weight: Optional[str] = None):
        if self.model is None:
            raise ValueError("Call train_random_forest once to set hyperparameters.")
        params = self.model.get_params()
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**{**params, "class_weight": class_weight})
        self.model.fit(self.X_train_valid, self.y_train_valid)

    def persist(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        model_path = os.path.join(data_dir, "random_forest_model.joblib")
        meta_path = os.path.join(data_dir, "rf_meta.json")

        joblib.dump(self.model, model_path)

        feature_columns = self.X_train_valid.columns.tolist()
        
        pd.Series({
            "target_col": self.target_col,
            "feature_columns": feature_columns,
        }).to_json(meta_path)

        print(f"Saved model to {model_path}")
        print(f"Saved meta to {meta_path}")

    def load(self, data_dir: str):
        model_path = os.path.join(data_dir, "random_forest_model.joblib")
        meta_path = os.path.join(data_dir, "rf_meta.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta not found at: {meta_path}")
        self.model = joblib.load(model_path)
        meta = pd.read_json(meta_path, typ="series")
        self.target_col = meta["target_col"]
        self._inference_feature_columns = list(meta["feature_columns"])

    def make_inference(self, pred_name: str = "rf_pred"):
        """Generate predictions on the full dataset"""
        if self.model is None:
            raise ValueError("Model is not loaded/trained. Call train_random_forest() or load() first.")

        # Get the feature columns used during training
        feat_cols_saved = getattr(self, "_inference_feature_columns", self.X_train_valid.columns.tolist())
        
        # Use df_full if available, else transformed_df
        base_df = getattr(self, "df_full", self.transformed_df)
        
        # Build feature matrix matching training features
        missing_features = []
        X_cols = []
        
        for col in feat_cols_saved:
            if col in base_df.columns:
                X_cols.append(base_df[col])
            else:
                # Missing feature - fill with zeros
                missing_features.append(col)
                X_cols.append(pd.Series(0, index=base_df.index))
        
        if missing_features:
            print(f"WARNING: {len(missing_features)} features missing during inference, filling with 0")
            print(f"Sample missing: {missing_features[:5]}")
        
        # Create feature matrix
        X = pd.DataFrame(dict(zip(feat_cols_saved, X_cols)), index=base_df.index)
        X = self._clean_dataframe_from_inf_and_nan(X)

        # Generate predictions
        y_pred_proba = self.model.predict_proba(X)
        class1_proba = y_pred_proba[:, 1]  # Probability of positive class

        # Add predictions to dataframe
        base_df[pred_name] = class1_proba

        # Add ranking by date if Date column exists
        if "Date" in base_df.columns:
            base_df[f"{pred_name}_rank"] = base_df.groupby("Date")[pred_name].rank(method="first", ascending=False)
        else:
            base_df[f"{pred_name}_rank"] = base_df[pred_name].rank(method="first", ascending=False)

        print(f"Generated predictions '{pred_name}' and '{pred_name}_rank'")
        
        # Return subset for inspection
        keep_cols = [pred_name, f"{pred_name}_rank"]
        if "Date" in base_df.columns:
            keep_cols.append("Date")
        if "Ticker" in base_df.columns:
            keep_cols.append("Ticker")
            
        return base_df[keep_cols]

    def daily_topk_stats(self, k: int = 5, split: str = "test", pred_col: str = "rf_prob_30d") -> dict:
        """Calculate daily top-k performance statistics"""
        if "split" not in self.df_full.columns:
            raise ValueError("df_full is missing 'split' column; call prepare_dataframe() first.")
        if pred_col not in self.df_full.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found. Run make_inference('{pred_col}') first.")
        if self.target_col not in self.df_full.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in df_full.")

        # Filter to specified split
        d = self.df_full[self.df_full["split"] == split].copy()
        if "Date" not in d.columns:
            raise ValueError("df_full is missing 'Date' column needed for daily Top-K stats.")
        
        # Sort by date and prediction score
        d = d.sort_values(["Date", pred_col], ascending=[True, False])

        # Calculate baseline hit rate (all stocks)
        base_daily = d.groupby("Date")[self.target_col].mean().mean()
        
        # Calculate top-k hit rate
        topk = d.groupby("Date").head(k)
        top_daily = topk.groupby("Date")[self.target_col].mean().mean()

        # Calculate growth lift if growth column exists
        lift = None
        growth_col = "growth_future_30d"
        if growth_col in d.columns:
            base_growth = d.groupby("Date")[growth_col].mean().mean()
            topk_growth = topk.groupby("Date")[growth_col].mean().mean()
            lift = topk_growth - base_growth

        return {
            "split": split, 
            "k": k, 
            "daily_hitrate_baseline": float(base_daily), 
            "daily_hitrate_topk": float(top_daily), 
            "avg_growth_lift_topk_vs_all": lift
        }