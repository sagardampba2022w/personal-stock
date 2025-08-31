
import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# Ensure local imports (stock_pipeline.py, tickerfetch.py) resolve
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

from stock_pipeline import StockDataPipeline
from tickerfetch import get_combined_universe

# ============================
# User-provided training code
# ============================

@dataclass
class TrainConfig:
    target_preference: List[str] = None
    rf_max_depth: int = 17
    rf_n_estimators: int = 200
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self):
        if self.target_preference is None:
            # Prefer 30d horizons first, then anything with 'is_positive' + 'future'
            self.target_preference = [
                "is_positive_growth_30d_future",
                "is_positive_future_30d",
                "is_positive_growth_20d_future",
                "is_positive_growth_14d_future",
                "is_positive_growth_7d_future",
            ]

class TrainModel:
    # Dataframes
    transformed_df: pd.DataFrame          # input from your transform stage
    df_full: pd.DataFrame                 # with dummies and splits
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    train_valid_df: pd.DataFrame
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    X_train_valid: pd.DataFrame
    X_all: pd.DataFrame

    # Feature sets
    GROWTH: List[str]
    OHLCV: List[str]
    CATEGORICAL: List[str]
    TO_PREDICT: List[str]
    TECHNICAL_INDICATORS: List[str]
    TECHNICAL_PATTERNS: List[str]
    MACRO: List[str]
    CUSTOM_NUMERICAL: List[str]
    NUMERICAL: List[str]
    DUMMIES: List[str]
    TO_DROP: List[str]
    OTHER: List[str]

    def __init__(self, transformed):
        # Expect object with .transformed_df (like your TransformData)
        self.transformed_df = transformed.transformed_df.copy(deep=True)

        # Ensure canonical types
        if "Date" in self.transformed_df.columns:
            self.transformed_df["Date"] = pd.to_datetime(self.transformed_df["Date"]).dt.tz_localize(None)
        if "Ticker" in self.transformed_df.columns:
            self.transformed_df["Ticker"] = self.transformed_df["Ticker"].astype(str).str.upper()

        # Safe ln_volume
        if "Volume" in self.transformed_df.columns:
            self.transformed_df["ln_volume"] = self.transformed_df["Volume"].apply(lambda x: np.log(x) if pd.notnull(x) and x > 0 else np.nan)

        # Derive Month/Weekday if missing (keeps Month_x if supplied by your joins)
        if "Month_x" not in self.transformed_df.columns:
            if "Date" in self.transformed_df.columns:
                self.transformed_df["Month"] = self.transformed_df["Date"].dt.month_name()
            else:
                self.transformed_df["Month"] = np.nan
        if "Weekday" not in self.transformed_df.columns:
            if "Date" in self.transformed_df.columns:
                self.transformed_df["Weekday"] = self.transformed_df["Date"].dt.day_name()
            else:
                self.transformed_df["Weekday"] = np.nan

        # a placeholder for the trained model
        self.model = None
        self.config = TrainConfig()

    # ---------- Feature set helpers ----------
    def _present(self, cols: List[str]) -> List[str]:
        """Return only the columns present in the dataframe."""
        return [c for c in cols if c in self.transformed_df.columns]

    def _regex_cols(self, pattern: str) -> List[str]:
        rgx = re.compile(pattern)
        return [c for c in self.transformed_df.columns if rgx.search(c)]

    def _define_feature_sets(self):
        # Growth features (exclude future labels)
        self.GROWTH = [g for g in self.transformed_df.columns if g.startswith("growth_") and "future" not in g]

        # OHLCV (keep only those that exist)
        self.OHLCV = self._present(["Open", "High", "Low", "Close", "Volume"])

        # Categorical candidates (use what's present)
        categorical_candidates = ["Month_x", "Month", "Weekday", "Ticker", "ticker_type"]
        self.CATEGORICAL = self._present(categorical_candidates)

        # Targets: Anything with "future" + preferably "is_positive"
        self.TO_PREDICT = self._regex_cols(r"is_positive.*future|future.*is_positive|is_positive_future")

        # Macro set — include only if present
        macro_candidates = [
            # FRED
            "gdppot_us_yoy", "gdppot_us_qoq",
            "cpi_core_yoy", "cpi_core_mom",
            "FEDFUNDS", "DGS1", "DGS5", "DGS10",
            # Market proxies from Yahoo (commonly merged)
            "vix_adj_close", "btc_usd_adj_close", "gdaxi_adj_close",
        ]
        self.MACRO = self._present(macro_candidates)

        # Custom numerical columns, if they exist
        self.CUSTOM_NUMERICAL = self._present([
            "SMA10", "SMA20", "growing_moving_average", "high_minus_low_relative",
            "volatility", "ln_volume", "SMA50", "SMA200",
        ])

        # TA-Lib indicators (canonical list) — filtered to what's present
        canonical_ta = [
            "adx", "adxr", "apo", "aroon_1", "aroon_2", "aroonosc", "bop", "cci", "cmo", "dx",
            "macd", "macdsignal", "macdhist",
            "macd_ext", "macdsignal_ext", "macdhist_ext",
            "macd_fix", "macdsignal_fix", "macdhist_fix",
            "mfi", "minus_di", "mom", "plus_di", "ppo",
            "roc", "rocp", "rocr", "rocr100", "rsi",
            "slowk", "slowd", "fastk", "fastd", "fastk_rsi", "fastd_rsi",
            "trix", "ultosc", "willr",
            "ad", "adosc", "obv", "atr", "natr",
            "ht_dcperiod", "ht_dcphase", "ht_phasor_inphase", "ht_phasor_quadrature",
            "ht_sine_sine", "ht_sine_leadsine", "ht_trendmod",
            "avgprice", "medprice", "typprice", "wclprice"
        ]
        self.TECHNICAL_INDICATORS = self._present(canonical_ta)

        # Candlestick patterns
        self.TECHNICAL_PATTERNS = [c for c in self.transformed_df.columns if c.startswith("cdl")]

        # Numerical = all numeric feature groups we will feed into the model
        self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + self.CUSTOM_NUMERICAL + self.MACRO

        # Drop list: artifacts or unused originals — keep only those present
        drop_candidates = ["Year", "Date", "Month_x", "Month_y", "index", "Quarter", "index_x", "index_y"]
        self.TO_DROP = self._present(drop_candidates) + self.CATEGORICAL + self.OHLCV

        # Check if anything else is left (for debugging/inspection)
        self.OTHER = [k for k in self.transformed_df.columns if k not in set(self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT)]

    def _define_dummies(self):
        # Guarantee string dtype for categorical dims (skip if not present)
        if "Month_x" in self.CATEGORICAL:
            # Some pipelines set Month_x as datetime → stringify to form groups
            self.transformed_df["Month_x"] = self.transformed_df["Month_x"].astype(str)
        if "Month" in self.CATEGORICAL:
            self.transformed_df["Month"] = self.transformed_df["Month"].astype(str)
        if "Weekday" in self.CATEGORICAL:
            self.transformed_df["Weekday"] = self.transformed_df["Weekday"].astype(str)
        if "Ticker" in self.CATEGORICAL:
            self.transformed_df["Ticker"] = self.transformed_df["Ticker"].astype(str)
        if "ticker_type" in self.CATEGORICAL:
            self.transformed_df["ticker_type"] = self.transformed_df["ticker_type"].astype(str)

        if len(self.CATEGORICAL) > 0:
            dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype="int32")
            self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
            self.DUMMIES = list(dummy_variables.columns)
        else:
            self.df_full = self.transformed_df.copy(deep=True)
            self.DUMMIES = []

    def _perform_temporal_split(self, df: pd.DataFrame, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
        if "Date" not in df.columns:
            raise ValueError("Expected a 'Date' column for temporal split.")

        # Normalize to datetime
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
        return df

    def _choose_target(self) -> str:
        # Prefer in config order if present
        for t in self.config.target_preference:
            if t in self.TO_PREDICT:
                return t

        # Otherwise pick most common pattern: 'is_positive.*future'
        alt = [c for c in self.TO_PREDICT if c.startswith("is_positive")]
        if alt:
            return alt[0]

        # Otherwise any column with 'future'
        if self.TO_PREDICT:
            return self.TO_PREDICT[0]

        raise ValueError("No suitable target column found. Ensure a column like 'is_positive_growth_30d_future' exists.")

    # ---------- Public pipeline ----------
    def prepare_dataframe(self):
        self._define_feature_sets()
        self._define_dummies()

        # Temporal split
        if "Date" not in self.df_full.columns:
            raise ValueError("Missing 'Date' column in dataframe.")
        min_date_df = pd.to_datetime(self.df_full["Date"]).min()
        max_date_df = pd.to_datetime(self.df_full["Date"]).max()
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

        self._define_dataframes_for_ML()

    def _define_dataframes_for_ML(self):
        features_list = list(dict.fromkeys(self.NUMERICAL + self.DUMMIES))  # de-dup, preserve order
        target = self._choose_target()
        self.target_col = target  # persist for inference

        # Partition
        self.train_df = self.df_full[self.df_full["split"] == "train"].copy(deep=True)
        self.valid_df = self.df_full[self.df_full["split"] == "validation"].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full["split"].isin(["train", "validation"])].copy(deep=True)
        self.test_df = self.df_full[self.df_full["split"] == "test"].copy(deep=True)

        # Build X matrices including the target (to clean jointly), then split y
        def build_X(df):
            cols = [c for c in features_list if c in df.columns] + ([target] if target in df.columns else [])
            return df[cols].copy(deep=True)

        self.X_train = build_X(self.train_df)
        self.X_valid = build_X(self.valid_df)
        self.X_train_valid = build_X(self.train_valid_df)
        self.X_test = build_X(self.test_df)
        self.X_all = build_X(self.df_full)

        # Clean NaN/inf
        self.X_train = self._clean_dataframe_from_inf_and_nan(self.X_train)
        self.X_valid = self._clean_dataframe_from_inf_and_nan(self.X_valid)
        self.X_train_valid = self._clean_dataframe_from_inf_and_nan(self.X_train_valid)
        self.X_test = self._clean_dataframe_from_inf_and_nan(self.X_test)
        self.X_all = self._clean_dataframe_from_inf_and_nan(self.X_all)

        # y vectors
        self.y_train = self.X_train[target].copy()
        self.y_valid = self.X_valid[target].copy()
        self.y_train_valid = self.X_train_valid[target].copy()
        self.y_test = self.X_test[target].copy()
        self.y_all = self.X_all[target].copy()

        # Drop target from X
        for X in [self.X_train, self.X_valid, self.X_train_valid, self.X_test, self.X_all]:
            if target in X.columns:
                X.drop(columns=[target], inplace=True)

        print(f"length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}")
        print(f"  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}")
        print(f"Target used: {self.target_col}")

    @staticmethod
    def _clean_dataframe_from_inf_and_nan(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.fillna(0)

    # ---------- Model ----------
    def train_random_forest(self, max_depth: Optional[int] = None, n_estimators: Optional[int] = None):
        from sklearn.ensemble import RandomForestClassifier  # local import to keep module import light

        max_depth = self.config.rf_max_depth if max_depth is None else max_depth
        n_estimators = self.config.rf_n_estimators if n_estimators is None else n_estimators

        print(f"Training RandomForestClassifier (max_depth={max_depth}, n_estimators={n_estimators})")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        self.model.fit(self.X_train_valid, self.y_train_valid)

    # ---------- Persistence ----------
    def persist(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        model_path = os.path.join(data_dir, "random_forest_model.joblib")
        meta_path = os.path.join(data_dir, "rf_meta.json")

        joblib.dump(self.model, model_path)
        # Save columns & target for consistent inference
        pd.Series({
            "target_col": self.target_col,
            "feature_columns": self.X_train_valid.columns.tolist()
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

    # ---------- Inference ----------
    def make_inference(self, pred_name: str = "rf_pred"):
        if self.model is None:
            raise ValueError("Model is not loaded/trained. Call train_random_forest() or load() first.")

        # Ensure we use the same feature columns used during training
        feat_cols = getattr(self, "_inference_feature_columns", None)
        if feat_cols is None:
            feat_cols = self.X_train_valid.columns.tolist()  # after training within this instance

        X = self.df_full.reindex(columns=feat_cols, fill_value=0).copy()
        X = self._clean_dataframe_from_inf_and_nan(X)

        # predict_proba
        y_pred_all = self.model.predict_proba(X)
        class1 = np.array([p[1] for p in y_pred_all])

        self.df_full[pred_name] = class1

        # Rank within day (higher prob = higher rank)
        if "Date" in self.df_full.columns:
            self.df_full[f"{pred_name}_rank"] = self.df_full.groupby("Date")[pred_name].rank(method="first", ascending=False)
        else:
            self.df_full[f"{pred_name}_rank"] = self.df_full[pred_name].rank(method="first", ascending=False)

        return self.df_full[[pred_name, f"{pred_name}_rank"] + (["Date", "Ticker"] if "Date" in self.df_full.columns and "Ticker" in self.df_full.columns else [])]

# Small adapter to mimic a TransformData object (expects .transformed_df)
class _TransformAdapter:
    def __init__(self, df: pd.DataFrame):
        self.transformed_df = df

def main():
    # Get tickers
    universe_df = get_combined_universe(include_sp500=True, include_nasdaq100=True)
    tickers = universe_df['Ticker'].tolist()[:10]  # Use first 10 for testing

    # Configuration
    LOOKBACKS = [1, 3, 7, 30, 90, 252, 365]
    HORIZONS = [30]
    BINARY_THRESHOLDS = {30: 1.05}

    # Run data pipeline
    pipeline = StockDataPipeline(
        tickers=tickers,
        lookbacks=LOOKBACKS,
        horizons=HORIZONS,
        binarize_thresholds=BINARY_THRESHOLDS
    )

    print("Generating data...")
    final_data = pipeline.run_complete_pipeline()

    # Debug columns
    print("Available columns (first 10):", final_data.columns.tolist()[:10])
    print("Looking for date column in:", [col for col in final_data.columns if 'date' in col.lower()])

    # NOTE: We keep 'Date'/'Ticker' as-is since TrainModel expects those names natively.
    # If you prefer lower-case, add aliases but keep originals:
    # final_data['date'] = final_data['Date']
    # final_data['ticker'] = final_data['Ticker']

    # Train models using the provided TrainModel
    print("Training models...")
    tm = TrainModel(_TransformAdapter(final_data))
    tm.prepare_dataframe()
    tm.train_random_forest(max_depth=17, n_estimators=200)
    tm.persist("./artifacts")  # saves model + rf_meta.json

    # Inference back onto the full panel
    tm.make_inference("rf_prob_30d")

    # Show a tiny sample of predictions
    pred_cols = [c for c in tm.df_full.columns if c.startswith("rf_prob")]
    out_cols = ['Date','Ticker'] + pred_cols if set(['Date','Ticker']).issubset(tm.df_full.columns) else pred_cols
    print(tm.df_full[out_cols].tail(10))

if __name__ == "__main__":
    main()
