
import os
import re
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

# ================================================================
# Robust, alias-aware trainer that:
# - handles case/underscore name variants (macdsignal vs macd_signal, aroon_1 vs aroon_up, SMA10 vs sma10, etc.)
# - avoids leakage by filtering any feature containing "future"
# - persists feature *aliases* so future inference maps new schemas correctly
# - includes eval helpers + threshold search + proper temporal split
# ================================================================

@dataclass
class TrainConfig:
    target_preference: List[str] = None
    rf_max_depth: int = 17
    rf_n_estimators: int = 200
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
    # ---------- Name normalization helpers ----------
    # Map many possible spellings to ONE canonical alias (both for DF columns and requested names)
    _NORM_EQUIVS = {
        # Date/ticker/cats
        "date":"date","ticker":"ticker","ticker_type":"ticker_type",
        "month":"month","month_x":"month_x","weekday":"weekday","year":"year",
        "wom":"wom","month_wom":"month_wom",
        # SMA
        "sma10":"sma10","sma20":"sma20","sma50":"sma50","sma200":"sma200",
        "sma_10":"sma10","sma_20":"sma20","sma_50":"sma50","sma_200":"sma200",
        # MACD family
        "macd":"macd",
        "macdsignal":"macd_signal","macd_signal":"macd_signal",
        "macdhist":"macd_hist","macd_hist":"macd_hist",
        "macdext":"macd_ext","macd_ext":"macd_ext",
        "macdsignalext":"macd_signal_ext","macd_signal_ext":"macd_signal_ext",
        "macdhistext":"macd_hist_ext","macd_hist_ext":"macd_hist_ext",
        "macdfix":"macd_fix","macd_fix":"macd_fix",
        "macdsignalfix":"macd_signal_fix","macd_signal_fix":"macd_signal_fix",
        "macdhistfix":"macd_hist_fix","macd_hist_fix":"macd_hist_fix",
        # Aroon
        "aroon_1":"aroon_up","aroon_up":"aroon_up",
        "aroon_2":"aroon_down","aroon_down":"aroon_down",
        "aroonosc":"aroonosc",
        # Stochastic + StochRSI
        "slowk":"stoch_slowk","stoch_slowk":"stoch_slowk",
        "slowd":"stoch_slowd","stoch_slowd":"stoch_slowd",
        "fastk":"stoch_fastk","stoch_fastk":"stoch_fastk",
        "fastd":"stoch_fastd","stoch_fastd":"stoch_fastd",
        "fastk_rsi":"stochrsi_fastk","stochrsi_fastk":"stochrsi_fastk",
        "fastd_rsi":"stochrsi_fastd","stochrsi_fastd":"stochrsi_fastd",
        # Hilbert
        "ht_trendmod":"ht_trendmode","ht_trendmode":"ht_trendmode",
        # Other TA keep-as-is canonical (normalized later)
        # Macros & customs (normalized later)
    }
    def _normalize_name(self, s: str) -> str:
        key = str(s).strip().replace(" ", "_").lower()
        return self._NORM_EQUIVS.get(key, key)

    def _build_norm_map(self, df: pd.DataFrame = None) -> dict:
        """Map normalized-name -> actual column name present in df."""
        df = self.transformed_df if df is None else df
        m = {}
        for c in df.columns:
            m[self._normalize_name(c)] = c
        return m

    def _present_any(self, names: List[str]) -> List[str]:
        """Return actual df columns matching names by normalized equivalence."""
        m = self._build_norm_map()
        out = []
        for n in names:
            k = self._normalize_name(n)
            if k in m:
                out.append(m[k])
        # Deduplicate preserving order
        seen, deduped = set(), []
        for c in out:
            if c not in seen:
                deduped.append(c); seen.add(c)
        return deduped

    # ---------- Init ----------
    def __init__(self, transformed):
        self.transformed_df = transformed.transformed_df.copy(deep=True)

        # Ensure canonical types
        if "Date" in self.transformed_df.columns:
            self.transformed_df["Date"] = pd.to_datetime(self.transformed_df["Date"]).dt.tz_localize(None)
        elif "date" in self.transformed_df.columns:
            self.transformed_df["Date"] = pd.to_datetime(self.transformed_df["date"]).dt.tz_localize(None)

        if "Ticker" in self.transformed_df.columns:
            self.transformed_df["Ticker"] = self.transformed_df["Ticker"].astype(str).str.upper()
        elif "ticker" in self.transformed_df.columns:
            self.transformed_df["Ticker"] = self.transformed_df["ticker"].astype(str).str.upper()

        if "Volume" in self.transformed_df.columns:
            self.transformed_df["ln_volume"] = self.transformed_df["Volume"].apply(
                lambda x: np.log(x) if pd.notnull(x) and x > 0 else np.nan
            )

        # Derive Month/Weekday if missing
        if "Month_x" not in self.transformed_df.columns and "Month" not in self.transformed_df.columns and "month" not in self.transformed_df.columns:
            if "Date" in self.transformed_df.columns:
                self.transformed_df["month"] = self.transformed_df["Date"].dt.month_name()
        if "Weekday" not in self.transformed_df.columns and "weekday" not in self.transformed_df.columns:
            if "Date" in self.transformed_df.columns:
                self.transformed_df["weekday"] = self.transformed_df["Date"].dt.day_name()

        self.model = None
        self.config = TrainConfig()

    # ---------- Feature sets ----------
    # def _define_feature_sets(self):
    #     # Growth (exclude any 'future')
    #     self.GROWTH = [g for g in self.transformed_df.columns
    #                    if g.lower().startswith("growth_") and "future" not in g.lower()]

    #     # OHLCV
    #     self.OHLCV = self._present_any(["Open","High","Low","Close","Volume"] +
    #                                    ["open","high","low","close","volume"])

    #     # Categoricals (include lowercase variants from your pipeline)
    #     self.CATEGORICAL = self._present_any([
    #         "Month_x","Month","month","Weekday","weekday","Ticker","ticker","ticker_type","year","wom","month_wom"
    #     ])

    #     # Targets
    #     self.TO_PREDICT = [c for c in self.transformed_df.columns
    #                        if re.search(r"is_positive.*future|future.*is_positive|is_positive_future", c, re.I)]

    #     # Macros (use your pipeline's names; safe if absent)
    #     macro_candidates = [
    #         "gdppot_yoy","gdppot_qoq",
    #         "cpilfesl_yoy","cpilfesl_qoq","cpi_core_yoy","cpi_core_mom",
    #         "fedfunds_yoy","fedfunds_qoq","FEDFUNDS",
    #         "dgs1_yoy","dgs1_qoq","DGS1",
    #         "dgs5_yoy","dgs5_qoq","DGS5",
    #         "dgs10_yoy","dgs10_qoq","DGS10",
    #         "vix_adj_close","btc_usd_adj_close","gdaxi_adj_close",
    #         # optionally keep some growth_* macro proxies if you want
    #     ]
    #     self.MACRO = self._present_any(macro_candidates)

    #     # Custom numericals
    #     self.CUSTOM_NUMERICAL = self._present_any([
    #         "SMA10","sma10","SMA20","sma20","SMA50","sma50","SMA200","sma200",
    #         "growing_moving_average","high_minus_low_relative","volatility","ln_volume","sharpe"
    #     ])

    #     # TA indicators (TA-Lib + underscore variants + stoch/stochrsi + hilbert)
    #     ta_base = [
    #         "adx","adxr","apo","bop","cci","cmo","dx","mfi","minus_di","plus_di","mom","ppo",
    #         "roc","rocp","rocr","rocr100","rsi",
    #         "ad","adosc","obv","atr","natr",
    #         "avgprice","medprice","typprice","wclprice",
    #         "ht_dcperiod","ht_dcphase","ht_phasor_inphase","ht_phasor_quadrature",
    #         "ht_sine_sine","ht_sine_leadsine","ht_trendmode", "trix","ultosc","willr"   
    #     ]
    #     macd_variants  = ["macd","macd_signal","macd_hist","macd_ext","macd_signal_ext","macd_hist_ext",
    #                       "macd_fix","macd_signal_fix","macd_hist_fix"]
    #     aroon_variants = ["aroon_up","aroon_down","aroonosc","aroon_1","aroon_2"]
    #     stoch_variants = ["stoch_slowk","stoch_slowd","stoch_fastk","stoch_fastd",
    #                       "stochrsi_fastk","stochrsi_fastd"]

    #     canonical_ta = ta_base + macd_variants + aroon_variants + stoch_variants
    #     self.TECHNICAL_INDICATORS = self._present_any(canonical_ta)

    #     # Candlestick patterns
    #     self.TECHNICAL_PATTERNS = [c for c in self.transformed_df.columns if c.lower().startswith("cdl")]

    #     # Numerical features
    #     self.NUMERICAL = self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + self.CUSTOM_NUMERICAL + self.MACRO

    #     # Drop list
    #     drop_candidates = ["Year","year","Date","date","Month_x","Month_y","Month","month","index","Quarter","index_x","index_y"]
    #     self.TO_DROP = self._present_any(drop_candidates) + self.CATEGORICAL + self.OHLCV

    #     # Debug bucket
    #     self.OTHER = [k for k in self.transformed_df.columns
    #                   if k not in set(self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT)]

    # def _define_dummies(self):
    #     # Ensure string types for categoricals
    #     for c in self.CATEGORICAL:
    #         self.transformed_df[c] = self.transformed_df[c].astype(str)

    #     if len(self.CATEGORICAL) > 0:
    #         dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype="int32")
    #         self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
    #         self.DUMMIES = list(dummy_variables.columns)
    #     else:
    #         self.df_full = self.transformed_df.copy(deep=True)
    #         self.DUMMIES = []

    def _define_feature_sets(self):
    # Helper: only keep columns that exist
        def present(cols):
            return [c for c in cols if c in self.transformed_df.columns]

        # 1) Core groups
        self.GROWTH = [c for c in self.transformed_df.columns
                    if c.startswith("growth_") and "future" not in c]

        # Include adj_close in OHLCV if you want to explicitly drop it later
        self.OHLCV = present(["Open","High","Low","Close","Volume","adj_close"])

        # Categorical that actually appear in your pipeline
        cat_candidates = ["Month_x","Month","month","Weekday","weekday","Ticker","ticker","ticker_type"]
        self.CATEGORICAL = present(cat_candidates)

        # Targets: prefer is_positive*future; fall back to any *future*
        self.TO_PREDICT = [c for c in self.transformed_df.columns
                        if ("future" in c and ("is_positive" in c or "is_positive_future" in c))]
        if not self.TO_PREDICT:
            self.TO_PREDICT = [c for c in self.transformed_df.columns if "future" in c]

        # Macro (matches your merge logs)
        macro_rates = present([
            "gdppot_yoy","gdppot_qoq",
            "cpilfesl_yoy","cpilfesl_qoq",
            "fedfunds_yoy","fedfunds_qoq",
            "dgs1_yoy","dgs1_qoq",
            "dgs5_yoy","dgs5_qoq",
            "dgs10_yoy","dgs10_qoq",
        ])
        macro_growth_regex = re.compile(r"^growth_(btc|vix|dax|snp500|dji|epi|gold|brent_oil|crude_oil)_(1d|3d|7d|30d|90d|252d|365d)$")
        macro_growth = [c for c in self.transformed_df.columns if macro_growth_regex.match(c)]
        self.MACRO = macro_rates + macro_growth

        # Custom numerics that exist in your dataset
        self.CUSTOM_NUMERICAL = present([
            "sma10","sma20","sma50","sma200",
            "growing_moving_average","volatility","sharpe",
            "high_minus_low_relative","ln_volume"
        ])

        # 2) Technical indicators, with common synonym spellings included
        ta_candidates = [
            # trend/momentum/vol
            "adx","adxr","apo","bop","cci","cmo","dx","mfi","mom","ppo","roc","rocp","rocr","rocr100","rsi",
            "slowk","slowd","fastk","fastd","fastk_rsi","fastd_rsi",
            "trix","ultosc","willr","obv","atr","natr","ad","adosc",
            # macd spellings
            "macd","macdsignal","macdhist","macd_signal","macd_hist",
            "macd_ext","macdsignal_ext","macdhist_ext",
            "macd_fix","macdsignal_fix","macdhist_fix",
            # aroon spellings
            "aroon_1","aroon_2","aroonosc","aroon_up","aroon_down",
            # HT / price types
            "ht_dcperiod","ht_dcphase","ht_phasor_inphase","ht_phasor_quadrature",
            "ht_sine_sine","ht_sine_leadsine","ht_trendmod",
            "avgprice","medprice","typprice","wclprice",
            # occasionally present extras
            "plus_di","minus_di","plus_dm","trange"
        ]
        self.TECHNICAL_INDICATORS = present(ta_candidates)

        # Candle patterns
        self.TECHNICAL_PATTERNS = [c for c in self.transformed_df.columns if c.startswith("cdl")]

        # 3) Combine numeric features actually used for modeling
        self.NUMERICAL = list(dict.fromkeys(
            self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS + self.CUSTOM_NUMERICAL + self.MACRO
        ))

        # 4) Columns to drop from X (we’ll still keep them in df_full)
        drop_artifacts = ["Year","year","Date","Month_x","Month_y","index","Quarter","index_x","index_y","split"]
        self.TO_DROP = present(drop_artifacts) + self.CATEGORICAL + self.OHLCV

        # 5) For inspection/debugging
        used = set(self.OHLCV + self.CATEGORICAL + self.NUMERICAL + self.TO_DROP + self.TO_PREDICT)
        self.OTHER = [k for k in self.transformed_df.columns if k not in used]


    def _define_dummies(self):
        # Make sure categorical columns are string-typed if they exist
        for c in ["Month_x","Month","month","Weekday","weekday","Ticker","ticker","ticker_type"]:
            if c in self.CATEGORICAL:
                self.transformed_df[c] = self.transformed_df[c].astype(str)

        if self.CATEGORICAL:
            dummy_variables = pd.get_dummies(self.transformed_df[self.CATEGORICAL], dtype="int32")
            self.df_full = pd.concat([self.transformed_df, dummy_variables], axis=1)
            self.DUMMIES = list(dummy_variables.columns)
        else:
            self.df_full = self.transformed_df.copy(deep=True)
            self.DUMMIES = []


    def _perform_temporal_split(self, df: pd.DataFrame, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
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
        return df

    def _choose_target(self) -> str:
        for t in self.config.target_preference:
            if t in self.TO_PREDICT:
                return t
        alt = [c for c in self.TO_PREDICT if c.lower().startswith("is_positive")]
        if alt:
            return alt[0]
        if self.TO_PREDICT:
            return self.TO_PREDICT[0]
        raise ValueError("No suitable target column found. Ensure a column like 'is_positive_growth_30d_future' exists.")

    # ---------- Public pipeline ----------
    # def prepare_dataframe(self):
    #     self._define_feature_sets()
    #     self._define_dummies()

    #     if "Date" not in self.df_full.columns:
    #         raise ValueError("Missing 'Date' column in dataframe.")
    #     min_date_df = pd.to_datetime(self.df_full["Date"]).min()
    #     max_date_df = pd.to_datetime(self.df_full["Date"]).max()
    #     self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

    #     self._define_dataframes_for_ML()
    


    def prepare_dataframe(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self._define_feature_sets()
        self._define_dummies()

        if "Date" not in self.df_full.columns:
            raise ValueError("Missing 'Date' column in dataframe.")

        # normalize dates
        self.df_full["Date"] = pd.to_datetime(self.df_full["Date"]).dt.tz_localize(None)

        # optional range filter
        if start_date is not None:
            start = pd.Timestamp(start_date).tz_localize(None)
            self.df_full = self.df_full[self.df_full["Date"] >= start].copy()

        if end_date is not None:
            end = pd.Timestamp(end_date).tz_localize(None)
            self.df_full = self.df_full[self.df_full["Date"] <= end].copy()

        # temporal split on the filtered window
        min_date_df = self.df_full["Date"].min()
        max_date_df = self.df_full["Date"].max()
        self._perform_temporal_split(self.df_full, min_date=min_date_df, max_date=max_date_df)

        self._define_dataframes_for_ML()


    # def _define_dataframes_for_ML(self):
    #     features_list = list(dict.fromkeys(self.NUMERICAL + self.DUMMIES))
    #     # HARD FILTER to prevent any future leakage
    #     features_list = [c for c in features_list if "future" not in c.lower()]

    #     target = self._choose_target()
    #     self.target_col = target

    #     def build_X(df):
    #         cols = [c for c in features_list if c in df.columns] + ([target] if target in df.columns else [])
    #         return df[cols].copy(deep=True)

    #     self.train_df = self.df_full[self.df_full["split"] == "train"].copy(deep=True)
    #     self.valid_df = self.df_full[self.df_full["split"] == "validation"].copy(deep=True)
    #     self.train_valid_df = self.df_full[self.df_full["split"].isin(["train","validation"])].copy(deep=True)
    #     self.test_df = self.df_full[self.df_full["split"] == "test"].copy(deep=True)

    #     self.X_train = self._clean_dataframe_from_inf_and_nan(build_X(self.train_df))
    #     self.X_valid = self._clean_dataframe_from_inf_and_nan(build_X(self.valid_df))
    #     self.X_train_valid = self._clean_dataframe_from_inf_and_nan(build_X(self.train_valid_df))
    #     self.X_test = self._clean_dataframe_from_inf_and_nan(build_X(self.test_df))
    #     self.X_all = self._clean_dataframe_from_inf_and_nan(build_X(self.df_full))

    #     self.y_train = self.X_train[target].copy()
    #     self.y_valid = self.X_valid[target].copy()
    #     self.y_train_valid = self.X_train_valid[target].copy()
    #     self.y_test = self.X_test[target].copy()
    #     self.y_all = self.X_all[target].copy()

    #     for X in [self.X_train, self.X_valid, self.X_train_valid, self.X_test, self.X_all]:
    #         if target in X.columns:
    #             X.drop(columns=[target], inplace=True)

    #     print(f"length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}")
    #     print(f"  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}")
    #     print(f"Target used: {self.target_col}")

    def _define_dataframes_for_ML(self):
        # ⬅️ include dummies here
        features_list = list(dict.fromkeys(self.NUMERICAL + getattr(self, "DUMMIES", [])))

        target = self._choose_target()
        self.target_col = target

        self.train_df = self.df_full[self.df_full["split"] == "train"].copy(deep=True)
        self.valid_df = self.df_full[self.df_full["split"] == "validation"].copy(deep=True)
        self.train_valid_df = self.df_full[self.df_full["split"].isin(["train","validation"])].copy(deep=True)
        self.test_df  = self.df_full[self.df_full["split"] == "test"].copy(deep=True)

        def build_X(df):
            cols = [c for c in features_list if c in df.columns] + ([target] if target in df.columns else [])
            return df[cols].copy(deep=True)

        self.X_train        = self._clean_dataframe_from_inf_and_nan(build_X(self.train_df))
        self.X_valid        = self._clean_dataframe_from_inf_and_nan(build_X(self.valid_df))
        self.X_train_valid  = self._clean_dataframe_from_inf_and_nan(build_X(self.train_valid_df))
        self.X_test         = self._clean_dataframe_from_inf_and_nan(build_X(self.test_df))
        self.X_all          = self._clean_dataframe_from_inf_and_nan(build_X(self.df_full))

        # split y / drop target
        for X in [self.X_train, self.X_valid, self.X_train_valid, self.X_test, self.X_all]:
            if target in X.columns:
                setattr(self, f"y_{'all' if X is self.X_all else 'train' if X is self.X_train else 'valid' if X is self.X_valid else 'train_valid' if X is self.X_train_valid else 'test'}", X[target].copy())
                X.drop(columns=[target], inplace=True)

        # helpful debug
        dummies_cnt = len(getattr(self, "DUMMIES", []))
        unused_non_future = [c for c in self.df_full.columns if c not in set(features_list + [target]) and "future" not in c]
        print(f"Used: {len(features_list)} | Dropped: {len(self.TO_DROP)} | Future: {len(self.TO_PREDICT)} | Unused(non-future): {len(unused_non_future)}")
        print(f"Dummy columns: {dummies_cnt} (sample: {getattr(self, 'DUMMIES', [])[:5]})")

        print(f"length: X_train {self.X_train.shape},  X_validation {self.X_valid.shape}, X_test {self.X_test.shape}")
        print(f"  X_train_valid = {self.X_train_valid.shape},  all combined: X_all {self.X_all.shape}")
        print(f"Target used: {self.target_col}")


    @staticmethod
    def _clean_dataframe_from_inf_and_nan(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.fillna(0)

    # ---------- Model ----------
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

    # ---------- Persistence ----------
    def persist(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        model_path = os.path.join(data_dir, "random_forest_model.joblib")
        meta_path = os.path.join(data_dir, "rf_meta.json")

        joblib.dump(self.model, model_path)

        feature_columns = self.X_train_valid.columns.tolist()
        feature_aliases = [self._normalize_name(c) for c in feature_columns]

        pd.Series({
            "target_col": self.target_col,
            "feature_columns": feature_columns,
            "feature_aliases": feature_aliases
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
        self._inference_feature_aliases = list(meta.get("feature_aliases", [self._normalize_name(c) for c in self._inference_feature_columns]))

    # ---------- Inference ----------
    def make_inference(self, pred_name: str = "rf_pred"):
        if self.model is None:
            raise ValueError("Model is not loaded/trained. Call train_random_forest() or load() first.")

        feat_cols_saved = getattr(self, "_inference_feature_columns", self.X_train_valid.columns.tolist())
        feat_aliases_saved = getattr(self, "_inference_feature_aliases",
                                     [self._normalize_name(c) for c in feat_cols_saved])

        # Choose df_full if available, else transformed_df
        base_df = getattr(self, "df_full", self.transformed_df)
        norm_map = self._build_norm_map(base_df)

        # X = pd.DataFrame(index=base_df.index)
        # for actual_saved, alias_saved in zip(feat_cols_saved, feat_aliases_saved):
        #     src = norm_map.get(alias_saved)
        #     if src is None:
        #         X[actual_saved] = 0  # unseen feature → fill 0
        #     else:
        #         X[actual_saved] = base_df[src]

        cols = []
        for actual_saved, alias_saved in zip(feat_cols_saved, feat_aliases_saved):
            src = norm_map.get(alias_saved)
            cols.append(base_df[src].to_numpy() if src is not None else np.zeros(len(base_df)))
        X = pd.DataFrame(np.column_stack(cols), index=base_df.index, columns=feat_cols_saved)
        
        X = self._clean_dataframe_from_inf_and_nan(X)


        y_pred_all = self.model.predict_proba(X)
        class1 = np.array([p[1] for p in y_pred_all])

        base_df[pred_name] = class1

        # Ensure Date/Ticker exist (map from aliases if needed)
        nm = self._build_norm_map(base_df)
        if "date" in nm and "Date" not in base_df.columns:
            base_df["Date"] = pd.to_datetime(base_df[nm["date"]]).dt.tz_localize(None)
        if "ticker" in nm and "Ticker" not in base_df.columns:
            base_df["Ticker"] = base_df[nm["ticker"]].astype(str).str.upper()

        if "Date" in base_df.columns:
            base_df[f"{pred_name}_rank"] = base_df.groupby("Date")[pred_name].rank(method="first", ascending=False)
        else:
            base_df[f"{pred_name}_rank"] = base_df[pred_name].rank(method="first", ascending=False)

        keep = [pred_name, f"{pred_name}_rank"]
        if {"Date","Ticker"}.issubset(base_df.columns):
            keep += ["Date","Ticker"]
        return base_df[keep]

    # ---------- Evaluation ----------
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

    # def summarize_performance(self, threshold: float = 0.5, splits: tuple = ("validation","test")) -> pd.DataFrame:
    #     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
    #     rows = []
    #     for s in splits:
    #         y_true, proba = self._get_split_arrays(s)
    #         y_pred = (proba >= threshold).astype(int)
    #         cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    #         if cm.shape == (2,2):
    #             tn, fp, fn, tp = cm.ravel()
    #         else:
    #             tn = cm[0,0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
    #             fp = cm[0,1] if cm.shape[1] > 1 else 0
    #             fn = cm[1,0] if cm.shape[0] > 1 else 0
    #             tp = cm[1,1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    #         # --- add these lines ---
    #         calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    #         sk_prec = precision_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan
    #         # warn if mismatch (helps catch metric/threshold mixups)
    #         if not np.isnan(sk_prec) and abs(sk_prec - calc_prec) > 1e-12:
    #             print(f"[WARN] precision mismatch on '{s}': sklearn={sk_prec:.6f} vs tp/(tp+fp)={calc_prec:.6f}")

    #         row = {
    #             "split": s,
    #             "n_samples": int(len(y_true)),
    #             "pos_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
    #             "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
    #             "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
    #             "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
    #             "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
    #             "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan,
    #             "pr_auc": float(average_precision_score(y_true, proba)) if len(y_true) else np.nan,
    #             "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    #             "threshold": threshold,
    #         }
    #         rows.append(row)
    #     return pd.DataFrame(rows, columns=["split","n_samples","pos_rate","accuracy","precision","recall","f1","roc_auc","pr_auc","tn","fp","fn","tp","threshold"])

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

            # Sanity-check precision against TP/(TP+FP)
            calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sk_prec = precision_score(y_true, y_pred, zero_division=0)
            if abs(sk_prec - calc_prec) > 1e-12:
                print(f"[WARN] precision mismatch on '{s}': sklearn={sk_prec:.6f} vs tp/(tp+fp)={calc_prec:.6f}")

            # Core metrics
            acc = accuracy_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1  = f1_score(y_true, y_pred, zero_division=0)

            # Extras
            pred_pos_rate = float(y_pred.mean())
            specificity   = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            bal_acc       = (rec + specificity) / 2 if not (np.isnan(specificity)) else float("nan")
            roc_auc       = float(roc_auc_score(y_true, proba)) if np.unique(y_true).size > 1 else float("nan")
            pr_auc        = float(average_precision_score(y_true, proba)) if y_true.sum() > 0 else float("nan")
            brier         = float(brier_score_loss(y_true, proba))

            row = {
                "split": s,
                "n_samples": int(len(y_true)),
                "pos_rate": float(np.mean(y_true)),
                "pred_pos_rate": pred_pos_rate,
                "accuracy": float(acc),
                "precision": float(sk_prec),
                "recall": float(rec),
                "specificity": float(specificity),
                "balanced_accuracy": float(bal_acc),
                "f1": float(f1),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "brier": brier,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "threshold": threshold,
                "precision_tp_fp": float(calc_prec),  # manual check value
            }
            rows.append(row)

        cols = ["split","n_samples","pos_rate","pred_pos_rate","accuracy","precision",
                "recall","specificity","balanced_accuracy","f1","roc_auc","pr_auc","brier",
                "tn","fp","fn","tp","threshold","precision_tp_fp"]
        return pd.DataFrame(rows, columns=cols)


    def text_classification_report(self, split: str = "test", threshold: float = 0.5) -> str:
        from sklearn.metrics import classification_report
        y_true, proba = self._get_split_arrays(split)
        y_pred = (proba >= threshold).astype(int)
        return classification_report(y_true, y_pred, digits=4, zero_division=0)

    def daily_topk_stats(self, k: int = 5, split: str = "test", pred_col: str = "rf_prob_30d") -> dict:
        if "split" not in self.df_full.columns:
            raise ValueError("df_full is missing 'split' column; call prepare_dataframe() first.")
        if pred_col not in self.df_full.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found. Run make_inference('{pred_col}') first or pass the correct name.")
        if self.target_col not in self.df_full.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in df_full.")

        d = self.df_full[self.df_full["split"] == split].copy()
        if "Date" not in d.columns:
            raise ValueError("df_full is missing 'Date' column needed for daily Top-K stats.")
        d = d.sort_values(["Date", pred_col], ascending=[True, False])

        base_daily = d.groupby("Date")[self.target_col].mean().mean()
        topk = d.groupby("Date").head(k)
        top_daily = topk.groupby("Date")[self.target_col].mean().mean()

        lift = None
        gcol = "growth_future_30d"
        if gcol in d.columns:
            lift = (topk.groupby("Date")[gcol].mean().mean()
                    - d.groupby("Date")[gcol].mean().mean())

        return {"split": split, "k": k, "daily_hitrate_baseline": float(base_daily), "daily_hitrate_topk": float(top_daily), "avg_growth_lift_topk_vs_all": (None if lift is None else float(lift))}

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

    def refit_on_train_valid(self, class_weight: Optional[str] = None):
        if self.model is None:
            raise ValueError("Call train_random_forest once to set hyperparameters.")
        params = self.model.get_params()
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**{**params, "class_weight": class_weight})
        self.model.fit(self.X_train_valid, self.y_train_valid)
    
    def debug_feature_coverage(self):
        all_cols = set(self.df_full.columns)
        used = set(self.X_train_valid.columns)
        dropped = set(self.TO_DROP)
        futurey = {c for c in all_cols if "future" in c.lower()}
        not_used = sorted(all_cols - used - dropped - futurey - {self.target_col})
        print(f"Used: {len(used)} | Dropped: {len(dropped)} | Future: {len(futurey)} | Unused(non-future): {len(not_used)}")
        print("Unused sample:", not_used[:30])

