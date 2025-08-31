import logging
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_history_bulk(
    tickers: List[str],
    pause: float = 1.0
) -> pd.DataFrame:
    """
    Fetches max-history daily data for multiple tickers in one call.
    """
    # This returns a nested DataFrame: columns=(ticker, field)
    raw = yf.download(
        tickers,
        period="max",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=True,
    )
    # Unstack into long form
    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.levels[0]:
            logger.warning("No data for %s", ticker)
            continue

        df = raw[ticker].copy()
        df = df.rename_axis("Date").reset_index()
        df.loc[:, "Ticker"] = ticker
        frames.append(df)

        time.sleep(pause)  # to be gentle on the API

    return pd.concat(frames, ignore_index=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, "Date"] = pd.to_datetime(df["Date"])
    df.loc[:, "Year"] = df["Date"].dt.year
    df.loc[:, "Month"] = df["Date"].dt.month
    df.loc[:, "Weekday"] = df["Date"].dt.weekday
    df.loc[:, "wom"] = ((df["Date"].dt.day - 1) // 7 + 1).astype(int)
    df.loc[:, "month_wom"] = (
        df["Date"].dt.month_name() + "_w" + df["wom"].astype(str)
    )
    return df


# def add_growth_features(
#     df: pd.DataFrame,
#     lookbacks: List[int] = [1, 3, 7, 30, 90, 252, 365],
#     horizons: List[int] = [30],
#     binarize_thresholds: Optional[Dict[int, float]] = None
# ) -> pd.DataFrame:
#     df = df.copy()
#     # 1) historical growth
#     for j in lookbacks:
#         df[f"growth_{j}d"] = df["Close"] / df["Close"].shift(j)

#     # 2) forward-looking growth
#     for h in horizons:
#         df[f"growth_future_{h}d"] = df["Close"].shift(-h) / df["Close"]

#     # 3) optional binarization - CREATE BOTH NAMING CONVENTIONS
#     if binarize_thresholds:
#         for h, thresh in binarize_thresholds.items():
#             if h not in horizons:
#                 raise ValueError(f"horizon {h} not in `horizons` list")
#             col = f"growth_future_{h}d"
#             # Your pipeline naming
#             df[f"is_positive_future_{h}d"] = (df[col] > thresh).astype(int)
#             # Training code expected naming
#             df[f"is_positive_growth_{h}d_future"] = (df[col] > thresh).astype(int)

#     return df


# In extract.py - clean up the target variable creation

def add_growth_features(
    df: pd.DataFrame,
    lookbacks: List[int] = [1, 3, 7, 30, 90, 252, 365],
    horizons: List[int] = [30],
    binarize_thresholds: Optional[Dict[int, float]] = None
) -> pd.DataFrame:
    """
    Add growth features with clean target variable naming
    """
    df = df.copy()
    
    # 1) Historical growth (backward-looking)
    for j in lookbacks:
        df[f"growth_{j}d"] = df["Close"] / df["Close"].shift(j)

    # 2) Forward-looking growth (continuous target)
    for h in horizons:
        df[f"growth_future_{h}d"] = df["Close"].shift(-h) / df["Close"]

    # 3) Binary targets (for classification)
    if binarize_thresholds:
        for h, thresh in binarize_thresholds.items():
            if h not in horizons:
                raise ValueError(f"horizon {h} not in `horizons` list")
            
            continuous_col = f"growth_future_{h}d"
            
            # Create ONE consistent binary target name
            binary_col = f"is_positive_growth_{h}d_future"
            df[binary_col] = (df[continuous_col] > thresh).astype(int)
            
            print(f"Created binary target: {binary_col} (threshold={thresh})")

    return df


def add_moving_averages(
    df: pd.DataFrame,
    windows: List[int] = [10, 20]
) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(w).mean()
    if len(windows) >= 2:
        df.loc[:, "growing_moving_average"] = (
            df[f"SMA{windows[0]}"] > df[f"SMA{windows[1]}"]
        ).astype(int)
    return df


def add_volatility_and_sharpe(
    df: pd.DataFrame,
    vol_window: int = 30,
    risk_free: float = 0.045
) -> pd.DataFrame:
    df = df.copy()
    # Fix deprecated fill_method parameter
    daily_returns = df["Close"].pct_change()
    df.loc[:, "volatility"] = (
        daily_returns.rolling(vol_window).std() * np.sqrt(252)
    )
    if "growth_252d" not in df.columns:
        df.loc[:, "growth_252d"] = df["Close"] / df["Close"].shift(252)
    df.loc[:, "Sharpe"] = (df["growth_252d"] - risk_free) / df["volatility"]
    return df


def add_price_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, "high_minus_low_relative"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )
    return df

def add_additional_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing technical features that training code expects"""
    df = df.copy()
    
    # Add simple moving averages if not present
    if 'sma10' not in df.columns:
        df['sma10'] = df['close'].rolling(window=10).mean()
    if 'sma20' not in df.columns:
        df['sma20'] = df['close'].rolling(window=20).mean()
        
    # Moving average crossover signal
    if 'growing_moving_average' not in df.columns:
        df['growing_moving_average'] = (df['sma10'] > df['sma20']).astype(int)
    
    # Price range relative to close
    if 'high_minus_low_relative' not in df.columns:
        df['high_minus_low_relative'] = (df['high'] - df['low']) / df['close']
    
    return df

def build_stock_dataframe(
    tickers: List[str],
    lookbacks: List[int] = [1, 3, 7, 30, 90, 252, 365],
    horizons: List[int] = [30],
    binarize_thresholds: Optional[Dict[int, float]] = None,
    ma_windows: List[int] = [10, 20],
    vol_window: int = 30,
    risk_free: float = 0.045,
    pause: float = 1.0
) -> pd.DataFrame:
    """
    Fetches history for `tickers`, applies all feature pipelines,
    and returns a cleaned, sorted DataFrame.
    """
    df_all = fetch_history_bulk(tickers, pause=pause)

    df_all = (
        df_all
        .pipe(add_time_features)
        .pipe(add_growth_features, lookbacks, horizons, binarize_thresholds)
        .pipe(add_moving_averages, ma_windows)
        .pipe(add_volatility_and_sharpe, vol_window, risk_free)
        .pipe(add_price_range)
        #.pipe(add_additional_technical_features)  # ADD THIS LINE

    )

    # ADD MISSING FEATURES FOR TRAINING COMPATIBILITY
    df_all['ln_volume'] = np.log(df_all['Volume']).fillna(0)
    df_all['ticker_type'] = 'US'  # Simple classification - can be enhanced later

    # Only drop nulls from historical data, preserve recent data
    future_cols = [f"growth_future_{h}d" for h in horizons]
    if future_cols:
        max_horizon = max(horizons)
        cutoff_date = df_all['Date'].max() - pd.Timedelta(days=max_horizon + 5)
        
        # Keep recent data even with null future columns
        recent_mask = df_all['Date'] > cutoff_date
        df_all = pd.concat([
            df_all[~recent_mask].dropna(subset=future_cols),  # Clean historical data
            df_all[recent_mask]  # Preserve recent data
        ], ignore_index=True)

    # Final ordering & reset
    df_all.sort_values(["Ticker", "Date"], inplace=True)
    df_all.reset_index(drop=True, inplace=True)

    # Normalize column names at the very end
    df_all.columns = (
        df_all.columns
        .str.lower()
        .str.replace(" ", "_")
    )
    
    return df_all