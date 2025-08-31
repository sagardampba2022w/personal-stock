import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
import numpy as np

def fetch_and_build_macros(
    start: str,
    lookbacks: list[int] = [1, 3, 7, 30, 90, 365]
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # 1) yfinance-based macros (daily data)
    yf_map = {
        "btc":      "BTC-USD",
        "vix":      "^VIX", 
        "dax":      "^GDAXI",
        "snp500":   "^GSPC",
        "dji":      "^DJI",
        "epi":      "EPI",
        "gold":     "GC=F",
        "brent_oil":"BZ=F", 
        "crude_oil":"CL=F",
    }
    
    for key, sym in yf_map.items():
        try:
            df = yf.Ticker(sym).history(period="max", interval="1d")[["Close"]].copy()
            df.index = df.index.tz_localize(None).normalize()
            
            # Calculate growth rates
            for i in lookbacks:
                df[f"growth_{key}_{i}d"] = df["Close"] / df["Close"].shift(i) - 1
            
            df = df.ffill().drop(columns=["Close"])
            out[key] = df
            print(f"Successfully fetched {key}: {len(df)} rows")
        except Exception as e:
            print(f"Failed to fetch {key}: {e}")

    # 2) FRED-based macros - CORRECTED APPROACH
    fred_map = {
        "gdppot":   ("GDPPOT", "Q"),    
        "cpilfesl": ("CPILFESL", "M"),  
        "fedfunds": ("FEDFUNDS", "M"),  
        "dgs1":     ("DGS1", "D"),      
        "dgs5":     ("DGS5", "D"),      
        "dgs10":    ("DGS10", "D"),     
    }
    
    for key, (series, freq) in fred_map.items():
        try:
            df = pdr.DataReader(series, "fred", start=start).copy()
            df.rename(columns={series: key}, inplace=True)
            df.index = pd.to_datetime(df.index).normalize()
            
            # STEP 1: Calculate growth rates at original frequency BEFORE resampling
            if freq == "Q":
                df[f"{key}_yoy"] = df[key] / df[key].shift(4) - 1  # 4 quarters
                df[f"{key}_qoq"] = df[key] / df[key].shift(1) - 1  # 1 quarter
            elif freq == "M": 
                df[f"{key}_yoy"] = df[key] / df[key].shift(12) - 1  # 12 months
                df[f"{key}_qoq"] = df[key] / df[key].shift(3) - 1   # 3 months
            elif freq == "D":
                # For daily data, use business day approximations
                df[f"{key}_yoy"] = df[key] / df[key].shift(252) - 1  # ~252 business days
                df[f"{key}_qoq"] = df[key] / df[key].shift(63) - 1   # ~63 business days
            
            # STEP 2: Resample to daily frequency 
            if freq != "D":
                # For non-daily data, resample and forward-fill
                df = df.resample("D").ffill()
            else:
                # For daily data, just forward-fill missing values
                df = df.ffill()
                
            out[key] = df
            print(f"Successfully processed {key}: {len(df)} rows")
            
            # Validation check
            yoy_col = f"{key}_yoy"
            if yoy_col in df.columns:
                recent_yoy = df[yoy_col].dropna().tail(10)
                if len(recent_yoy) > 0:
                    yoy_range = (recent_yoy.min(), recent_yoy.max())
                    print(f"  Recent YoY range: [{yoy_range[0]:.4f}, {yoy_range[1]:.4f}]")
                    
        except Exception as e:
            print(f"Failed to fetch {key}: {e}")

    return out

def merge_all_macros(
    stocks_df: pd.DataFrame,
    macro_frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Merge macro data with stock data"""
    merged = stocks_df.copy()
    merged['Date'] = pd.to_datetime(merged['Date']).dt.normalize()
    
    print(f"Starting merge with {len(merged)} stock rows...")
    
    for key, df in macro_frames.items():
        # Select relevant columns (growth rates, YoY, QoQ)
        cols = [c for c in df.columns if (
            c.startswith(f"growth_{key}_") or 
            c.endswith(("_yoy", "_qoq"))
        )]
        
        if cols:
            print(f"Merging {key} with {len(cols)} columns: {cols}")
            
            before_merge = len(merged)
            merged = merged.merge(
                df[cols],
                how="left", 
                left_on="Date",
                right_index=True,
                validate="many_to_one"
            )
            after_merge = len(merged)
            
            if before_merge != after_merge:
                print(f"  Warning: Row count changed: {before_merge} -> {after_merge}")
            else:
                print(f"  Merge successful: {after_merge} rows")
    
    # Forward fill macro columns to handle weekends/holidays
    macro_cols = [c for c in merged.columns if (
        c.startswith("growth_") or c.endswith(("_yoy", "_qoq"))
    )]
    
    print(f"Forward filling {len(macro_cols)} macro columns...")
    for col in macro_cols:
        null_before = merged[col].isnull().sum() 
        merged[col] = merged[col].ffill()
        null_after = merged[col].isnull().sum()
        if null_before != null_after:
            print(f"  {col}: {null_before} -> {null_after} nulls")
    
    return merged

def validate_macro_data(final_df: pd.DataFrame) -> None:
    """Basic validation of macro data quality"""
    print("\nMacro Data Validation:")
    print("=" * 40)
    
    # Get macro columns
    macro_cols = [c for c in final_df.columns if (
        c.startswith("growth_") or c.endswith(("_yoy", "_qoq"))
    )]
    
    print(f"Found {len(macro_cols)} macro columns")
    
    # Check for excessive nulls
    for col in macro_cols[:10]:  # Check first 10 to avoid spam
        null_pct = (final_df[col].isnull().sum() / len(final_df)) * 100
        if null_pct > 50:
            print(f"  Warning: {col} has {null_pct:.1f}% nulls")
        elif null_pct > 0:
            print(f"  {col}: {null_pct:.1f}% nulls")
        else:
            print(f"  {col}: Complete")
    
    if len(macro_cols) > 10:
        print(f"  ... and {len(macro_cols) - 10} more columns")
    
    # Sample some YoY values for reasonableness
    yoy_cols = [c for c in macro_cols if c.endswith("_yoy")]
    if yoy_cols:
        print(f"\nSample YoY values (should be reasonable percentages):")
        for col in yoy_cols[:3]:
            recent_vals = final_df[col].dropna().tail(5)
            if len(recent_vals) > 0:
                val_range = (recent_vals.min(), recent_vals.max())
                print(f"  {col}: [{val_range[0]:.4f}, {val_range[1]:.4f}]")

# Main execution
if __name__ == "__main__":
    print("MACRO DATA PIPELINE")
    print("=" * 40)
    
    # Assuming augmented_df exists from previous steps
    stock_start_date = augmented_df['Date'].min().strftime('%Y-%m-%d')
    print(f"Fetching macro data from {stock_start_date}")
    
    macros = fetch_and_build_macros(start=stock_start_date)
    final_df = merge_all_macros(augmented_df, macros)
    validate_macro_data(final_df)
    
    print(f"\nFinal dataset: {final_df.shape}")
    print("Pipeline complete!")