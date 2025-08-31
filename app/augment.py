import pandas as pd
import pandas as pd
import warnings

# Add this import to get the TA functions
from technicals import (
    talib_get_momentum_indicators_for_one_ticker,
    talib_get_volume_volatility_cycle_price_indicators,
    talib_get_pattern_recognition_indicators
)

def augment_with_ta_fixed(df_ticker, ticker_name=None):
    """Fixed version that handles missing Ticker column and column name consistency"""
    # 1) Copy and cast numeric columns
    df = df_ticker.copy()
    
    # 2) Handle missing Ticker column (when include_groups=False)
    if 'Ticker' not in df.columns and ticker_name is not None:
        df['Ticker'] = ticker_name
    elif 'Ticker' not in df.columns:
        # Fallback - shouldn't happen but just in case
        df['Ticker'] = 'UNKNOWN'
    
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    # 3) CRITICAL FIX: Normalize Date column BEFORE calling TA functions
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
    
    # 4) Generate TA frames
    mom = talib_get_momentum_indicators_for_one_ticker(df)
    vol = talib_get_volume_volatility_cycle_price_indicators(df)
    patt = talib_get_pattern_recognition_indicators(df)

    # 5) CRITICAL FIX: Normalize column names for consistency
    # TA functions return 'date'/'ticker', but we need 'Date'/'Ticker' for merging
    for ta_df in [mom, vol, patt]:
        if 'date' in ta_df.columns:
            ta_df.rename(columns={'date': 'Date'}, inplace=True)
        if 'ticker' in ta_df.columns:
            ta_df.rename(columns={'ticker': 'Ticker'}, inplace=True)
        
        # Ensure date consistency and normalize ticker
        ta_df['Date'] = pd.to_datetime(ta_df['Date']).dt.tz_localize(None)
        ta_df['Ticker'] = ta_df['Ticker'].astype(str).str.upper()

    # 6) Merge - now all Date/Ticker columns should be consistent
    merged = df.merge(mom, on=['Date','Ticker'], how='left') \
               .merge(vol, on=['Date','Ticker'], how='left') \
               .merge(patt, on=['Date','Ticker'], how='left')
    
    return merged

import warnings

def create_augmented_dataset(stocks_df):
    """Create clean augmented dataset with TA indicators"""
    
    print("Creating augmented dataset with technical indicators...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        
        # Apply TA augmentation to each ticker group
        augmented_parts = []
        unique_tickers = stocks_df['Ticker'].unique()
        
        for i, ticker in enumerate(unique_tickers, 1):
            print(f"Processing TA indicators for {ticker} ({i}/{len(unique_tickers)})...")
            
            ticker_df = stocks_df[stocks_df['Ticker'] == ticker].copy()
            ticker_df['Ticker'] = ticker_df['Ticker'].str.upper()
            augmented_ticker = augment_with_ta_fixed(ticker_df)
            augmented_parts.append(augmented_ticker)
        
        augmented_df = pd.concat(augmented_parts, ignore_index=True)
    
    return augmented_df

# def create_augmented_dataset(stocks_df):
#     """Create clean augmented dataset with TA indicators"""
    
#     print("Creating augmented dataset with technical indicators...")
    
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", FutureWarning)
        
#         # Apply TA augmentation to each ticker group
#         augmented_parts = []
#         for ticker in stocks_df['Ticker'].unique():
#             ticker_df = stocks_df[stocks_df['Ticker'] == ticker].copy()
#             ticker_df['Ticker'] = ticker_df['Ticker'].str.upper()
#             augmented_ticker = augment_with_ta_fixed(ticker_df)
#             augmented_parts.append(augmented_ticker)
        
#         augmented_df = pd.concat(augmented_parts, ignore_index=True)
    
#     return augmented_df

# # Create your clean augmented dataset
# augmented_df = create_augmented_dataset(stocks_df)

# print("SUCCESS! Technical Analysis Integration Complete")
# print("=" * 60)
# print(f"Dataset: {augmented_df.shape[0]} rows × {augmented_df.shape[1]} columns")
# print(f"Date range: {augmented_df['Date'].min()} to {augmented_df['Date'].max()}")
# print(f"Tickers: {', '.join(augmented_df['Ticker'].unique())}")

# # Count technical indicators (excluding basic OHLCV columns)
# basic_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
# ta_indicators = [col for col in augmented_df.columns if col not in basic_cols]
# print(f"Features: {len(basic_cols)} basic + {len(ta_indicators)} technical indicators")

# # Quick data quality check
# ticker_balance = augmented_df['Ticker'].value_counts()
# print(f"\nData Balance:")
# for ticker, count in ticker_balance.items():
#     ticker_data = augmented_df[augmented_df['Ticker'] == ticker]
#     years = (ticker_data['Date'].max() - ticker_data['Date'].min()).days / 365.25
#     print(f"  {ticker}: {count:,} rows ({years:.1f} years)")

# # Show sample technical indicators
# print(f"\nSample Technical Indicators ({len(ta_indicators)} total):")
# for indicator in ta_indicators[:15]:  # Show first 15
#     print(f"  • {indicator}")
# if len(ta_indicators) > 15:
#     print(f"  • ... and {len(ta_indicators)-15} more indicators")

# print(f"\nREADY FOR MACRO INTEGRATION!")
# print("=" * 60)
# print("Next step: Run your macro pipeline to get the final feature dataset")

# # Data quality check - look for any merge issues
# null_counts = augmented_df.isnull().sum()
# if null_counts.sum() > 0:
#     print(f"\nData Quality Check:")
#     high_null_cols = null_counts[null_counts > 0].head(10)
#     for col, null_count in high_null_cols.items():
#         pct = (null_count / len(augmented_df)) * 100
#         print(f"  {col}: {null_count:,} nulls ({pct:.1f}%)")

# print(f"\nFor macro integration, use this dataset:")
# print(f"   Variable name: augmented_df")
# print(f"   Shape: {augmented_df.shape}")
# print(f"   Date dtype: {augmented_df['Date'].dtype}")
# print(f"   Ready: Yes")