"""
Complete Stock Market Data Pipeline
===================================

This pipeline integrates:
1. Stock data fetching and basic feature engineering
2. Technical analysis indicators (TA-Lib)
3. Macro economic indicators
4. Data validation and quality checks

Run this as: python stock_pipeline.py
"""



import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import List, Dict, Optional

# Import from your existing modules
from extract import build_stock_dataframe  # This should work since your file is extract.py
from tickerfetch import get_combined_universe

from technicals import (
    talib_get_momentum_indicators_for_one_ticker,
    talib_get_volume_volatility_cycle_price_indicators, 
    talib_get_pattern_recognition_indicators,
    
)
from macros import fetch_and_build_macros, merge_all_macros, validate_macro_data
from augment import augment_with_ta_fixed,create_augmented_dataset  # if you need this specific function

# Then use the StockDataPipeline class from the previous artifact
class StockDataPipeline:
    """Complete stock market data pipeline with TA and macro indicators"""
    
    def __init__(self, 
                 tickers: List[str],
                 lookbacks: List[int] = [1, 3, 7, 30, 90, 252, 365],
                 horizons: List[int] = [30],
                 binarize_thresholds: Optional[Dict[int, float]] = None,
                 ma_windows: List[int] = [10, 20],
                 vol_window: int = 30,
                 risk_free: float = 0.045):
        """
        Initialize the pipeline with configuration parameters
        
        Parameters
        ----------
        tickers : List[str]
            List of stock tickers to fetch
        lookbacks : List[int]
            Historical lookback periods for growth calculations
        horizons : List[int]
            Forward-looking prediction horizons
        binarize_thresholds : Dict[int, float], optional
            Thresholds for binary target creation
        ma_windows : List[int]
            Moving average windows
        vol_window : int
            Volatility calculation window
        risk_free : float
            Risk-free rate for Sharpe ratio
        """
        self.tickers = tickers
        self.lookbacks = lookbacks
        self.horizons = horizons
        self.binarize_thresholds = binarize_thresholds
        self.ma_windows = ma_windows
        self.vol_window = vol_window
        self.risk_free = risk_free
        
        # Pipeline state
        self.raw_stock_df = None
        self.augmented_df = None
        self.macro_data = None
        self.final_df = None
        
    def step_1_fetch_stock_data(self):
        """Step 1: Fetch raw stock data and apply basic feature engineering"""
        print("Step 1: Fetching stock data and basic features...")
        print("-" * 50)
        
        # This would call your stock_data.build_stock_dataframe function
        self.raw_stock_df = build_stock_dataframe(
            tickers=self.tickers,
            lookbacks=self.lookbacks,
            horizons=self.horizons,
            binarize_thresholds=self.binarize_thresholds,
            ma_windows=self.ma_windows,
            vol_window=self.vol_window,
            risk_free=self.risk_free
        )
        
        # Normalize column names to handle case sensitivity
        self.raw_stock_df.columns = (
            self.raw_stock_df.columns
            .str.replace(' ', '_')
            .str.lower()
        )
        
        # Ensure proper column capitalization for TA functions
        rename_map = {}
        for col in self.raw_stock_df.columns:
            if col == 'date':
                rename_map[col] = 'Date'
            elif col == 'ticker':
                rename_map[col] = 'Ticker'
            elif col in ['open', 'high', 'low', 'close', 'volume']:
                rename_map[col] = col.capitalize()
        
        if rename_map:
            self.raw_stock_df.rename(columns=rename_map, inplace=True)
        
        print(f"Raw stock data: {self.raw_stock_df.shape}")
        print(f"Date range: {self.raw_stock_df['Date'].min()} to {self.raw_stock_df['Date'].max()}")
        print(f"Tickers: {', '.join(self.raw_stock_df['Ticker'].unique())}")
        
        return self.raw_stock_df
    
    def step_2_add_technical_indicators(self):
        """Step 2: Add technical analysis indicators"""
        print("\nStep 2: Adding technical analysis indicators...")
        print("-" * 50)
        
        if self.raw_stock_df is None:
            raise ValueError("Must run step_1_fetch_stock_data first")
        
        # This would call your create_augmented_dataset function
        self.augmented_df = create_augmented_dataset(self.raw_stock_df)
        
        # Count technical indicators
        basic_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        ta_indicators = [col for col in self.augmented_df.columns if col not in basic_cols]
        
        print(f"Technical analysis complete: {self.augmented_df.shape}")
        print(f"Added {len(ta_indicators)} technical indicators")
        
        # Sample indicators
        momentum_indicators = [col for col in ta_indicators if any(x in col.lower() for x in ['rsi', 'macd', 'adx', 'stoch'])]
        volume_indicators = [col for col in ta_indicators if any(x in col.lower() for x in ['obv', 'ad', 'mfi'])]
        pattern_indicators = [col for col in ta_indicators if col.startswith('cdl')]
        
        print(f"  - Momentum indicators: {len(momentum_indicators)}")
        print(f"  - Volume indicators: {len(volume_indicators)}")
        print(f"  - Pattern indicators: {len(pattern_indicators)}")
        
        return self.augmented_df
    
    def step_3_add_macro_indicators(self):
        """Step 3: Add macro economic indicators"""
        print("\nStep 3: Adding macro economic indicators...")
        print("-" * 50)
        
        if self.augmented_df is None:
            raise ValueError("Must run step_2_add_technical_indicators first")
        
        # Get start date from stock data
        stock_start_date = self.augmented_df['Date'].min().strftime('%Y-%m-%d')
        
        # Fetch macro data
        print(f"Fetching macro data from {stock_start_date}")
        self.macro_data = fetch_and_build_macros(
            start=stock_start_date,
            lookbacks=self.lookbacks
        )
        
        # Merge with stock data
        self.final_df = merge_all_macros(self.augmented_df, self.macro_data)
        
        # Validate macro data
        validate_macro_data(self.final_df)
        
        print(f"Macro integration complete: {self.final_df.shape}")
        
        return self.final_df
    
    def step_4_final_validation_and_cleanup(self):
        """Step 4: Final data validation and cleanup"""
        print("\nStep 4: Final validation and cleanup...")
        print("-" * 50)
        
        if self.final_df is None:
            raise ValueError("Must run step_3_add_macro_indicators first")
        
        # Data quality checks
        print("Data Quality Summary:")
        print(f"  Total rows: {len(self.final_df):,}")
        print(f"  Total columns: {len(self.final_df.columns):,}")
        print(f"  Date range: {self.final_df['Date'].min()} to {self.final_df['Date'].max()}")
        print(f"  Tickers: {', '.join(sorted(self.final_df['Ticker'].unique()))}")
        
        # Check for missing data
        null_summary = self.final_df.isnull().sum()
        high_null_cols = null_summary[null_summary > len(self.final_df) * 0.1]  # >10% missing
        
        if len(high_null_cols) > 0:
            print(f"\nColumns with >10% missing data:")
            for col, null_count in high_null_cols.head(10).items():
                pct = (null_count / len(self.final_df)) * 100
                print(f"  {col}: {pct:.1f}% missing")
        else:
            print("\nNo columns with excessive missing data")
        
        # Feature summary by category
        feature_categories = self._categorize_features()
        print(f"\nFeature Summary:")
        for category, features in feature_categories.items():
            print(f"  {category}: {len(features)} features")
        
        # Sort by Date and Ticker for consistency
        self.final_df = self.final_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        
        return self.final_df
    
    def _categorize_features(self):
        """Categorize features by type"""
        if self.final_df is None:
            return {}
        
        features = {
            'Basic OHLCV': [],
            'Time Features': [],
            'Growth Features': [],
            'Moving Averages': [],
            'Volatility': [],
            'Technical - Momentum': [],
            'Technical - Volume': [], 
            'Technical - Patterns': [],
            'Technical - Cycle': [],
            'Macro - Growth Rates': [],
            'Macro - YoY/QoQ': [],
            'Other': []
        }
        
        for col in self.final_df.columns:
            col_lower = col.lower()
            
            if col in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']:
                features['Basic OHLCV'].append(col)
            elif any(x in col_lower for x in ['year', 'month', 'weekday', 'wom']):
                features['Time Features'].append(col)
            elif col_lower.startswith('growth_') and not any(x in col_lower for x in ['btc', 'vix', 'dax', 'snp500', 'dji']):
                features['Growth Features'].append(col)
            elif any(x in col_lower for x in ['sma', 'moving_average']):
                features['Moving Averages'].append(col)
            elif any(x in col_lower for x in ['volatility', 'sharpe', 'atr', 'natr', 'trange']):
                features['Volatility'].append(col)
            elif any(x in col_lower for x in ['rsi', 'macd', 'adx', 'stoch', 'mom', 'roc', 'cci', 'willr']):
                features['Technical - Momentum'].append(col)
            elif any(x in col_lower for x in ['obv', 'ad', 'adosc', 'mfi']):
                features['Technical - Volume'].append(col)
            elif col_lower.startswith('cdl'):
                features['Technical - Patterns'].append(col)
            elif any(x in col_lower for x in ['ht_', 'avgprice', 'medprice', 'typprice', 'wclprice']):
                features['Technical - Cycle'].append(col)
            elif col_lower.startswith('growth_') and any(x in col_lower for x in ['btc', 'vix', 'dax', 'snp500', 'dji', 'gold', 'oil']):
                features['Macro - Growth Rates'].append(col)
            elif any(x in col_lower for x in ['_yoy', '_qoq']):
                features['Macro - YoY/QoQ'].append(col)
            else:
                features['Other'].append(col)
        
        return features
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish"""
        print("STOCK MARKET DATA PIPELINE")
        print("=" * 70)
        print(f"Processing {len(self.tickers)} tickers: {', '.join(self.tickers)}")
        print(f"Lookback periods: {self.lookbacks}")
        print(f"Prediction horizons: {self.horizons}")
        print()
        
        start_time = datetime.now()
        
        try:
            # Run all pipeline steps
            self.step_1_fetch_stock_data()
            self.step_2_add_technical_indicators()
            self.step_3_add_macro_indicators()
            self.step_4_final_validation_and_cleanup()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETE!")
            print("=" * 70)
            print(f"Total processing time: {duration:.1f} seconds")
            print(f"Final dataset: {self.final_df.shape[0]:,} rows × {self.final_df.shape[1]:,} columns")
            print(f"Ready for modeling!")
            
            return self.final_df
            
        except Exception as e:
            print(f"\nPipeline failed at step: {e}")
            raise
    
    # def save_checkpoint(self, filename: str = None):
    #     """Save the final dataset"""
    #     if self.final_df is None:
    #         raise ValueError("No data to save - run pipeline first")
        
    #     if filename is None:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"stock_data_complete_{timestamp}.parquet"
        
    #     self.final_df.to_parquet(filename)
    #     print(f"Data saved to: {filename}")
    #     return filename

    # def save_checkpoint(self, filename: str = None):
    #     """Save the final dataset"""
    #     if self.final_df is None:
    #         raise ValueError("No data to save - run pipeline first")
        
    #     # Create data directory if it doesn't exist
    #     import os
    #     data_dir = "../data"  # Go up one level, then into data folder
    #     os.makedirs(data_dir, exist_ok=True)
        
    #     if filename is None:
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"{data_dir}/stock_data_complete_{timestamp}.parquet"
    #     else:
    #         filename = f"{data_dir}/{filename}"
        
    #     self.final_df.to_parquet(filename)
    #     print(f"Data saved to: {filename}")
    #     return filename
    
    def save_checkpoint(self, filename: str = None):
        """Save the final dataset to project_root/data/ regardless of execution directory"""
        import os
        from pathlib import Path
        
        # Find project root by looking for pyproject.toml or .git
        current_path = Path(__file__).resolve()
        project_root = None
        
        for parent in [current_path] + list(current_path.parents):
            if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
                project_root = parent
                break
        
        if project_root is None:
            # Fallback: assume we're in data-prep, so project root is parent
            project_root = current_path.parent.parent
        
        # Data directory is always project_root/data
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_complete_{timestamp}.parquet"
        
        # Full path for the file
        filepath = data_dir / filename
        
        if self.final_df is None:
            raise ValueError("No data to save - run pipeline first")
        
        self.final_df.to_parquet(filepath)
        print(f"Data saved to: {filepath}")
        return str(filepath)

# # Example usage
# def main():
#     """Example of running the complete pipeline"""

#         # Step 0: Fetch ticker universe
#     print("Step 0: Fetching ticker universe...")
#     print("-" * 50)
    
#     universe_df = get_combined_universe(
#         include_sp500=True,
#         include_nasdaq100=True,
#         save_components=True
#     )
    
#     if universe_df.empty:
#         print("Failed to fetch any tickers!")
#         return None
    
#     # Extract ticker list
#     #TICKERS = universe_df['Ticker'].tolist()
#     TICKERS = universe_df['Ticker'].tolist()[:500]

    
#     print("Step 1: Getting data...")
#     print("-" * 50)

#     # Configuration
#     #TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
#     LOOKBACKS = [1, 3, 7, 30, 90, 252, 365]
#     HORIZONS = [30]
#     BINARY_THRESHOLDS = {30: 1.05}  # 5% gain threshold for 30-day horizon
    
#     # Initialize and run pipeline
#     pipeline = StockDataPipeline(
#         tickers=TICKERS,
#         lookbacks=LOOKBACKS,
#         horizons=HORIZONS,
#         binarize_thresholds=BINARY_THRESHOLDS
#     )
    
#     # Run complete pipeline
#     final_data = pipeline.run_complete_pipeline()
    
#     # Save results
#     pipeline.save_checkpoint()
    
#     return final_data


def main():
    """Run pipeline with batch processing for large ticker lists"""
    import gc
    
    # Step 0: Fetch ticker universe
    print("Step 0: Fetching ticker universe...")
    print("-" * 50)
    
    universe_df = get_combined_universe(
        include_sp500=True,
        include_nasdaq100=True,
        save_components=True
    )
    
    if universe_df.empty:
        print("Failed to fetch any tickers!")
        return None
    
    # Configuration
    all_tickers = universe_df['Ticker'].tolist()[:500]  # Limit to 500
    batch_size = 100  # Process in batches of 100
    LOOKBACKS = [1, 3, 7, 30, 90, 252, 365]
    HORIZONS = [30]
    BINARY_THRESHOLDS = {30: 1.05}
    
    print(f"Processing {len(all_tickers)} tickers in batches of {batch_size}")
    print("=" * 70)
    
    # Process in batches
    all_results = []
    saved_files = []
    total_batches = (len(all_tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_tickers), batch_size):
        batch_num = (i // batch_size) + 1
        batch_tickers = all_tickers[i:i + batch_size]
        
        print(f"\nBATCH {batch_num}/{total_batches}")
        print(f"Processing {len(batch_tickers)} tickers: {', '.join(batch_tickers[:5])}{'...' if len(batch_tickers) > 5 else ''}")
        print("=" * 70)
        
        try:
            # Initialize pipeline for this batch
            pipeline = StockDataPipeline(
                tickers=batch_tickers,
                lookbacks=LOOKBACKS,
                horizons=HORIZONS,
                binarize_thresholds=BINARY_THRESHOLDS
            )
            
            # Run pipeline for this batch
            batch_result = pipeline.run_complete_pipeline()
            
            # Save batch result
            batch_filename = f"stock_data_batch_{batch_num:02d}.parquet"
            pipeline.save_checkpoint(batch_filename)
            saved_files.append(batch_filename)
            all_results.append(batch_result)
            
            print(f"Batch {batch_num} completed: {batch_result.shape}")
            
            # Clean up memory
            del pipeline, batch_result
            gc.collect()
            
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            print(f"Skipping batch and continuing...")
            continue
    
    # Combine all successful batches
    if all_results:
        print(f"\nCombining {len(all_results)} successful batches...")
        final_data = pd.concat(all_results, ignore_index=True)
        
        # Save combined result
        combined_filename = f"stock_data_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        final_data.to_parquet(f"../data/{combined_filename}")
        
        print("=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Final combined dataset: {final_data.shape}")
        print(f"Successful batches: {len(all_results)}/{total_batches}")
        print(f"Combined data saved to: ../data/{combined_filename}")
        print(f"Individual batch files: {saved_files}")
        
        return final_data
    else:
        print("No batches completed successfully!")
        return None

if __name__ == "__main__":
    final_dataset = main()