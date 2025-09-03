#!/usr/bin/env python3
"""
Data Extraction Pipeline
========================

Separate script for data extraction, feature engineering, and saving.
Run this first before model training.

Usage:
    python run_data_extraction.py --mode small    # 10 tickers for testing
    python run_data_extraction.py --mode batch    # Full pipeline with batching
    python run_data_extraction.py --mode custom   # Custom ticker list
"""

import os
import sys
import gc
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

PROJECT_ROOT = HERE.parent  # <- CHANGE THIS
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from stock_pipeline import StockDataPipeline
from tickerfetch import get_combined_universe

class DataExtractionRunner:
    """Handles data extraction with different modes and configurations"""
    
    def __init__(self):
        self.config = {
            'LOOKBACKS': [1, 3, 7, 30, 90, 252, 365],
            'HORIZONS': [30],
            'BINARY_THRESHOLDS': {30: 1.01}  # 1% gain threshold
        }
        self.universe_df = None
        self.final_data = None
    
    def fetch_ticker_universe(self, include_sp500=True, include_nasdaq100=True):
        """Fetch ticker universe"""
        print("=" * 60)
        print("FETCHING TICKER UNIVERSE")
        print("=" * 60)
        
        self.universe_df = get_combined_universe(
            include_sp500=include_sp500,
            include_nasdaq100=include_nasdaq100,
            save_components=True
        )
        
        if self.universe_df.empty:
            raise ValueError("Failed to fetch any tickers!")
        
        print(f"✓ Fetched {len(self.universe_df)} unique tickers")
        return self.universe_df
    
    def run_small_extraction(self, num_tickers=10):
        """Extract data for small number of tickers (testing)"""
        print("=" * 60)
        print(f"SMALL DATA EXTRACTION ({num_tickers} tickers)")
        print("=" * 60)
        
        if self.universe_df is None:
            self.fetch_ticker_universe()
        
        tickers = self.universe_df['Ticker'].tolist()[:num_tickers]
        print(f"Processing tickers: {', '.join(tickers)}")
        
        # Run pipeline
        pipeline = StockDataPipeline(
            tickers=tickers,
            lookbacks=self.config['LOOKBACKS'],
            horizons=self.config['HORIZONS'],
            binarize_thresholds=self.config['BINARY_THRESHOLDS']
        )
        
        self.final_data = pipeline.run_complete_pipeline()
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_data_small_{num_tickers}tickers_{timestamp}.parquet"
        saved_path = pipeline.save_checkpoint(filename)
        
        print(f"\n✓ Small extraction complete!")
        print(f"  Data shape: {self.final_data.shape}")
        print(f"  Saved to: {saved_path}")
        
        return self.final_data, saved_path
    
    def run_batch_extraction(self, max_tickers=500, batch_size=50):
        """Extract data in batches for large ticker lists"""
        print("=" * 60)
        print(f"BATCH DATA EXTRACTION ({max_tickers} tickers, batch size: {batch_size})")
        print("=" * 60)
        
        if self.universe_df is None:
            self.fetch_ticker_universe()
        
        all_tickers = self.universe_df['Ticker'].tolist()[:max_tickers]
        total_batches = (len(all_tickers) + batch_size - 1) // batch_size
        
        print(f"Processing {len(all_tickers)} tickers in {total_batches} batches")
        
        saved_files = []
        successful_batches = 0
        
        for i in range(0, len(all_tickers), batch_size):
            batch_num = (i // batch_size) + 1
            batch_tickers = all_tickers[i:i + batch_size]
            
            print(f"\nBATCH {batch_num}/{total_batches}")
            print(f"Processing {len(batch_tickers)} tickers: {', '.join(batch_tickers[:3])}...")
            print("-" * 50)
            
            try:
                pipeline = StockDataPipeline(
                    tickers=batch_tickers,
                    lookbacks=self.config['LOOKBACKS'],
                    horizons=self.config['HORIZONS'],
                    binarize_thresholds=self.config['BINARY_THRESHOLDS']
                )
                
                batch_result = pipeline.run_complete_pipeline()
                
                # Save batch
                batch_filename = f"stock_data_batch_{batch_num:02d}.parquet"
                saved_path = pipeline.save_checkpoint(batch_filename)
                saved_files.append(saved_path)
                successful_batches += 1
                
                print(f"✓ Batch {batch_num} completed: {batch_result.shape}")
                
                # Clean up memory
                del pipeline, batch_result
                gc.collect()
                
            except Exception as e:
                print(f"✗ Batch {batch_num} failed: {e}")
                continue
        
        # Combine all batches
        if saved_files:
            print(f"\nCombining {len(saved_files)} successful batches...")
            all_data = []
            
            for filepath in saved_files:
                batch_data = pd.read_parquet(filepath)
                all_data.append(batch_data)
                print(f"  Loaded batch: {batch_data.shape}")
            
            self.final_data = pd.concat(all_data, ignore_index=True)
            
            # Save combined result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_filename = f"stock_data_combined_{timestamp}.parquet"
            combined_path = DATA_DIR / combined_filename
            self.final_data.to_parquet(combined_path, index=False)
            
            print(f"\n✓ Batch extraction complete!")
            print(f"  Final dataset: {self.final_data.shape}")
            print(f"  Successful batches: {successful_batches}/{total_batches}")
            print(f"  Combined data saved to: {combined_path}")
            print(f"  Individual batch files: {len(saved_files)} files")
            
            return self.final_data, str(combined_path), saved_files
        else:
            raise ValueError("No batches completed successfully!")
    
    def run_custom_extraction(self, ticker_list):
        """Extract data for custom ticker list"""
        print("=" * 60)
        print(f"CUSTOM DATA EXTRACTION ({len(ticker_list)} tickers)")
        print("=" * 60)
        print(f"Tickers: {', '.join(ticker_list)}")
        
        pipeline = StockDataPipeline(
            tickers=ticker_list,
            lookbacks=self.config['LOOKBACKS'],
            horizons=self.config['HORIZONS'],
            binarize_thresholds=self.config['BINARY_THRESHOLDS']
        )
        
        self.final_data = pipeline.run_complete_pipeline()
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_data_custom_{len(ticker_list)}tickers_{timestamp}.parquet"
        saved_path = pipeline.save_checkpoint(filename)
        
        print(f"\n✓ Custom extraction complete!")
        print(f"  Data shape: {self.final_data.shape}")
        print(f"  Saved to: {saved_path}")
        
        return self.final_data, saved_path
    
    def validate_extracted_data(self):
        """Validate the extracted data"""
        if self.final_data is None:
            print("No data to validate!")
            return
        
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        
        df = self.final_data
        
        # Basic info
        print(f"📊 Dataset Overview:")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        if 'Date' in df.columns:
            print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"  Date span: {(df['Date'].max() - df['Date'].min()).days:,} days")
        if 'Ticker' in df.columns:
            print(f"  Unique tickers: {df['Ticker'].nunique()}")
            print(f"  Sample tickers: {', '.join(df['Ticker'].unique()[:10])}")
        
        # Feature categories
        feature_categories = {
            'Basic OHLCV': [c for c in df.columns if c in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']],
            'Growth Features': [c for c in df.columns if c.startswith('growth_') and 'future' not in c],
            'Technical Indicators': [c for c in df.columns if any(x in c.lower() for x in ['rsi', 'macd', 'sma', 'adx', 'cci'])],
            'Candlestick Patterns': [c for c in df.columns if c.startswith('cdl')],
            'Macro Features': [c for c in df.columns if c.endswith(('_yoy', '_qoq')) or 'btc' in c.lower() or 'vix' in c.lower()],
            'Target Variables': [c for c in df.columns if 'future' in c and ('positive' in c or 'growth' in c)]
        }
        
        print(f"\n📈 Feature Categories:")
        for category, features in feature_categories.items():
            print(f"  {category}: {len(features)} features")
            if features and len(features) <= 5:
                print(f"    {', '.join(features)}")
        
        # Data quality
        print(f"\n🔍 Data Quality:")
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts > len(df) * 0.1]
        if len(high_null_cols) > 0:
            print(f"  Columns with >10% nulls: {len(high_null_cols)}")
            for col, null_count in high_null_cols.head(5).items():
                pct = (null_count / len(df)) * 100
                print(f"    {col}: {pct:.1f}% nulls")
        else:
            print(f"  ✓ No columns with excessive missing data")
        
        # Target distribution
        target_cols = [c for c in df.columns if 'is_positive' in c and 'future' in c]
        if target_cols:
            print(f"\n🎯 Target Variables:")
            for target in target_cols[:3]:  # Show first 3
                pos_rate = df[target].mean()
                print(f"  {target}: {pos_rate:.1%} positive")
        
        print(f"\n✅ Validation complete - data ready for modeling!")


def main():
    parser = argparse.ArgumentParser(description="Data Extraction Pipeline")
    parser.add_argument("--mode", choices=["small", "batch", "custom"], default="small",
                       help="Extraction mode: small (10 tickers), batch (500 tickers), custom (specify tickers)")
    parser.add_argument("--num-tickers", type=int, default=10,
                       help="Number of tickers for small mode")
    parser.add_argument("--max-tickers", type=int, default=500,
                       help="Maximum tickers for batch mode")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for batch mode")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
                       help="Custom ticker list for custom mode")
    parser.add_argument("--validate", action="store_true",
                       help="Run data validation after extraction")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = DataExtractionRunner()
    
    try:
        # Run extraction based on mode
        if args.mode == "small":
            result = runner.run_small_extraction(num_tickers=args.num_tickers)
        elif args.mode == "batch":
            result = runner.run_batch_extraction(
                max_tickers=args.max_tickers,
                batch_size=args.batch_size
            )
        elif args.mode == "custom":
            result = runner.run_custom_extraction(ticker_list=args.tickers)
        
        # Validate if requested
        if args.validate:
            runner.validate_extracted_data()
        
        print(f"\n🎉 Data extraction completed successfully!")
        print(f"Next steps:")
        print(f"  1. Run model training: python run_model_training.py")
        print(f"  2. Run predictions: python run_predictions.py")
        
    except Exception as e:
        print(f"\n❌ Data extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()