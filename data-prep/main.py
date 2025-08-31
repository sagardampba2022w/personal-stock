import os
import sys
import gc
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure local imports resolve
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

from stock_pipeline import StockDataPipeline
from tickerfetch import get_combined_universe
from train_model_new import TrainModel  

# Small adapter to mimic a TransformData object (expects .transformed_df)
class _TransformAdapter:
    def __init__(self, df: pd.DataFrame):
        self.transformed_df = df

def process_batch(batch_tickers, batch_num, total_batches, config):
    """Process a single batch of tickers"""
    print(f"\nBATCH {batch_num}/{total_batches}")
    print(f"Processing {len(batch_tickers)} tickers: {', '.join(batch_tickers[:5])}{'...' if len(batch_tickers) > 5 else ''}")
    print("=" * 70)
    
    try:
        # Initialize pipeline for this batch
        pipeline = StockDataPipeline(
            tickers=batch_tickers,
            lookbacks=config['LOOKBACKS'],
            horizons=config['HORIZONS'],
            binarize_thresholds=config['BINARY_THRESHOLDS']
        )
        
        # Run pipeline for this batch
        batch_result = pipeline.run_complete_pipeline()
        
        # Save batch result
        batch_filename = f"stock_data_batch_{batch_num:02d}.parquet"
        pipeline.save_checkpoint(batch_filename)
        
        print(f"Batch {batch_num} completed: {batch_result.shape}")
        
        return batch_result, batch_filename
        
    except Exception as e:
        print(f"Error in batch {batch_num}: {e}")
        print(f"Skipping batch and continuing...")
        return None, None

def run_batch_pipeline(max_tickers=500, batch_size=100):
    """Run pipeline with batch processing for large ticker lists"""
    
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
    all_tickers = universe_df['Ticker'].tolist()[:max_tickers]
    config = {
        'LOOKBACKS': [1, 3, 7, 30, 90, 252, 365],
        'HORIZONS': [30],
        'BINARY_THRESHOLDS': {30: 1.05}
    }
    
    print(f"Processing {len(all_tickers)} tickers in batches of {batch_size}")
    print("=" * 70)
    
    # Process in batches
    all_results = []
    saved_files = []
    total_batches = (len(all_tickers) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_tickers), batch_size):
        batch_num = (i // batch_size) + 1
        batch_tickers = all_tickers[i:i + batch_size]
        
        batch_result, batch_filename = process_batch(
            batch_tickers, batch_num, total_batches, config
        )
        
        if batch_result is not None:
            all_results.append(batch_result)
            saved_files.append(batch_filename)
        
        # Clean up memory after each batch
        if batch_result is not None:
            del batch_result
        gc.collect()
    
    # Combine all successful batches
    if all_results:
        print(f"\nCombining {len(all_results)} successful batches...")
        final_data = pd.concat(all_results, ignore_index=True)
        
        # Save combined result
        combined_filename = f"stock_data_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        
        # Ensure data directory exists
        os.makedirs("../data", exist_ok=True)
        final_data.to_parquet(f"../data/{combined_filename}")
        
        print("=" * 70)
        print("BATCH PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Final combined dataset: {final_data.shape}")
        print(f"Successful batches: {len(all_results)}/{total_batches}")
        print(f"Combined data saved to: ../data/{combined_filename}")
        print(f"Individual batch files: {saved_files}")
        
        return final_data
    else:
        print("No batches completed successfully!")
        return None

def run_small_pipeline(num_tickers=10):
    """Run pipeline for small number of tickers (testing/development)"""
    
    # Get tickers
    universe_df = get_combined_universe(include_sp500=True, include_nasdaq100=True)
    tickers = universe_df['Ticker'].tolist()[:num_tickers]

    # Config
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
    
    return final_data

def train_and_evaluate_model(final_data):
    """Train and evaluate model on the dataset"""
    
    # Debug columns
    print("Available columns (first 10):", final_data.columns.tolist()[:10])
    print("Looking for date column in:", [col for col in final_data.columns if 'date' in col.lower()])

    # Train
    print("Training models...")
    tm = TrainModel(_TransformAdapter(final_data))
    tm.prepare_dataframe(start_date="2000-01-01")
    
    # Debug feature coverage
    tm.debug_feature_coverage()

    # VALIDATION: Check temporal split dates
    print("\n" + "="*60)
    print("DATA SPLIT VALIDATION")
    print("="*60)
    
    def get_split_date_ranges(df):
        """Get date ranges for each split"""
        ranges = {}
        for split in ['train', 'validation', 'test']:
            split_data = df[df['split'] == split]
            if len(split_data) > 0:
                ranges[split] = {
                    'start': split_data['Date'].min(),
                    'end': split_data['Date'].max(),
                    'count': len(split_data),
                    'unique_dates': split_data['Date'].nunique(),
                    'tickers': split_data['Ticker'].nunique() if 'Ticker' in split_data.columns else 'N/A'
                }
        return ranges
    
    split_ranges = get_split_date_ranges(tm.df_full)
    
    print("📅 TEMPORAL SPLIT SUMMARY:")
    for split_name, info in split_ranges.items():
        print(f"\n  {split_name.upper()}:")
        print(f"    📅 Date range: {info['start'].strftime('%Y-%m-%d')} to {info['end'].strftime('%Y-%m-%d')}")
        print(f"    📊 Duration: {(info['end'] - info['start']).days:,} days")
        print(f"    📈 Samples: {info['count']:,}")
        print(f"    📆 Unique dates: {info['unique_dates']:,}")
        print(f"    🏢 Tickers: {info['tickers']}")
        
        # Calculate years
        years = (info['end'] - info['start']).days / 365.25
        print(f"    ⏱️  Years: {years:.1f}")
    
    # Check for temporal overlap (should be none)
    print("\n🔍 TEMPORAL SPLIT VALIDATION:")
    
    # Check if splits are properly ordered
    splits_ordered = ['train', 'validation', 'test']
    properly_ordered = True
    
    for i in range(len(splits_ordered) - 1):
        current_split = splits_ordered[i]
        next_split = splits_ordered[i + 1]
        
        if current_split in split_ranges and next_split in split_ranges:
            current_end = split_ranges[current_split]['end']
            next_start = split_ranges[next_split]['start']
            
            if current_end >= next_start:
                print(f"  ❌ OVERLAP: {current_split} ends {current_end.strftime('%Y-%m-%d')}, {next_split} starts {next_start.strftime('%Y-%m-%d')}")
                properly_ordered = False
            else:
                gap_days = (next_start - current_end).days
                print(f"  ✅ {current_split} → {next_split}: {gap_days} day gap")
    
    if properly_ordered:
        print("  ✅ All splits are properly ordered with no temporal overlap")
    else:
        print("  ❌ WARNING: Temporal overlap detected - this may cause data leakage!")
    
    # Check target distribution across splits
    print(f"\n📊 TARGET DISTRIBUTION ({tm.target_col}):")
    target_dist = tm.df_full.groupby('split')[tm.target_col].agg(['count', 'mean', 'sum']).round(3)
    for split in target_dist.index:
        count = target_dist.loc[split, 'count']
        mean = target_dist.loc[split, 'mean']
        positive = target_dist.loc[split, 'sum']
        print(f"  {split.upper()}: {positive:.0f}/{count:.0f} positive ({mean:.1%})")
    
    # Check for significant distribution shifts
    train_pos_rate = target_dist.loc['train', 'mean'] if 'train' in target_dist.index else 0
    test_pos_rate = target_dist.loc['test', 'mean'] if 'test' in target_dist.index else 0
    
    if abs(train_pos_rate - test_pos_rate) > 0.1:  # 10% difference
        print(f"  ⚠️  WARNING: Large distribution shift between train ({train_pos_rate:.1%}) and test ({test_pos_rate:.1%})")
    else:
        print(f"  ✅ Reasonable distribution consistency across splits")
        
    print("="*60)

    print("\n" + "="*80)
    print("STEP 1: HONEST MODEL EVALUATION (Train-only model)")
    print("="*80)
    
    # Train on TRAIN only (honest evaluation)
    print("Training RandomForest on TRAIN set only...")
    tm.train_random_forest(max_depth=17, n_estimators=200, train_on='train', class_weight=None)

    # Verify no data leakage
    leaky = [c for c in tm.X_train_valid.columns if 'future' in c.lower()]
    print(f"✓ Leaky feature check: {len(leaky)} features with 'future' (should be 0)")
    
    if leaky:
        print(f"⚠️  WARNING: Found leaky features: {leaky[:5]}")

    # Find optimal threshold on CLEAN validation set
    best_thr = tm.find_best_threshold(split='validation', metric='f1')
    print(f"✓ Best threshold on validation (F1): {best_thr:.3f}")

    # HONEST evaluation on both validation and test
    print(f"\n📊 HONEST METRICS (Train-only model, threshold={best_thr:.3f}):")
    honest_metrics = tm.summarize_performance(threshold=best_thr, splits=("validation", "test"))
    print(honest_metrics.to_string(index=False))

    # Also show performance at standard 0.50 threshold
    print(f"\n📊 HONEST METRICS (Train-only model, threshold=0.50):")
    honest_metrics_50 = tm.summarize_performance(threshold=0.5, splits=("validation", "test"))
    print(honest_metrics_50.to_string(index=False))

    # Classification report on test set
    print(f"\n📋 Classification Report (TEST set @ threshold=0.50):")
    print(tm.text_classification_report(split="test", threshold=0.5))

    print("\n" + "="*80)
    print("STEP 2: PRODUCTION MODEL (Train+Valid model)")  
    print("="*80)

    # Now refit on train+valid for production use (more data = potentially better model)
    print("Refitting model on TRAIN+VALIDATION for production use...")
    tm.refit_on_train_valid(class_weight=None)
    
    # ⚠️  IMPORTANT: Validation metrics are now MEANINGLESS (data contamination)
    print("⚠️  NOTE: Validation set is now contaminated - only TEST metrics are valid!")

    # Save the production model
    tm.persist("./artifacts")
    print("✓ Saved production model to ./artifacts/")

    # Generate predictions on full dataset
    print("Generating predictions on full dataset...")
    tm.make_inference("rf_prob_30d")

    # Show recent predictions
    pred_cols = [c for c in tm.df_full.columns if c.startswith("rf_prob")]
    out_cols = ['Date','Ticker'] + pred_cols if {'Date','Ticker'}.issubset(tm.df_full.columns) else pred_cols
    print("\n📈 Recent Predictions (last 10 days):")
    print(tm.df_full[out_cols].tail(10))

    # FINAL evaluation - ONLY on test set (using production model)
    print(f"\n📊 FINAL TEST METRICS (Production model, threshold={best_thr:.3f}):")
    final_test_metrics = tm.summarize_performance(threshold=best_thr, splits=("test",))
    print(final_test_metrics.to_string(index=False))

    print(f"\n📊 FINAL TEST METRICS (Production model, threshold=0.50):")
    final_test_metrics_50 = tm.summarize_performance(threshold=0.5, splits=("test",))
    print(final_test_metrics_50.to_string(index=False))

    # Strategy evaluation
    print(f"\n📈 Trading Strategy Performance (TEST set):")
    topk_stats = tm.daily_topk_stats(k=5, split="test", pred_col="rf_prob_30d")
    print(f"  Daily Top-5 Strategy:")
    print(f"    Baseline hit rate: {topk_stats['daily_hitrate_baseline']:.1%}")  
    print(f"    Top-5 hit rate: {topk_stats['daily_hitrate_topk']:.1%}")
    
    improvement = topk_stats['daily_hitrate_topk'] - topk_stats['daily_hitrate_baseline']
    print(f"    Improvement: {improvement:.1%} {'✓' if improvement > 0 else '✗'}")
    
    if topk_stats['avg_growth_lift_topk_vs_all'] is not None:
        growth_lift = topk_stats['avg_growth_lift_topk_vs_all']
        print(f"    Growth lift: {growth_lift:.1%} {'✓' if growth_lift > 0 else '✗'}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Extract key metrics for summary
    test_honest = honest_metrics[honest_metrics['split'] == 'test'].iloc[0]
    test_final = final_test_metrics.iloc[0]
    
    print("🎯 KEY PERFORMANCE METRICS:")
    print(f"   Model: Random Forest (max_depth=17, n_estimators=200)")
    print(f"   Features: {len(tm.X_train_valid.columns)} (including {len(getattr(tm, 'DUMMIES', []))} dummies)")
    print(f"   Training samples: {tm.X_train.shape[0]:,}")
    print(f"   Test samples: {tm.X_test.shape[0]:,}")
    print()
    print(f"📊 HONEST EVALUATION (train-only model):")
    print(f"   Test ROC-AUC: {test_honest['roc_auc']:.3f}")
    print(f"   Test F1 (@ {best_thr:.3f}): {test_honest['f1']:.3f}")
    print(f"   Test Precision (@ 0.50): {honest_metrics_50[honest_metrics_50['split']=='test']['precision'].iloc[0]:.3f}")
    print(f"   Test Recall (@ 0.50): {honest_metrics_50[honest_metrics_50['split']=='test']['recall'].iloc[0]:.3f}")
    print()
    print(f"🏭 PRODUCTION MODEL:")
    print(f"   Test ROC-AUC: {test_final['roc_auc']:.3f}")
    print(f"   Test F1 (@ {best_thr:.3f}): {test_final['f1']:.3f}")
    print(f"   Daily Top-5 vs Baseline: +{improvement:.1%}")
    
    if improvement > 0.01:  # 1% improvement
        print(f"   Status: ✅ Model shows predictive power")
    elif improvement > 0:
        print(f"   Status: ⚠️  Marginal improvement")  
    else:
        print(f"   Status: ❌ No improvement over baseline")

    print(f"\n💾 Production model saved to: ./artifacts/")
    print(f"🎲 Random baseline hit rate: {topk_stats['daily_hitrate_baseline']:.1%}")
    print(f"🎯 Model hit rate (Top-5): {topk_stats['daily_hitrate_topk']:.1%}")
    
    print("\n" + "="*80)

def main():
    """Main function - choose between small pipeline or batch processing"""
    
    # Configuration options
    RUN_MODE = "batch"  # Options: "small", "batch"
    
    if RUN_MODE == "small":
        print("Running SMALL pipeline (10 tickers for testing)...")
        final_data = run_small_pipeline(num_tickers=10)
        
    elif RUN_MODE == "batch":
        print("Running BATCH pipeline (500 tickers in batches of 100)...")
        final_data = run_batch_pipeline(max_tickers=120, batch_size=100)
    
    else:
        print(f"Invalid RUN_MODE: {RUN_MODE}. Use 'small' or 'batch'")
        return
    
    if final_data is not None:
        print(f"\nData generation complete. Starting model training...")
        train_and_evaluate_model(final_data)
    else:
        print("Data generation failed!")

if __name__ == "__main__":
    main()