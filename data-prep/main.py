import os
import sys
import numpy as np
import pandas as pd

# Ensure local imports resolve
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

from stock_pipeline import StockDataPipeline
from tickerfetch import get_combined_universe
from train_model import TrainModel  

# Small adapter to mimic a TransformData object (expects .transformed_df)
class _TransformAdapter:
    def __init__(self, df: pd.DataFrame):
        self.transformed_df = df

def main():
    # Get tickers
    universe_df = get_combined_universe(include_sp500=True, include_nasdaq100=True)
    tickers = universe_df['Ticker'].tolist()[:10]  # Use few for smoke test

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

    # Debug columns
    print("Available columns (first 50):", final_data.columns.tolist()[:10])
    print("Looking for date column in:", [col for col in final_data.columns if 'date' in col.lower()])

    # Train
    print("Training models...")
    tm = TrainModel(_TransformAdapter(final_data))
    tm.prepare_dataframe(start_date="2000-01-01")
    # no future columns in features
    tm.debug_feature_coverage()



    # Train on TRAIN only (honest VAL), try class_weight if imbalanced
    tm.train_random_forest(max_depth=17, n_estimators=200, train_on='train', class_weight=None)

        # no future columns in features
    leaky = [c for c in tm.X_train_valid.columns if 'future' in c.lower()]
    print("Leaky feature candidates:", leaky)  # should be []

    # used = set(tm.X_train_valid.columns)
    # all_cols = set(tm.df_full.columns) - {tm.target_col}
    # unused = sorted(c for c in all_cols if c not in used and "future" not in c.lower())
    # print(f"Used features: {len(used)}  |  Unused (non-future): {len(unused)}")
    # print("Sample unused:", unused[:30])


    # Pick threshold on validation (try 'precision' or 'f1')
    best_thr = tm.find_best_threshold(split='validation', metric='f1')
    print(f"Best threshold on validation (f1): {best_thr:.2f}")

    print("\n=== Metrics (thr=best on VAL, precision) ===")
    print(tm.summarize_performance(threshold=best_thr, splits=("validation","test")).to_string(index=False))

    # Refit on TRAIN+VALID for final model & persist
    tm.refit_on_train_valid(class_weight=None)
    tm.persist("./artifacts")

    # Inference on full panel
    tm.make_inference("rf_prob_30d")

    # Sample predictions
    pred_cols = [c for c in tm.df_full.columns if c.startswith("rf_prob")]
    out_cols = ['Date','Ticker'] + pred_cols if {'Date','Ticker'}.issubset(tm.df_full.columns) else pred_cols
    print(tm.df_full[out_cols].tail(10))

    # Metrics @ 0.50 and @ best
    print("\n=== Metrics (thr=0.50) ===")
    metrics = tm.summarize_performance(threshold=0.5, splits=("validation","test"))
    print(metrics.to_string(index=False))

    print("\n=== Classification report (TEST @ 0.50) ===")
    print(tm.text_classification_report(split="test", threshold=0.5))

    print("\n=== Daily Top-5 stats on TEST ===")
    print(tm.daily_topk_stats(k=5, split="test", pred_col="rf_prob_30d"))

if __name__ == "__main__":
    main()
