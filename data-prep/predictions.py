import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import glob
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Ensure local imports resolve
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

from train_model_new import TrainModel  

class _TransformAdapter:
    def __init__(self, df: pd.DataFrame):
        self.transformed_df = df

class PredictionComparator:
    """Compare different prediction strategies against each other"""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.prediction_cols = []
        
    def add_manual_predictions(self):
        """Add rule-based manual predictions"""
        print("Creating manual rule-based predictions...")
        
        def safe_column_access(df, col_name, default_value=0):
            if col_name in df.columns:
                return df[col_name].fillna(default_value)
            else:
                print(f"Warning: Column '{col_name}' not found, using default value {default_value}")
                return pd.Series(default_value, index=df.index)
        
        # pred0: CCI > 200 (momentum breakout strategy)
        cci = safe_column_access(self.df, 'cci', 0)
        self.df['pred0_manual_cci'] = (cci > 200).astype(int)
        
        # pred1: Previous 30d growth > 1 (momentum strategy)
        growth_30d = safe_column_access(self.df, 'growth_30d', 0)
        self.df['pred1_manual_prev_g1'] = (growth_30d > 1).astype(int)
        
        # pred2: Previous 30d growth > 1 AND S&P500 30d growth > 1
        growth_snp500_30d = safe_column_access(self.df, 'growth_snp500_30d', 0)
        self.df['pred2_manual_prev_g1_and_snp'] = (
            (growth_30d > 1) & (growth_snp500_30d > 1)
        ).astype(int)
        
        # pred3: Low interest rate environment
        dgs10 = safe_column_access(self.df, 'dgs10', 0)
        dgs5 = safe_column_access(self.df, 'dgs5', 0)
        self.df['pred3_manual_dgs10_5'] = (
            (dgs10 <= 4) & (dgs5 <= 1)
        ).astype(int)
        
        # pred4: High bond yields but accommodative fed
        fedfunds = safe_column_access(self.df, 'fedfunds', 0)
        self.df['pred4_manual_dgs10_fedfunds'] = (
            (dgs10 > 4) & (fedfunds <= 4.795)
        ).astype(int)
        
        manual_preds = ['pred0_manual_cci', 'pred1_manual_prev_g1', 'pred2_manual_prev_g1_and_snp', 
                       'pred3_manual_dgs10_5', 'pred4_manual_dgs10_fedfunds']
        self.prediction_cols.extend(manual_preds)
        
        print("Manual prediction summary:")
        for pred in manual_preds:
            positive_rate = self.df[pred].mean()
            print(f"  {pred}: {positive_rate:.1%} positive predictions")
        
        return manual_preds
    
    def add_ml_predictions(self, model, feature_cols: List[str], thresholds: List[float] = [0.21, 0.5, 0.65, 0.8, 0.9]):
        """Add ML model predictions at different thresholds - INCLUDING HIGH PRECISION THRESHOLDS"""
        print(f"Creating ML predictions at thresholds: {thresholds}")
        print("Including high thresholds (0.8, 0.9) for maximum precision strategies")
        
        # Build feature matrix matching training
        available_features = [col for col in feature_cols if col in self.df.columns]
        missing_features = [col for col in feature_cols if col not in self.df.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing, filling with 0")
            print(f"Sample missing: {missing_features[:5]}")
        
        # Create feature matrix
        X = pd.DataFrame(index=self.df.index)
        for col in feature_cols:
            if col in self.df.columns:
                X[col] = self.df[col].fillna(0)
            else:
                X[col] = 0
        
        # Generate probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]
        self.df['rf_prob_30d'] = y_pred_proba
        
        # Add binary predictions at different thresholds
        ml_preds = []
        for i, thresh in enumerate(thresholds):
            pred_name = f'pred{10+i}_rf_thresh_{int(thresh*100)}'
            self.df[pred_name] = (y_pred_proba >= thresh).astype(int)
            ml_preds.append(pred_name)
            
            positive_rate = self.df[pred_name].mean()
            print(f"  {pred_name}: {positive_rate:.1%} positive predictions")
        
        self.prediction_cols.extend(ml_preds)
        return ml_preds
    
    def add_ensemble_predictions(self):
        """Add ensemble predictions combining manual and ML"""
        print("Creating ensemble predictions...")
        
        manual_preds = [col for col in self.prediction_cols if 'manual' in col]
        ml_preds = [col for col in self.prediction_cols if 'rf_thresh' in col]
        
        if not manual_preds or not ml_preds:
            print("Need both manual and ML predictions for ensemble")
            return []
        
        ensemble_preds = []
        
        # Ensemble 1: ML high confidence OR manual momentum
        if 'pred10_rf_thresh_21' in ml_preds and 'pred1_manual_prev_g1' in manual_preds:
            self.df['pred20_ensemble_ml_or_momentum'] = (
                (self.df['pred10_rf_thresh_21'] == 1) | 
                (self.df['pred1_manual_prev_g1'] == 1)
            ).astype(int)
            ensemble_preds.append('pred20_ensemble_ml_or_momentum')
        
        # Ensemble 2: ML medium confidence AND good market timing
        if 'pred11_rf_thresh_50' in ml_preds and 'pred2_manual_prev_g1_and_snp' in manual_preds:
            self.df['pred21_ensemble_ml_and_market'] = (
                (self.df['pred11_rf_thresh_50'] == 1) & 
                (self.df['pred2_manual_prev_g1_and_snp'] == 1)
            ).astype(int)
            ensemble_preds.append('pred21_ensemble_ml_and_market')
        
        # Ensemble 3: Majority vote from manual rules
        if len(manual_preds) >= 3:
            manual_sum = self.df[manual_preds].sum(axis=1)
            self.df['pred22_ensemble_majority_manual'] = (manual_sum >= 3).astype(int)
            ensemble_preds.append('pred22_ensemble_majority_manual')
        
        self.prediction_cols.extend(ensemble_preds)
        
        for pred in ensemble_preds:
            positive_rate = self.df[pred].mean()
            print(f"  {pred}: {positive_rate:.1%} positive predictions")
        
        return ensemble_preds
    
    def evaluate_all_predictions(self, split: str = 'test') -> pd.DataFrame:
        """Evaluate all prediction strategies - RANKED BY PRECISION"""
        print(f"\nEvaluating all {len(self.prediction_cols)} prediction strategies on {split} set...")
        print("PRIMARY METRIC: PRECISION (minimizing false positives)")
        
        eval_df = self.df[self.df['split'] == split].copy()
        
        if len(eval_df) == 0:
            print(f"No data found for split '{split}'")
            return pd.DataFrame()
        
        results = []
        
        for pred_col in self.prediction_cols:
            if pred_col not in eval_df.columns:
                continue
                
            y_pred = eval_df[pred_col]
            y_true = eval_df[self.target_col]
            
            # Skip strategies that make no predictions
            if y_pred.sum() == 0:
                print(f"  Skipping {pred_col}: No positive predictions")
                continue
            
            metrics = self._calculate_prediction_metrics(y_true, y_pred, eval_df, pred_col)
            metrics['strategy'] = pred_col
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # RANK BY PRECISION (primary), then F1 (secondary), then daily_improvement (tertiary)
        results_df = results_df.sort_values(['precision', 'f1_score', 'daily_improvement'], 
                                          ascending=[False, False, False])
        
        return results_df
    
    def _calculate_prediction_metrics(self, y_true, y_pred, eval_df, pred_col):
        """Calculate metrics for a prediction strategy"""
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        n_predictions = y_pred.sum()
        prediction_rate = y_pred.mean()
        
        # Daily metrics
        baseline_hit_rate = y_true.mean()
        positive_preds = eval_df[eval_df[pred_col] == 1]
        
        if len(positive_preds) > 0:
            strategy_hit_rate = positive_preds[self.target_col].mean()
            improvement = strategy_hit_rate - baseline_hit_rate
            
            # Growth lift
            growth_lift = np.nan
            growth_col = 'growth_future_30d'
            if growth_col in eval_df.columns:
                baseline_growth = eval_df[growth_col].mean()
                strategy_growth = positive_preds[growth_col].mean()
                growth_lift = strategy_growth - baseline_growth
        else:
            strategy_hit_rate = 0.0
            improvement = -baseline_hit_rate
            growth_lift = np.nan
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_predictions': int(n_predictions),
            'prediction_rate': prediction_rate,
            'daily_hit_rate': strategy_hit_rate,
            'daily_baseline': baseline_hit_rate,
            'daily_improvement': improvement,
            'avg_growth_lift': growth_lift
        }
    
    def print_comparison_report(self, results_df: pd.DataFrame):
        """Print comprehensive comparison report - PRECISION FOCUSED"""
        print("\n" + "="*100)
        print("STRATEGY COMPARISON REPORT - RANKED BY PRECISION")
        print("="*100)
        
        if len(results_df) == 0:
            print("No results to display")
            return
        
        # Top strategies summary - precision focused
        print("TOP 10 STRATEGIES (ranked by PRECISION, then F1, then daily improvement):")
        print("-" * 100)
        
        summary_cols = ['strategy', 'precision', 'f1_score', 'recall', 'daily_improvement', 
                       'avg_growth_lift', 'prediction_rate', 'n_predictions']
        
        display_df = results_df[summary_cols].head(10).copy()
        
        # Format for better readability
        display_df['precision'] = (display_df['precision'] * 100).round(1).astype(str) + '%'
        display_df['f1_score'] = display_df['f1_score'].round(3)
        display_df['recall'] = (display_df['recall'] * 100).round(1).astype(str) + '%'
        display_df['daily_improvement'] = (display_df['daily_improvement'] * 100).round(1).astype(str) + '%'
        display_df['prediction_rate'] = (display_df['prediction_rate'] * 100).round(1).astype(str) + '%'
        
        # Handle growth lift formatting
        if 'avg_growth_lift' in display_df.columns:
            display_df['avg_growth_lift'] = display_df['avg_growth_lift'].apply(
                lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A"
            )
        
        print(display_df.to_string(index=False))
        
        # Precision tiers analysis
        print(f"\nPRECISION TIER ANALYSIS:")
        print("-" * 60)
        
        # Convert back to numeric for analysis
        precision_values = results_df['precision'].values
        
        high_precision = results_df[results_df['precision'] >= 0.6]
        medium_precision = results_df[(results_df['precision'] >= 0.4) & (results_df['precision'] < 0.6)]
        low_precision = results_df[results_df['precision'] < 0.4]
        
        tiers = [
            ('High Precision (≥60%)', high_precision),
            ('Medium Precision (40-60%)', medium_precision),
            ('Low Precision (<40%)', low_precision)
        ]
        
        for tier_name, tier_df in tiers:
            if len(tier_df) > 0:
                avg_precision = tier_df['precision'].mean()
                avg_improvement = tier_df['daily_improvement'].mean()
                avg_growth = tier_df['avg_growth_lift'].mean() if 'avg_growth_lift' in tier_df else np.nan
                
                print(f"  {tier_name}: {len(tier_df)} strategies")
                print(f"    Avg Precision: {avg_precision:.1%}")
                print(f"    Avg Daily Improvement: {avg_improvement:.1%}")
                if not pd.isna(avg_growth):
                    print(f"    Avg Growth Lift: {avg_growth:.1%}")
                print(f"    Best Strategy: {tier_df.iloc[0]['strategy']}")
        
        # Category analysis by precision
        print(f"\nSTRATEGY CATEGORY ANALYSIS (by precision):")
        print("-" * 60)
        
        categories = {
            'Manual Rules': results_df[results_df['strategy'].str.contains('manual')],
            'ML Thresholds': results_df[results_df['strategy'].str.contains('rf_thresh')], 
            'Ensemble': results_df[results_df['strategy'].str.contains('ensemble')]
        }
        
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                avg_precision = cat_df['precision'].mean()
                max_precision = cat_df['precision'].max()
                best_strategy = cat_df.iloc[0]['strategy'] if len(cat_df) > 0 else "N/A"
                avg_improvement = cat_df['daily_improvement'].mean()
                
                print(f"  {cat_name}: Avg Precision={avg_precision:.1%}, Max Precision={max_precision:.1%}")
                print(f"    Best: {best_strategy} | Avg Daily Improvement: {avg_improvement:.1%}")
        
        # Investment recommendation based on precision
        print(f"\nINVESTMENT STRATEGY RECOMMENDATIONS:")
        print("-" * 60)
        
        # High precision strategies for conservative portfolios
        conservative_strategies = results_df[results_df['precision'] >= 0.6]
        if len(conservative_strategies) > 0:
            best_conservative = conservative_strategies.iloc[0]
            print(f"🛡️  CONSERVATIVE (High Precision): {best_conservative['strategy']}")
            print(f"     Precision: {best_conservative['precision']:.1%}")
            print(f"     Hit Rate: {best_conservative['daily_hit_rate']:.1%}")
            print(f"     Frequency: {best_conservative['prediction_rate']:.1%} of time")
        
        # Balanced strategies
        balanced_strategies = results_df[
            (results_df['precision'] >= 0.4) & 
            (results_df['f1_score'] >= 0.3) &
            (results_df['prediction_rate'] >= 0.05)  # At least 5% prediction rate
        ]
        if len(balanced_strategies) > 0:
            best_balanced = balanced_strategies.iloc[0]
            print(f"⚖️  BALANCED (Good Precision + Activity): {best_balanced['strategy']}")
            print(f"     Precision: {best_balanced['precision']:.1%}")
            print(f"     F1: {best_balanced['f1_score']:.3f}")
            print(f"     Daily Improvement: {best_balanced['daily_improvement']:.1%}")
        
        # Growth-focused strategies (if growth data available)
        if 'avg_growth_lift' in results_df.columns:
            growth_strategies = results_df[
                (results_df['avg_growth_lift'] > 0) & 
                (results_df['precision'] >= 0.3)
            ].sort_values('avg_growth_lift', ascending=False)
            
            if len(growth_strategies) > 0:
                best_growth = growth_strategies.iloc[0]
                print(f"📈 GROWTH FOCUSED (Positive Growth Lift): {best_growth['strategy']}")
                print(f"     Growth Lift: {best_growth['avg_growth_lift']:.1%}")
                print(f"     Precision: {best_growth['precision']:.1%}")
                print(f"     Daily Improvement: {best_growth['daily_improvement']:.1%}")

def load_latest_data():
    """Load the most recent combined dataset"""
    data_dir = "../data"
    
    # Look for combined datasets first
    combined_files = glob.glob(os.path.join(data_dir, "stock_data_combined_*.parquet"))
    
    if combined_files:
        # Get the most recent combined file
        latest_file = max(combined_files, key=os.path.getctime)
        print(f"Loading latest combined dataset: {latest_file}")
        return pd.read_parquet(latest_file)
    
    # Fallback to batch files if no combined file
    batch_files = glob.glob(os.path.join(data_dir, "stock_data_batch_*.parquet"))
    
    if batch_files:
        print(f"Loading {len(batch_files)} batch files and combining...")
        batch_dfs = []
        for file in sorted(batch_files):
            batch_dfs.append(pd.read_parquet(file))
        return pd.concat(batch_dfs, ignore_index=True)
    
    # No data found
    raise FileNotFoundError(f"No data files found in {data_dir}")

def load_trained_model():
    """Load previously trained model"""
    model_dir = "./artifacts"
    
    if not os.path.exists(os.path.join(model_dir, "random_forest_model.joblib")):
        raise FileNotFoundError("No trained model found. Run training first.")
    
    # Create a dummy TrainModel instance and load the saved model
    dummy_df = pd.DataFrame({'dummy': [1]})  # Just for initialization
    tm = TrainModel(_TransformAdapter(dummy_df))
    tm.load(model_dir)
    
    print(f"Loaded trained model from {model_dir}")
    print(f"Model target: {tm.target_col}")
    print(f"Model features: {len(tm._inference_feature_columns)}")
    
    return tm

def compare_predictions_on_existing_data():
    """Main function to compare predictions using existing data"""
    
    print("="*80)
    print("PREDICTION COMPARISON PIPELINE")
    print("="*80)
    
    # Step 1: Load existing data
    print("Step 1: Loading existing data...")
    try:
        final_data = load_latest_data()
        print(f"✓ Loaded dataset: {final_data.shape}")
        print(f"  Date range: {final_data['Date'].min()} to {final_data['Date'].max()}")
        print(f"  Tickers: {final_data['Ticker'].nunique()}")
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Please run the data pipeline first to generate data.")
        return
    
    # Step 2: Prepare data for modeling (create splits, etc.)
    print("\nStep 2: Preparing data for modeling...")
    tm = TrainModel(_TransformAdapter(final_data))
    tm.prepare_dataframe(start_date="2000-01-01")
    
    # Print temporal split info
    split_summary = tm.df_full.groupby('split').agg({
        'Date': ['min', 'max', 'count'],
        tm.target_col if hasattr(tm, 'target_col') else 'is_positive_growth_30d_future': 'mean'
    }).round(3)
    print("Temporal split summary:")
    print(split_summary)
    
    # Step 3: Train model if not already trained
    print("\nStep 3: Training/Loading model...")
    try:
        # Try to load existing model
        trained_model = load_trained_model()
        print("✓ Using existing trained model")
        
        # Update the TrainModel instance with the loaded model
        tm.model = trained_model.model
        tm.target_col = trained_model.target_col
        tm._inference_feature_columns = trained_model._inference_feature_columns
        
    except FileNotFoundError:
        print("No existing model found. Training new model...")
        tm.train_random_forest(max_depth=17, n_estimators=200, train_on='train')
        best_thr = tm.find_best_threshold(split='validation', metric='f1')
        tm.refit_on_train_valid()
        tm.persist("./artifacts")
        print(f"✓ Trained and saved new model (best threshold: {best_thr:.3f})")
    
    # Step 4: Generate all predictions and compare
    print("\nStep 4: Generating and comparing predictions...")
    print("="*60)
    
    # Initialize comparator
    target_col = getattr(tm, 'target_col', 'is_positive_growth_30d_future')
    comparator = PredictionComparator(tm.df_full, target_col)
    
    # Add all prediction types
    manual_preds = comparator.add_manual_predictions()
    
    # Get ML feature columns
    feature_cols = getattr(tm, '_inference_feature_columns', tm.X_train_valid.columns.tolist())
    ml_preds = comparator.add_ml_predictions(tm.model, feature_cols, thresholds=[0.21, 0.5, 0.65, 0.8, 0.9])
    
    ensemble_preds = comparator.add_ensemble_predictions()
    
    # Step 5: Evaluate all strategies
    print(f"\nStep 5: Evaluating {len(comparator.prediction_cols)} strategies...")
    print("="*60)
    
    # Evaluate on test set
    test_results = comparator.evaluate_all_predictions(split='test')
    
    # Print comparison report
    comparator.print_comparison_report(test_results)
    
    # Step 6: Detailed analysis of top performers
    print(f"\nStep 6: Top Strategy Analysis...")
    print("="*60)
    
    if len(test_results) > 0:
        # Get top 3 strategies
        top_3 = test_results.head(3)
        
        print("DETAILED ANALYSIS - TOP 3 STRATEGIES:")
        for i, (idx, row) in enumerate(top_3.iterrows(), 1):
            strategy = row['strategy']
            print(f"\n{i}. {strategy}")
            print(f"   F1 Score: {row['f1_score']:.3f}")
            print(f"   Precision: {row['precision']:.1%} | Recall: {row['recall']:.1%}")
            print(f"   Hit Rate: {row['daily_hit_rate']:.1%} vs {row['daily_baseline']:.1%} baseline")
            print(f"   Improvement: {row['daily_improvement']:.1%}")
            print(f"   Predictions Made: {row['n_predictions']:,} ({row['prediction_rate']:.1%})")
            if not pd.isna(row['avg_growth_lift']):
                print(f"   Growth Lift: {row['avg_growth_lift']:.1%}")
        
        # Save detailed results to project results directory
        os.makedirs("./results", exist_ok=True)
        results_file = f"./results/prediction_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        test_results.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
    print("\n" + "="*80)
    print("PREDICTION COMPARISON COMPLETE")
    print("="*80)
    
    return test_results

def train_fresh_model_and_compare():
    """Alternative: Train fresh model from scratch and compare"""
    
    print("Training fresh model from scratch...")
    
    # This would call your existing pipeline
    from main_original import main as original_main  # Your previous main function
    
    # Run original pipeline
    final_data = original_main()
    
    if final_data is not None:
        return compare_predictions_on_existing_data()

def main():
    """Main function - choose between loading existing data or training fresh"""
    
    MODE = "load_existing"  # Options: "load_existing", "train_fresh"
    
    if MODE == "load_existing":
        print("MODE: Loading existing data and comparing predictions...")
        results = compare_predictions_on_existing_data()
        
    elif MODE == "train_fresh":
        print("MODE: Training fresh model and comparing...")
        results = train_fresh_model_and_compare()
        
    else:
        print(f"Invalid MODE: {MODE}")
        return
    
    if results is not None and len(results) > 0:
        top_strategy = results.iloc[0]
        print(f"\nWINNER (Highest Precision): {top_strategy['strategy']}")
        print(f"   Precision: {top_strategy['precision']:.1%}")
        print(f"   F1 Score: {top_strategy['f1_score']:.3f}")
        print(f"   Daily Improvement: {top_strategy['daily_improvement']:.1%}")
        print(f"   Prediction Rate: {top_strategy['prediction_rate']:.1%}")
        
        # Show precision spectrum
        print(f"\nPRECISION SPECTRUM:")
        precision_ranges = [
            (0.8, 1.0, "Ultra High"),
            (0.6, 0.8, "High"), 
            (0.4, 0.6, "Medium"),
            (0.0, 0.4, "Low")
        ]
        
        for min_p, max_p, label in precision_ranges:
            count = len(results[(results['precision'] >= min_p) & (results['precision'] < max_p)])
            if count > 0:
                avg_improvement = results[
                    (results['precision'] >= min_p) & (results['precision'] < max_p)
                ]['daily_improvement'].mean()
                print(f"   {label} Precision ({min_p:.0%}-{max_p:.0%}): {count} strategies, avg improvement: {avg_improvement:.1%}")

if __name__ == "__main__":
    main()