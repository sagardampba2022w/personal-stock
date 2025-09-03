#!/usr/bin/env python3
"""
Predictions and Strategy Comparison Pipeline
===========================================

Separate script for generating predictions, comparing strategies, and selecting optimal thresholds.
Run this after model training.

Usage:
    python run_predictions.py --mode compare      # Compare all prediction strategies
    python run_predictions.py --mode threshold    # Find optimal thresholds only
    python run_predictions.py --mode generate     # Generate predictions only
    python run_predictions.py --data-file custom_data.parquet  # Use specific data file
"""

import os
import sys
import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

PROJECT_ROOT = HERE.parent  # <- CHANGE THIS
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from train_model_new import TrainModel
from predictions import (
    load_latest_data, 
    load_model_and_features, 
    PredictionComparator, 
    _TransformAdapter
)

class PredictionRunner:
    """Handles prediction generation and strategy comparison"""
    
    def __init__(self, data_file: Optional[str] = None):
        self.data_file = data_file
        self.data = None
        self.train_model = None
        self.model = None
        self.feature_cols = None
        self.target_col = None
        self.comparator = None
        self.results = {}
    
    def load_data_and_model(self):
        """Load data and trained model"""
        print("=" * 60)
        print("LOADING DATA AND MODEL")
        print("=" * 60)
        
        # Load data
        if self.data_file:
            data_path = Path(self.data_file)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            self.data = pd.read_parquet(data_path)
            print(f"✓ Data loaded from: {data_path.name}")
        else:
            self.data = load_latest_data()
            print("✓ Latest data loaded")
        
        print(f"  Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]:,} columns")
        if 'Date' in self.data.columns:
            print(f"  Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        if 'Ticker' in self.data.columns:
            print(f"  Tickers: {self.data['Ticker'].nunique()} unique")
        
        # Prepare TrainModel (for data splits)
        print("\n🔧 Preparing data splits...")
        self.train_model = TrainModel(_TransformAdapter(self.data))
        self.train_model.prepare_dataframe(start_date="2000-01-01")
        
        # Print split info
        split_summary = self.train_model.df_full.groupby("split").agg({
            "Date": ["min", "max", "count"]
        })
        print("  Split summary:")
        for split in ['train', 'validation', 'test']:
            if split in split_summary.index:
                dates = split_summary.loc[split, 'Date']
                print(f"    {split}: {dates['count']:,} samples ({dates['min']} to {dates['max']})")
        
        # Load model
        print("\n🤖 Loading trained model...")
        try:
            self.model, self.feature_cols, target_from_model = load_model_and_features(str(ARTIFACTS_DIR))
            print(f"✓ Model loaded with {len(self.feature_cols)} features")
            
            # Set target column
            self.target_col = target_from_model or getattr(self.train_model, 'target_col', 'is_positive_growth_30d_future')
            print(f"  Target column: {self.target_col}")
            
            # Set model in train_model for consistency
            self.train_model.model = self.model
            self.train_model._inference_feature_columns = self.feature_cols
            if target_from_model:
                self.train_model.target_col = target_from_model
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}\nMake sure to run training first!")
        
        return self.data, self.model
    
    
    
    def generate_predictions(self):
        """Generate all types of predictions"""
        print("\n" + "=" * 60)
        print("GENERATING PREDICTIONS")
        print("=" * 60)
        
        if self.train_model is None:
            self.load_data_and_model()
        
        # Initialize comparator
        self.comparator = PredictionComparator(self.train_model.df_full, self.target_col)
        
        # 1. Manual rule-based predictions
        print("🔧 Creating manual rule-based predictions...")
        manual_preds = self.comparator.add_manual_predictions()
        print(f"  Created {len(manual_preds)} manual strategies")
        
        # 2. ML predictions with fixed thresholds
        print("\n🤖 Creating ML predictions with fixed thresholds...")
        ml_preds = self.comparator.add_ml_predictions(
            self.model, 
            self.feature_cols,
            thresholds=(0.21, 0.50, 0.65, 0.80, 0.90)
        )
        print(f"  Created {len(ml_preds)} ML threshold strategies")
        
        # 3. Adaptive ML thresholds based on validation set
        print("\n🎯 Creating adaptive ML thresholds...")
        auto_thresholds = self.comparator.add_ml_thresholds_from_validation(
            proba_col="rf_prob_30d",
            target_rates=(0.01, 0.03, 0.05)
        )
        print(f"  Created {len(auto_thresholds)} adaptive threshold strategies")
        for col, thr in auto_thresholds.items():
            print(f"    {col}: threshold = {thr:.3f}")
        
        # 4. Daily top-K strategies
        print("\n📈 Creating daily top-K strategies...")
        topk_strategies = []
        for k in [3, 5, 10]:
            topk_col = self.comparator.add_daily_topn(
                proba_col="rf_prob_30d", 
                n=k
            )
            if topk_col:
                topk_strategies.append(topk_col)
        print(f"  Created {len(topk_strategies)} top-K strategies")
        
        # 5. Ensemble strategies
        print("\n🔗 Creating ensemble strategies...")
        ensemble_preds = self.comparator.add_ensemble_predictions()
        print(f"  Created {len(ensemble_preds)} ensemble strategies")
        
        total_strategies = len(self.comparator.prediction_cols)
        print(f"\n✅ Prediction generation complete!")
        print(f"  Total strategies created: {total_strategies}")
        
        return self.comparator
    
    def find_optimal_thresholds(self, validation_metrics=['precision', 'f1', 'recall']):
        """Find optimal thresholds for different objectives"""
        print("\n" + "=" * 60)
        print("OPTIMAL THRESHOLD ANALYSIS")
        print("=" * 60)
        
        if self.train_model is None:
            self.load_data_and_model()
        
        # Generate base probabilities if not done
        if 'rf_prob_30d' not in self.train_model.df_full.columns:
            print("Generating base probabilities...")
            self.train_model.make_inference("rf_prob_30d")
        
        # Find optimal thresholds for different metrics
        optimal_thresholds = {}
        
        for metric in validation_metrics:
            print(f"\n🎯 Finding optimal threshold for {metric.upper()}...")
            try:
                best_thr = self.train_model.find_best_threshold(
                    split='validation', 
                    metric=metric
                )
                optimal_thresholds[metric] = best_thr
                print(f"  Best {metric} threshold: {best_thr:.3f}")
                
                # Evaluate on test set
                test_perf = self.train_model.summarize_performance(
                    threshold=best_thr,
                    splits=('test',)
                )
                test_row = test_perf.iloc[0]
                print(f"  Test {metric}: {test_row[metric]:.3f}")
                print(f"  Test precision: {test_row['precision']:.3f}")
                print(f"  Test recall: {test_row['recall']:.3f}")
                
            except Exception as e:
                print(f"  Failed to find threshold for {metric}: {e}")
                optimal_thresholds[metric] = 0.5
        
        # ROC-based threshold (Youden's J statistic)
        print(f"\n📊 Finding ROC-optimal threshold (Youden's J)...")
        try:
            roc_threshold = self.train_model.find_best_threshold(
                split='validation',
                metric='youden'
            )
            optimal_thresholds['youden'] = roc_threshold
            print(f"  Youden threshold: {roc_threshold:.3f}")
            
            # Evaluate
            roc_perf = self.train_model.summarize_performance(
                threshold=roc_threshold,
                splits=('test',)
            )
            roc_row = roc_perf.iloc[0]
            print(f"  Test balanced accuracy: {roc_row['balanced_accuracy']:.3f}")
            
        except Exception as e:
            print(f"  Failed to find Youden threshold: {e}")
            optimal_thresholds['youden'] = 0.5
        
        # Summary comparison
        print(f"\n📋 THRESHOLD SUMMARY:")
        print("-" * 50)
        for metric, threshold in optimal_thresholds.items():
            print(f"  {metric.upper():>12}: {threshold:.3f}")
        
        # Save threshold recommendations
        threshold_file = RESULTS_DIR / f"optimal_thresholds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(threshold_file, 'w') as f:
            json.dump(optimal_thresholds, f, indent=2)
        print(f"\n💾 Thresholds saved to: {threshold_file}")
        
        self.results['optimal_thresholds'] = optimal_thresholds
        return optimal_thresholds
    
    def compare_all_strategies(self, split='test'):
        """Compare all prediction strategies"""
        print("\n" + "=" * 60)
        print(f"STRATEGY COMPARISON ({split.upper()} SET)")
        print("=" * 60)
        
        if self.comparator is None:
            self.generate_predictions()
        
        # Evaluate all strategies
        print(f"🔍 Evaluating {len(self.comparator.prediction_cols)} strategies on {split} set...")
        results_df = self.comparator.evaluate_all_predictions(split=split)
        
        if results_df.empty:
            print("❌ No results generated!")
            return None
        
        # Print comparison report
        self.comparator.print_comparison_report(results_df)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = RESULTS_DIR / f"strategy_comparison_{split}_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Store results
        self.results['strategy_comparison'] = {
            'split': split,
            'results_df': results_df,
            'results_file': str(results_file),
            'best_strategy': results_df.iloc[0].to_dict() if len(results_df) > 0 else None
        }
        
        return results_df
    
    def analyze_prediction_patterns(self):
        """Analyze prediction patterns and correlations"""
        print("\n" + "=" * 60)
        print("PREDICTION PATTERN ANALYSIS")
        print("=" * 60)
        
        if self.comparator is None:
            self.generate_predictions()
        
        # Analyze prediction correlations
        test_data = self.comparator.df[self.comparator.df['split'] == 'test'].copy()
        
        # Get top strategies for correlation analysis
        if 'strategy_comparison' in self.results and self.results['strategy_comparison']['results_df'] is not None:
            top_strategies = self.results['strategy_comparison']['results_df'].head(10)['strategy'].tolist()
        else:
            # Fallback: use first 10 prediction columns
            top_strategies = self.comparator.prediction_cols[:10]
        
        print(f"🔗 Analyzing correlations between top {len(top_strategies)} strategies...")
        
        # Calculate correlations
        pred_matrix = test_data[top_strategies]
        correlations = pred_matrix.corr()
        
        # Find highly correlated strategies (>0.8)
        high_corr_pairs = []
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                corr_val = correlations.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append({
                        'strategy1': correlations.columns[i],
                        'strategy2': correlations.columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            print(f"  Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.8):")
            for pair in high_corr_pairs[:5]:  # Show top 5
                print(f"    {pair['strategy1']} ↔ {pair['strategy2']}: r={pair['correlation']:.3f}")
        else:
            print("  No highly correlated strategies found")
        
        # Analyze prediction frequency by strategy type
        print(f"\n📊 Prediction frequency by strategy type:")
        
        strategy_types = {
            'Manual Rules': [s for s in top_strategies if 'manual' in s],
            'ML Fixed Threshold': [s for s in top_strategies if 'rf_thresh' in s],
            'ML Adaptive Threshold': [s for s in top_strategies if 'rf_auto' in s],
            'Top-K Daily': [s for s in top_strategies if 'top' in s and 'daily' in s],
            'Ensemble': [s for s in top_strategies if 'ens_' in s or 'ensemble' in s]
        }
        
        for strategy_type, strategies in strategy_types.items():
            if strategies:
                avg_freq = test_data[strategies].mean().mean()
                print(f"  {strategy_type}: {avg_freq:.1%} average prediction rate")
        
        # Save correlation analysis
        corr_file = RESULTS_DIR / f"prediction_correlations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        correlations.to_csv(corr_file)
        print(f"\n💾 Correlation matrix saved to: {corr_file}")
        
        self.results['pattern_analysis'] = {
            'correlations': correlations,
            'high_corr_pairs': high_corr_pairs,
            'strategy_types': strategy_types
        }
        
        return correlations
    
    def generate_final_recommendations(self):
        """Generate final investment strategy recommendations"""
        print("\n" + "=" * 80)
        print("INVESTMENT STRATEGY RECOMMENDATIONS")
        print("=" * 80)
        
        if 'strategy_comparison' not in self.results:
            print("❌ No strategy comparison results available. Run comparison first!")
            return
        
        results_df = self.results['strategy_comparison']['results_df']
        if results_df.empty:
            print("❌ No strategy results to analyze!")
            return
        
        # Conservative strategy (high precision, low risk)
        conservative = results_df[results_df['precision'] >= 0.6].head(1)
        
        # Balanced strategy (good precision + reasonable activity)
        balanced = results_df[
            (results_df['precision'] >= 0.4) & 
            (results_df['f1_score'] >= 0.3) &
            (results_df['prediction_rate'] >= 0.02)
        ].head(1)
        
        # Aggressive strategy (high recall, catch more opportunities)
        aggressive = results_df[results_df['recall'] >= 0.5].head(1)
        
        # Best overall F1
        best_f1 = results_df.head(1)
        
        recommendations = {
            '🛡️  CONSERVATIVE (High Precision)': conservative,
            '⚖️  BALANCED (Precision + Activity)': balanced, 
            '🚀 AGGRESSIVE (High Recall)': aggressive,
            '🏆 BEST F1 (Overall Performance)': best_f1
        }
        
        print("📋 STRATEGY RECOMMENDATIONS:")
        print("=" * 60)
        
        for strategy_name, strategy_df in recommendations.items():
            if not strategy_df.empty:
                row = strategy_df.iloc[0]
                print(f"\n{strategy_name}")
                print(f"  Strategy: {row['strategy']}")
                print(f"  Precision: {row['precision']:.1%}")
                print(f"  Recall: {row['recall']:.1%}")
                print(f"  F1-Score: {row['f1_score']:.3f}")
                print(f"  Prediction Rate: {row['prediction_rate']:.1%}")
                print(f"  Daily Improvement: {row['daily_improvement']:.1%}")
                
                # Add interpretation
                if 'conservative' in strategy_name.lower():
                    print(f"  💡 Use when: You want to minimize false positives")
                elif 'balanced' in strategy_name.lower():
                    print(f"  💡 Use when: You want good performance with regular signals")
                elif 'aggressive' in strategy_name.lower():
                    print(f"  💡 Use when: You don't want to miss opportunities")
                else:
                    print(f"  💡 Use when: You want the best overall performance")
            else:
                print(f"\n{strategy_name}")
                print(f"  ❌ No strategies meet these criteria")
        
        # Optimal thresholds summary
        if 'optimal_thresholds' in self.results:
            print(f"\n🎯 OPTIMAL THRESHOLD RECOMMENDATIONS:")
            print("-" * 40)
            thresholds = self.results['optimal_thresholds']
            for objective, threshold in thresholds.items():
                print(f"  {objective.upper():>12}: {threshold:.3f}")
            
            print(f"\n💡 Threshold Usage Guide:")
            print(f"  • Precision-focused: Use precision threshold ({thresholds.get('precision', 0.5):.3f})")
            print(f"  • Balanced approach: Use F1 threshold ({thresholds.get('f1', 0.5):.3f})")
            print(f"  • Opportunity-focused: Use recall threshold ({thresholds.get('recall', 0.5):.3f})")
        
        # Save recommendations
        rec_file = RESULTS_DIR / f"investment_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(rec_file, 'w') as f:
            f.write("INVESTMENT STRATEGY RECOMMENDATIONS\n")
            f.write("=" * 50 + "\n\n")
            
            for strategy_name, strategy_df in recommendations.items():
                if not strategy_df.empty:
                    row = strategy_df.iloc[0]
                    f.write(f"{strategy_name}\n")
                    f.write(f"Strategy: {row['strategy']}\n")
                    f.write(f"Precision: {row['precision']:.1%}\n")
                    f.write(f"F1-Score: {row['f1_score']:.3f}\n")
                    f.write(f"Prediction Rate: {row['prediction_rate']:.1%}\n\n")
        
        print(f"\n💾 Recommendations saved to: {rec_file}")
        
        self.results['recommendations'] = recommendations
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Predictions and Strategy Pipeline")
    parser.add_argument("--mode", choices=["generate", "threshold", "compare", "analyze", "recommend"], 
                       default="compare",
                       help="Prediction mode: generate (predictions only), threshold (find optimal), compare (all strategies), analyze (patterns), recommend (final recommendations)")
    parser.add_argument("--data-file", type=str,
                       help="Specific data file to use (otherwise uses latest)")
    parser.add_argument("--split", choices=["test", "validation", "train"], default="test",
                       help="Data split to evaluate on")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Save prediction probabilities to file")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PredictionRunner(data_file=args.data_file)
    
    try:
        if args.mode == "generate":
            runner.generate_predictions()
            
        elif args.mode == "threshold":
            runner.find_optimal_thresholds()
            
        elif args.mode == "compare":
            runner.compare_all_strategies(split=args.split)
            
        elif args.mode == "analyze":
            runner.generate_predictions()
            runner.compare_all_strategies(split=args.split)
            runner.analyze_prediction_patterns()
            
        elif args.mode == "recommend":
            # Full pipeline
            runner.generate_predictions()
            runner.find_optimal_thresholds()
            runner.compare_all_strategies(split=args.split)
            runner.analyze_prediction_patterns()
            runner.generate_final_recommendations()
        
        # Save predictions if requested
        if args.save_predictions and runner.comparator is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pred_file = RESULTS_DIR / f"predictions_{timestamp}.parquet"
            runner.comparator.df.to_parquet(pred_file, index=False)
            print(f"\n💾 All predictions saved to: {pred_file}")
        
        print(f"\n🎉 Prediction pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Prediction pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()