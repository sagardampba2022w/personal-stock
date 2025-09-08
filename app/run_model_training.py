#!/usr/bin/env python3
"""
Model Training and Hyperparameter Tuning Pipeline
=================================================

Separate script for model training, hyperparameter tuning, and evaluation.
Run this after data extraction.

Usage:
    python run_model_training.py --mode basic        # Basic training with default params
    python run_model_training.py --mode tune         # Hyperparameter tuning
    python run_model_training.py --mode evaluate     # Load existing model and evaluate
    python run_model_training.py --data-file custom_data.parquet  # Use specific data file
"""

import os
import sys
import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import joblib
import json


HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))

PROJECT_ROOT = HERE.parent  # <- CHANGE THIS
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)



from train_model_new import TrainModel
from hyperparameter_tuning import RFHyperparameterTuner

class DataFrameAdapter:
    """Simple adapter for TrainModel"""
    def __init__(self, df):
        self.transformed_df = df

class ModelTrainingRunner:
    """Handles model training with different modes"""
    
    def __init__(self, data_file: Optional[str] = None):
        self.data_file = data_file
        self.data = None
        self.train_model = None
        self.results = {}
    
    def load_data(self):
        """Load data from file"""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        if self.data_file:
            data_path = Path(self.data_file)
        else:
            # Find most recent data file
            data_path = self._find_latest_data_file()
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading data from: {data_path.name}")
        self.data = pd.read_parquet(data_path)
        
        print(f"✓ Data loaded: {self.data.shape[0]:,} rows × {self.data.shape[1]:,} columns")
        if 'Date' in self.data.columns:
            print(f"  Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        if 'Ticker' in self.data.columns:
            print(f"  Tickers: {self.data['Ticker'].nunique()} unique")
        
        return self.data
    
    def _find_latest_data_file(self):
        """Find the most recent data file"""
        patterns = [
            "stock_data_combined_*.parquet",
            "stock_data_complete_*.parquet",
            "stock_data_small_*.parquet",
            "stock_data_custom_*.parquet",
            "stock_data_batch_*.parquet"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in patterns:
            files = list(DATA_DIR.glob(pattern))
            for file in files:
                if file.stat().st_mtime > latest_time:
                    latest_time = file.stat().st_mtime
                    latest_file = file
        
        if latest_file is None:
            raise FileNotFoundError(f"No data files found in {DATA_DIR}. Run data extraction first!")
        
        print(f"Auto-detected latest data file: {latest_file.name}")
        return latest_file
    
    def prepare_model(self, start_date: Optional[str] = None):
        """Prepare TrainModel instance"""
        print("\n" + "=" * 60)
        print("PREPARING MODEL")
        print("=" * 60)
        
        if self.data is None:
            self.load_data()
        
        # Initialize TrainModel
        self.train_model = TrainModel(DataFrameAdapter(self.data))
        
        # Prepare data splits
        prepare_params = {}
        if start_date:
            prepare_params['start_date'] = start_date
            print(f"Using start date: {start_date}")
        else:
            # Use reasonable default (last 10 years or all available data)
            min_date = self.data['Date'].min()
            default_start = max(min_date, pd.Timestamp('2010-01-01'))
            prepare_params['start_date'] = default_start.strftime('%Y-%m-%d')
            print(f"Using default start date: {prepare_params['start_date']}")
        
        self.train_model.prepare_dataframe(**prepare_params)
        
        # Print data split summary
        print(f"\n📊 Data Splits:")
        print(f"  Train: {len(self.train_model.X_train):,} samples")
        print(f"  Validation: {len(self.train_model.X_valid):,} samples")
        print(f"  Test: {len(self.train_model.X_test):,} samples")
        print(f"  Features: {len(self.train_model.X_train.columns):,}")
        print(f"  Target: {self.train_model.target_col}")
        
        # Show target distribution
        target_dist = {
            'train': self.train_model.y_train.mean(),
            'validation': self.train_model.y_valid.mean(),
            'test': self.train_model.y_test.mean()
        }
        print(f"  Target distribution: {', '.join([f'{k}: {v:.1%}' for k, v in target_dist.items()])}")
        
        return self.train_model
    
    def run_basic_training(self, max_depth=None, n_estimators=100, class_weight=None):
        """Run basic model training with default hyperparameters"""
        print("\n" + "=" * 60)
        print("BASIC MODEL TRAINING")
        print("=" * 60)
        
        if self.train_model is None:
            self.prepare_model()

             # Map CLI string to actual None/balanced
        if isinstance(class_weight, str):
            class_weight = None if class_weight.strip().lower() == "none" else class_weight
        
        
        print(f"Training Random Forest:")
        print(f"  max_depth: {max_depth}")
        print(f"  n_estimators: {n_estimators}")
        print(f"  class_weight: {class_weight}")

        # Train model on train set only (honest evaluation)
        print("\n🔧 Training on TRAIN set only (for honest evaluation)...")
        self.train_model.train_random_forest(
            max_depth=max_depth,
            n_estimators=n_estimators,
            train_on='train',
            class_weight=class_weight
        )
        
        # Find optimal threshold
        print("🎯 Finding optimal threshold on validation set...")
        best_threshold = self.train_model.find_best_threshold(split='validation', metric='f1')
        print(f"  Best F1 threshold: {best_threshold:.3f}")
        
        # Evaluate model
        print("\n📊 Model Evaluation (honest - train-only model):")
        honest_results = self.train_model.summarize_performance(
            threshold=best_threshold,
            splits=('validation', 'test')
        )
        print(honest_results.to_string(index=False))
        self._save_artifacts(suffix="train_only", threshold=best_threshold)

        
        # Also show results at 0.5 threshold
        print(f"\n📊 Model Evaluation at 0.5 threshold:")
        standard_results = self.train_model.summarize_performance(
            threshold=0.5,
            splits=('validation', 'test')
        )
        print(standard_results.to_string(index=False))
        
        # Train production model (train + validation)
        print("\n🏭 Refitting model on TRAIN+VALIDATION for production...")
        self.train_model.refit_on_train_valid(class_weight=class_weight)
        
        # Final test evaluation
        print("\n📈 Final Test Evaluation (production model):")
        final_results = self.train_model.summarize_performance(
            threshold=best_threshold,
            splits=('test',)
        )
        print(final_results.to_string(index=False))

        
        # Save model
        self._save_artifacts(suffix="train_valid", threshold=best_threshold)

        #self.train_model.persist(str(ARTIFACTS_DIR))

        
        # Store results
        self.results = {
            'mode': 'basic',
            'best_threshold': best_threshold,
            'honest_results': honest_results,
            'final_results': final_results,
            'hyperparameters': {
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'class_weight': class_weight
            }
        }
        
        print(f"\n✅ Basic training complete!")
        print(f"  Model saved to: {ARTIFACTS_DIR}")
        print(f"  Best threshold: {best_threshold:.3f}")
        test_f1 = final_results.iloc[0]['f1']
        print(f"  Test F1: {test_f1:.3f}")
        
        return self.results
    
    def run_hyperparameter_tuning(self, strategy='coarse', validation_method='static', 
                                 primary_metric='roc_auc'):
        """Run hyperparameter tuning"""
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        
        if self.train_model is None:
            self.prepare_model()
        
        # print(f"Tuning configuration:")
        # print(f"  Strategy: {strategy}")
        # print(f"  Validation: {validation_method}")
        # print(f"  Primary metric: {primary_metric}")
        
        print(f"Tuning configuration:\n  Strategy: {strategy}\n  Validation: {validation_method}\n  Metric: {primary_metric}")

        # Initialize tuner
        #tuner = RFHyperparameterTuner(self.train_model)

        tuner = RFHyperparameterTuner(
        self.train_model,
        results_dir=str(PROJECT_ROOT / "tuning_results")
        )

        
        # Run tuning
        tuning_results = tuner.run_full_tuning(
            strategy=strategy,
            primary_metric=primary_metric,
            validation_method=validation_method
        )
        
        # Save best model
        model_path = tuner.save_best_model(str(ARTIFACTS_DIR))
        #self._save_artifacts(suffix="tuned_best", threshold=None)

        
        # Store results
        self.results = {
            'mode': 'tuning',
            'strategy': strategy,
            'validation_method': validation_method,
            'primary_metric': primary_metric,
            'best_params': tuning_results['best_params'],
            'best_score': tuning_results['best_score'],
            'test_metrics': tuning_results['test_metrics'],
            'all_results': tuning_results['all_results']
        }
        
        print(f"\n✅ Hyperparameter tuning complete!")
        print(f"  Best {primary_metric}: {tuning_results['best_score']:.4f}")
        print(f"  Best params: {tuning_results['best_params']}")
        print(f"  Model saved to: {model_path}")
        
        return self.results
    
    def evaluate_existing_model(self):
        """Load and evaluate existing model"""
        print("\n" + "=" * 60)
        print("EVALUATING EXISTING MODEL")
        print("=" * 60)
        
        # Check if model exists
        model_files = list(ARTIFACTS_DIR.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {ARTIFACTS_DIR}")
        
        if self.train_model is None:
            self.prepare_model()
        
        # Load model
        self.train_model.load(str(ARTIFACTS_DIR))
        print("✓ Model loaded successfully")
        
        # Find optimal threshold
        best_threshold = self.train_model.find_best_threshold(split='validation', metric='f1')
        print(f"  Best F1 threshold: {best_threshold:.3f}")
        
        # Evaluate on all splits
        results = self.train_model.summarize_performance(
            threshold=best_threshold,
            splits=('train', 'validation', 'test')
        )
        
        print("\n📊 Model Performance:")
        print(results.to_string(index=False))
        
        # Calculate strategy performance
        self.train_model.make_inference("model_probs")
        topk_stats = self.train_model.daily_topk_stats(k=5, split="test", pred_col="model_probs")
        
        print(f"\n📈 Strategy Performance (Top-5 daily):")
        print(f"  Baseline hit rate: {topk_stats['daily_hitrate_baseline']:.1%}")
        print(f"  Strategy hit rate: {topk_stats['daily_hitrate_topk']:.1%}")
        improvement = topk_stats['daily_hitrate_topk'] - topk_stats['daily_hitrate_baseline']
        print(f"  Improvement: {improvement:.1%}")
        
        self.results = {
            'mode': 'evaluate',
            'threshold': best_threshold,
            'performance': results,
            'strategy_stats': topk_stats
        }
        
        return self.results
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        if not self.results:
            print("No results to report!")
            return
        
        print("\n" + "=" * 80)
        print("TRAINING REPORT")
        print("=" * 80)
        
        mode = self.results.get('mode', 'unknown')
        
        if mode == 'basic':
            params = self.results['hyperparameters']
            final = self.results['final_results'].iloc[0]
            
            print(f"🎯 BASIC TRAINING RESULTS")
            print(f"  Model: Random Forest")
            print(f"  Hyperparameters: max_depth={params['max_depth']}, n_estimators={params['n_estimators']}")
            print(f"  Best threshold: {self.results['best_threshold']:.3f}")
            print(f"  Test F1: {final['f1']:.3f}")
            print(f"  Test ROC-AUC: {final['roc_auc']:.3f}")
            print(f"  Test Precision: {final['precision']:.3f}")
            print(f"  Test Recall: {final['recall']:.3f}")
            
        elif mode == 'tuning':
            print(f"🔧 HYPERPARAMETER TUNING RESULTS")
            print(f"  Strategy: {self.results['strategy']}")
            print(f"  Validation: {self.results['validation_method']}")
            print(f"  Best {self.results['primary_metric']}: {self.results['best_score']:.4f}")
            print(f"  Best parameters: {self.results['best_params']}")
            
            if self.results['test_metrics']:
                test = self.results['test_metrics']
                print(f"  Test performance:")
                for metric, value in test.items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.3f}")
            
        elif mode == 'evaluate':
            test_perf = self.results['performance']
            test_row = test_perf[test_perf['split'] == 'test'].iloc[0]
            
            print(f"📊 MODEL EVALUATION RESULTS")
            print(f"  Threshold: {self.results['threshold']:.3f}")
            print(f"  Test F1: {test_row['f1']:.3f}")
            print(f"  Test ROC-AUC: {test_row['roc_auc']:.3f}")
            print(f"  Test Precision: {test_row['precision']:.3f}")
            print(f"  Test Recall: {test_row['recall']:.3f}")
            
            stats = self.results['strategy_stats']
            improvement = stats['daily_hitrate_topk'] - stats['daily_hitrate_baseline']
            print(f"  Strategy improvement: {improvement:.1%}")
        
        print(f"\n💾 Model artifacts saved to: {ARTIFACTS_DIR}")
        print(f"📊 Ready for predictions: python run_predictions.py")

    def _save_artifacts(self, suffix: str, threshold: float | None = None) -> tuple[str, str]:
        """
        Save the current model with a suffix (e.g., 'train_only', 'train_valid', 'tuned_best')
        plus a timestamp. Also writes a meta JSON with feature order, target column, and params.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"random_forest_{suffix}_{ts}.joblib"
        meta_name = f"rf_meta_{suffix}_{ts}.json"

        model_path = ARTIFACTS_DIR / model_name
        meta_path = ARTIFACTS_DIR / meta_name

        # Save the sklearn estimator
        joblib.dump(self.train_model.model, model_path)

        # Try to resolve the exact training feature order
        if hasattr(self.train_model.model, "feature_names_in_"):
            feat_cols = list(self.train_model.model.feature_names_in_)
        else:
            feat_cols = list(getattr(self.train_model, "_inference_feature_columns", self.train_model.X_train.columns))

        # Build metadata
        meta = {
            "saved_at": ts,
            "suffix": suffix,
            "target_col": getattr(self.train_model, "target_col", None),
            "inference_feature_columns": feat_cols,
            "best_threshold": float(threshold) if threshold is not None else None,
            "model_params": getattr(self.train_model.model, "get_params", lambda: {})(),
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved model to {model_path}")
        print(f"Saved meta  to {meta_path}")
        return str(model_path), str(meta_path)



def main():
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    parser.add_argument("--mode", choices=["basic", "tune", "evaluate"], default="basic",
                       help="Training mode: basic (default params), tune (hyperparameter tuning), evaluate (existing model)")
    parser.add_argument("--data-file", type=str,
                       help="Specific data file to use (otherwise uses latest)")
    parser.add_argument("--start-date", type=str,
                       help="Start date for data filtering (YYYY-MM-DD)")
    
    # Basic training parameters
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Max depth for basic training")
    parser.add_argument("--n-estimators", type=int, default=100,
                       help="Number of estimators for basic training")
    parser.add_argument("--class-weight", type=str, choices=["None", "balanced"], default="None",
                       help="Class weight for basic training")
    
    # Tuning parameters
    parser.add_argument("--tune-strategy", choices=["coarse", "fine", "progressive", "random"], default="coarse",
                       help="Hyperparameter tuning strategy")
    parser.add_argument("--tune-validation", choices=["static", "walk_forward"], default="static",
                       help="Validation method for tuning")
    parser.add_argument("--tune-metric", choices=["f1", "precision", "recall", "roc_auc"], default="roc_auc",
                       help="Primary metric for tuning")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ModelTrainingRunner(data_file=args.data_file)
    
    try:
        # Run based on mode
        if args.mode == "basic":
            runner.run_basic_training(
                max_depth=args.max_depth,
                n_estimators=args.n_estimators,
                class_weight=args.class_weight
            )
        elif args.mode == "tune":
            runner.run_hyperparameter_tuning(
                strategy=args.tune_strategy,
                validation_method=args.tune_validation,
                primary_metric=args.tune_metric
            )
        elif args.mode == "evaluate":
            runner.evaluate_existing_model()
        
        # Generate report
        runner.generate_training_report()
        
        print(f"\n🎉 Model training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Model training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()