# hyperparameter_tuning.py - Clean RandomForest Hyperparameter Tuning

import os
import sys
import json
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from pathlib import Path

# Ensure local imports resolve
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

# Project structure setup
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

from train_model_new import TrainModel


warnings.filterwarnings('ignore')

class RFHyperparameterTuner:
    """
    Hyperparameter tuning for RandomForest models with walk-forward validation support
    """
    
    def __init__(self, train_model: TrainModel, results_dir: str = None):
        self.tm = train_model
        
        # Use project-relative paths if no custom directory specified
        if results_dir is None:
            self.results_dir = PROJECT_ROOT / "tuning_results"
        else:
            self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        if not hasattr(self.tm, 'X_train') or self.tm.X_train is None:
            raise ValueError("TrainModel must have prepared data splits. Call prepare_dataframe() first.")
        
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
        print(f"Tuner initialized:")
        print(f"  Train: {len(self.tm.X_train):,}, Valid: {len(self.tm.X_valid):,}, Test: {len(self.tm.X_test):,}")
        print(f"  Features: {len(self.tm.X_train.columns):,}, Target: {self.tm.target_col}")
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """Define parameter grids for different tuning strategies"""
        return {
            'coarse': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced']
            },
            'fine': {
                'n_estimators': [150, 200, 250, 300],
                'max_depth': [12, 15, 17, 20, 25],
                'min_samples_split': [2, 3, 5, 7],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}]
            },
            'random': {
                'n_estimators': [50, 75, 100, 150, 200, 250, 300, 400, 500],
                'max_depth': [5, 8, 10, 12, 15, 17, 20, 25, 30, None],
                'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
                'min_samples_leaf': [1, 2, 3, 4, 5, 8],
                'max_features': ['sqrt', 'log2', 0.2, 0.3, 0.5, 0.7, None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced', {0: 1, 1: 1.5}, {0: 1, 1: 2}, {0: 1, 1: 3}]
            }
        }
    
    def evaluate_params(self, params: Dict[str, Any], 
                       validation_method: str = 'static',
                       eval_set: str = 'validation') -> Dict[str, float]:
        """Evaluate parameter combination with static or walk-forward validation"""
        if validation_method == 'static':
            return self._evaluate_static(params, eval_set)
        elif validation_method == 'walk_forward':
            return self._evaluate_walk_forward(params)
        else:
            raise ValueError("validation_method must be 'static' or 'walk_forward'")
    
    def _evaluate_static(self, params: Dict[str, Any], eval_set: str = 'validation') -> Dict[str, float]:
        """Static validation - traditional train/validation split"""
        try:
            rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
            rf.fit(self.tm.X_train, self.tm.y_train)
            
            if eval_set == 'validation':
                X_eval, y_eval = self.tm.X_valid, self.tm.y_valid
            elif eval_set == 'test':
                X_eval, y_eval = self.tm.X_test, self.tm.y_test
            else:
                raise ValueError("eval_set must be 'validation' or 'test'")
            
            y_pred_proba = rf.predict_proba(X_eval)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            return {
                'accuracy': accuracy_score(y_eval, y_pred),
                'precision': precision_score(y_eval, y_pred, zero_division=0),
                'recall': recall_score(y_eval, y_pred, zero_division=0),
                'f1': f1_score(y_eval, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_eval, y_pred_proba) if len(np.unique(y_eval)) > 1 else 0.5,
                'pr_auc': average_precision_score(y_eval, y_pred_proba) if y_eval.sum() > 0 else 0.0,
                'pos_rate': float(y_pred.mean()),
                'base_pos_rate': float(y_eval.mean())
            }
            
        except Exception as e:
            print(f"Error evaluating params: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0.5, 'pr_auc': 0, 'pos_rate': 0, 'base_pos_rate': 0}
    
    def _evaluate_walk_forward(self, params: Dict[str, Any], 
                              min_train_years: float = 2.0,
                              step_months: int = 3) -> Dict[str, float]:
        """Walk-forward validation for time series data"""
        try:
            if 'Date' not in self.tm.df_full.columns:
                raise ValueError("Date column required for walk-forward validation")
            
            train_valid_data = self.tm.df_full[
                self.tm.df_full['split'].isin(['train', 'validation'])
            ].copy().sort_values('Date')
            
            if len(train_valid_data) == 0:
                raise ValueError("No train+validation data found")
            
            start_date = train_valid_data['Date'].min()
            end_date = train_valid_data['Date'].max()
            first_pred_date = start_date + pd.DateOffset(years=min_train_years)
            
            current_date = first_pred_date
            step_offset = pd.DateOffset(months=step_months)
            
            all_predictions = []
            all_actuals = []
            fold_metrics = []
            fold_num = 0
            
            while current_date + step_offset <= end_date:
                fold_num += 1
                
                train_mask = train_valid_data['Date'] < current_date
                train_data = train_valid_data[train_mask]
                
                test_start = current_date
                test_end = current_date + step_offset
                test_mask = (train_valid_data['Date'] >= test_start) & (train_valid_data['Date'] < test_end)
                test_data = train_valid_data[test_mask]
                
                if len(train_data) < 100 or len(test_data) < 10:
                    current_date += step_offset
                    continue
                
                feature_cols = self.tm.X_train.columns.tolist()
                available_features = [col for col in feature_cols if col in train_data.columns]
                
                X_train_fold = train_data[available_features].fillna(0)
                y_train_fold = train_data[self.tm.target_col]
                X_test_fold = test_data[available_features].fillna(0)
                y_test_fold = test_data[self.tm.target_col]
                
                X_train_fold = X_train_fold.select_dtypes(include=[np.number])
                X_test_fold = X_test_fold.select_dtypes(include=[np.number])
                
                common_cols = X_train_fold.columns.intersection(X_test_fold.columns)
                if len(common_cols) == 0:
                    current_date += step_offset
                    continue
                
                X_train_fold = X_train_fold[common_cols]
                X_test_fold = X_test_fold[common_cols]
                
                rf_fold = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
                rf_fold.fit(X_train_fold, y_train_fold)
                
                y_pred_proba_fold = rf_fold.predict_proba(X_test_fold)[:, 1]
                y_pred_fold = (y_pred_proba_fold >= 0.5).astype(int)
                
                all_predictions.extend(y_pred_proba_fold)
                all_actuals.extend(y_test_fold.values)
                
                fold_f1 = f1_score(y_test_fold, y_pred_fold, zero_division=0)
                fold_precision = precision_score(y_test_fold, y_pred_fold, zero_division=0)
                fold_recall = recall_score(y_test_fold, y_pred_fold, zero_division=0)
                fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
                
                fold_metrics.append({
                    'fold': fold_num, 'f1': fold_f1, 'precision': fold_precision,
                    'recall': fold_recall, 'accuracy': fold_accuracy
                })
                
                current_date += step_offset
            
            if len(all_predictions) == 0:
                raise ValueError("No predictions generated in walk-forward validation")
            
            all_predictions = np.array(all_predictions)
            all_actuals = np.array(all_actuals)
            all_pred_binary = (all_predictions >= 0.5).astype(int)
            
            overall_metrics = {
                'accuracy': accuracy_score(all_actuals, all_pred_binary),
                'precision': precision_score(all_actuals, all_pred_binary, zero_division=0),
                'recall': recall_score(all_actuals, all_pred_binary, zero_division=0),
                'f1': f1_score(all_actuals, all_pred_binary, zero_division=0),
                'roc_auc': roc_auc_score(all_actuals, all_predictions) if len(np.unique(all_actuals)) > 1 else 0.5,
                'pr_auc': average_precision_score(all_actuals, all_predictions) if all_actuals.sum() > 0 else 0.0,
                'pos_rate': float(all_pred_binary.mean()),
                'base_pos_rate': float(all_actuals.mean()),
                'n_folds': len(fold_metrics),
                'total_test_samples': len(all_predictions)
            }
            
            if fold_metrics:
                fold_df = pd.DataFrame(fold_metrics)
                overall_metrics.update({
                    'f1_std': float(fold_df['f1'].std()),
                    'precision_std': float(fold_df['precision'].std()),
                    'recall_std': float(fold_df['recall'].std()),
                    'accuracy_std': float(fold_df['accuracy'].std()),
                })
            
            return overall_metrics
            
        except Exception as e:
            print(f"Error in walk-forward validation: {e}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0.5, 'pr_auc': 0, 'pos_rate': 0, 'base_pos_rate': 0, 'n_folds': 0}
    
    def grid_search(self, param_grid: Dict[str, List], 
                   max_combinations: Optional[int] = None,
                   primary_metric: str = 'f1',
                   validation_method: str = 'static') -> pd.DataFrame:
        """Grid search with progress tracking"""
        print(f"Grid search with {validation_method} validation...")
        
        param_combinations = list(ParameterGrid(param_grid))
        if max_combinations and len(param_combinations) > max_combinations:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_combinations]
        
        print(f"Testing {len(param_combinations)} combinations")
        
        results = []
        for i, params in enumerate(param_combinations, 1):
            if i % 10 == 0 or i == len(param_combinations):
                print(f"  Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%)")
            
            metrics = self.evaluate_params(params, validation_method=validation_method)
            
            result = {
                'combination_id': i, 'params': params.copy(), 'validation_method': validation_method,
                **metrics, 'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            if metrics[primary_metric] > self.best_score:
                self.best_score = metrics[primary_metric]
                self.best_params = params.copy()
                extra_info = f" (across {metrics['n_folds']} folds)" if validation_method == 'walk_forward' and 'n_folds' in metrics else ""
                print(f"    New best {primary_metric}: {self.best_score:.4f}{extra_info}")
        
        results_df = pd.DataFrame(results).sort_values(primary_metric, ascending=False)
        
        # Add flattened parameter columns
        param_cols = {}
        for param_name in param_grid.keys():
            param_cols[f'param_{param_name}'] = results_df['params'].apply(lambda x: x.get(param_name, None))
        results_df = pd.concat([results_df, pd.DataFrame(param_cols)], axis=1)
        
        print(f"Grid search complete. Best {primary_metric}: {self.best_score:.4f}")
        return results_df
    
    def run_full_tuning(self, strategy: str = 'progressive', 
                       primary_metric: str = 'f1',
                       validation_method: str = 'static') -> Dict[str, Any]:
        """Run complete tuning workflow"""
        print(f"🚀 Starting tuning: {strategy} strategy, {validation_method} validation, {primary_metric} metric")
        
        param_grids = self.get_param_grids()
        
        if strategy == 'progressive':
            # Multi-stage tuning
            coarse_results = self.grid_search(param_grids['coarse'], primary_metric=primary_metric, validation_method='static')
            fine_results = self.grid_search(param_grids['fine'], max_combinations=50 if validation_method == 'walk_forward' else 100, primary_metric=primary_metric, validation_method=validation_method)
            combined_results = pd.concat([coarse_results, fine_results], ignore_index=True)
        elif strategy in param_grids:
            max_comb = 50 if validation_method == 'walk_forward' else None
            combined_results = self.grid_search(param_grids[strategy], max_combinations=max_comb, primary_metric=primary_metric, validation_method=validation_method)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        combined_results = combined_results.sort_values(primary_metric, ascending=False)
        
        # Test on holdout test set
        test_metrics = None
        if self.best_params:
            print(f"\n🧪 Testing best model on holdout test set...")
            test_metrics = self.evaluate_params(self.best_params, validation_method='static', eval_set='test')
            print(f"Test {primary_metric}: {test_metrics.get(primary_metric, 0):.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"tuning_results_{timestamp}.csv"
        combined_results.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'test_metrics': test_metrics,
            'all_results': combined_results,
            'strategy': strategy,
            'validation_method': validation_method,
            'primary_metric': primary_metric
        }
    
    def save_best_model(self, save_dir: str = None) -> str:
        """Train and save the best model"""
        if not self.best_params:
            raise ValueError("No best parameters found. Run tuning first.")
        
        # Use project-relative artifacts directory if no custom directory specified
        if save_dir is None:
            save_path = ARTIFACTS_DIR
        else:
            save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **self.best_params)
        best_rf.fit(self.tm.X_train_valid, self.tm.y_train_valid)
        
        model_file = save_path / "best_rf_model.joblib"
        meta_file = save_path / "model_metadata.json"
        
        import joblib
        joblib.dump(best_rf, model_file)
        
        metadata = {
            'best_params': self.best_params,
            'best_validation_score': float(self.best_score),
            'feature_columns': list(self.tm.X_train_valid.columns),
            'target_column': self.tm.target_col,
            'training_samples': int(len(self.tm.X_train_valid)),
            'tuning_timestamp': datetime.now().isoformat(),
            'feature_importances': best_rf.feature_importances_.tolist()
        }
        
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Best model saved to: {model_file}")
        return str(save_path)


# Simple usage example
def example_usage():
    """
    Example of how to use the tuner - replace with your actual data loading
    """
    print("📋 Example Usage (replace with your data loading):")
    print("=" * 50)
    
    # Your data loading code here - this is just a placeholder
    try:
        # Example: Load your data
        import pandas as pd
        
        # This is just an example - replace with your actual data loading:
        # data = pd.read_parquet("../data/your_data.parquet")
        # 
        # class DataAdapter:
        #     def __init__(self, df): self.transformed_df = df
        # 
        # tm = TrainModel(DataAdapter(data))
        # tm.prepare_dataframe(start_date="2015-01-01")
        # 
        # tuner = RFHyperparameterTuner(tm)
        # results = tuner.run_full_tuning(
        #     strategy='progressive',
        #     primary_metric='f1',
        #     validation_method='walk_forward'
        # )
        # tuner.save_best_model("artifacts")
        
        print("Replace this example with your actual data loading code.")
        print("Use integration_examples.py to run with your data.")
        
    except Exception as e:
        print(f"This is just an example. Error: {e}")
        print("\nTo use this tuner:")
        print("1. Load your data into a DataFrame")  
        print("2. Create TrainModel instance")
        print("3. Call prepare_dataframe()")
        print("4. Initialize RFHyperparameterTuner")
        print("5. Run tuning with run_full_tuning()")


if __name__ == "__main__":
    example_usage()