# integration_examples.py - Simple integration for hyperparameter tuning

import pandas as pd
from pathlib import Path

# OPTION A: Load specific parquet file directly
def option_a_specific_file(filepath: str):
    """
    Load a specific parquet file and run hyperparameter tuning
    
    Parameters:
    -----------
    filepath : str
        Path to your specific parquet file
    """
    from hyperparameter_tuning import RFHyperparameterTuner
    from train_model_new import TrainModel
    
    print(f"📁 OPTION A: Loading specific file: {filepath}")
    
    # Load the file
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = pd.read_parquet(filepath)
    print(f"✅ Loaded data: {data.shape[0]:,} rows, {data.shape[1]:,} columns")
    
    # Simple adapter
    class DataFrameAdapter:
        def __init__(self, df): 
            self.transformed_df = df
    
    # Prepare TrainModel
    tm = TrainModel(DataFrameAdapter(data))
    tm.prepare_dataframe(start_date="2010-01-01")
    
    print(f"Data prepared - Train: {len(tm.X_train):,}, Valid: {len(tm.X_valid):,}, Test: {len(tm.X_test):,}")
    
    # Run tuning
    tuner = RFHyperparameterTuner(tm)
    results = tuner.run_full_tuning(
        strategy='progressive',
        primary_metric='f1',
        validation_method='walk_forward'
    )
    
    # Save model to artifacts directory
    model_path = tuner.save_best_model("artifacts")
    print(f"✅ Option A complete! Model saved to: {model_path}")
    
    return results


# OPTION B: Custom prepare_dataframe parameters
def option_b_custom_params():
    """
    Load most recent data file and run tuning with custom prepare_dataframe parameters
    """
    from hyperparameter_tuning import RFHyperparameterTuner
    from train_model_new import TrainModel
    
    print("⚙️ OPTION B: Loading most recent file with custom parameters")
    
    # Find most recent parquet file
    data_dir = Path("../data")
    patterns = ["stock_data_combined_*.parquet", "stock_data_complete_*.parquet", "stock_data_batch_*.parquet"]
    
    latest_file = None
    latest_time = 0
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        for file in files:
            if file.stat().st_mtime > latest_time:
                latest_time = file.stat().st_mtime
                latest_file = file
    
    if latest_file is None:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    # Load data
    data = pd.read_parquet(latest_file)
    print(f"✅ Loaded most recent file: {latest_file}")
    print(f"   Data: {data.shape[0]:,} rows, {data.shape[1]:,} columns")
    
    # Simple adapter
    class DataFrameAdapter:
        def __init__(self, df): 
            self.transformed_df = df
    
    # Prepare TrainModel with CUSTOM PARAMETERS
    tm = TrainModel(DataFrameAdapter(data))
    
    # EDIT THESE PARAMETERS AS NEEDED:
    custom_params = {
        'start_date': "2015-01-01",    # Only use data from 2015 onwards
        'end_date': None               # Use all data until end (or set a specific date)
    }
    
    print(f"Preparing data with custom parameters: {custom_params}")
    tm.prepare_dataframe(**custom_params)
    
    print(f"Data prepared - Train: {len(tm.X_train):,}, Valid: {len(tm.X_valid):,}, Test: {len(tm.X_test):,}")
    
    # Run tuning
    tuner = RFHyperparameterTuner(tm)
    results = tuner.run_full_tuning(
        strategy='progressive',
        primary_metric='f1',
        validation_method='walk_forward'
    )
    
    # Save model to artifacts directory
    model_path = tuner.save_best_model("artifacts")
    print(f"✅ Option B complete! Model saved to: {model_path}")
    
    return results


def main():
    """
    Main function to choose between Option A, Option B, or Quick Mode
    """
    print("🔧 Hyperparameter Tuning - Choose Option")
    print("=" * 50)
    print("A: Load specific parquet file")
    print("B: Load most recent file with custom prepare_dataframe parameters") 
    print("Q: Quick tuning mode (fast, basic optimization)")
    
    choice = input("\nEnter choice (A, B, or Q): ").strip().upper()
    
    if choice == "A":
        # EDIT THIS FILE PATH FOR YOUR SPECIFIC FILE:
        filepath = "../data/stock_data_combined_20241215_143022.parquet"  # CHANGE THIS PATH
        
        print(f"Using file path: {filepath}")
        confirm = input("Correct? (y/n): ").strip().lower()
        
        if confirm != 'y':
            filepath = input("Enter your file path: ").strip()
        
        return option_a_specific_file(filepath)
        
    elif choice == "B":
        return option_b_custom_params()
        
    elif choice == "Q":
        return quick_tuning_mode()
        
    else:
        print("Invalid choice. Please enter A, B, or Q")
        return main()


def quick_tuning_mode():
    """
    QUICK MODE: More parameters with static validation for speed
    Uses static validation but broader parameter exploration
    """
    from hyperparameter_tuning import RFHyperparameterTuner
    from train_model_new import TrainModel
    
    print("⚡ QUICK TUNING MODE: Broad parameter search with static validation")
    print("=" * 50)
    
    # Find most recent parquet file
    data_dir = Path("./data")
    patterns = ["stock_data_combined_*.parquet", "stock_data_complete_*.parquet"]
    
    latest_file = None
    latest_time = 0
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        for file in files:
            if file.stat().st_mtime > latest_time:
                latest_time = file.stat().st_mtime
                latest_file = file
    
    if latest_file is None:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    # Load data
    data = pd.read_parquet(latest_file)
    print(f"✅ Using: {latest_file.name}")
    print(f"   Data: {data.shape[0]:,} rows, {data.shape[1]:,} columns")
    
    # Simple adapter
    class DataFrameAdapter:
        def __init__(self, df): 
            self.transformed_df = df
    
    # Prepare with recent data for speed
    tm = TrainModel(DataFrameAdapter(data))
    tm.prepare_dataframe(start_date="2018-01-01")  # More data than before but still manageable
    
    print(f"📊 Quick data prep - Train: {len(tm.X_train):,}, Valid: {len(tm.X_valid):,}, Test: {len(tm.X_test):,}")
    
    # Initialize tuner
    tuner = RFHyperparameterTuner(tm)
    
    # EXPANDED parameter grid for better exploration (but still static validation)
    expanded_quick_grid = {
        'n_estimators': [50, 100, 150, 200, 300],        # 5 values
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 17, 20, 25, None],  # 15 values (1-10 + existing)
        'min_samples_split': [2, 3, 5, 7, 10],          # 5 values
        'min_samples_leaf': [1, 2, 3, 4],               # 4 values
        'max_features': ['sqrt', 'log2', 0.3, 0.5],     # 4 values
        'class_weight': [None, 'balanced']               # 2 values
    }
    
    total_combinations = 5 * 15 * 5 * 4 * 4 * 2
    print(f"🚀 Expanded grid: {total_combinations:,} combinations")
    print(f"   Max depth range: 1-10 + [15, 17, 20, 25, None] for comprehensive exploration")
    print(f"   Estimated time: 15-25 minutes (static validation)")
    print(f"   Will test shallow (1-10) and deep (15-25) trees plus unlimited depth")
    
    # Run expanded tuning with static validation for speed
    results_df = tuner.grid_search(
        param_grid=expanded_quick_grid,
        primary_metric='f1',
        validation_method='static',  # Keep static for speed
        max_combinations=1000        # Sample if too many combinations
    )
    
    # Save model
    model_path = tuner.save_best_model()
    
    print(f"⚡ Expanded quick tuning complete!")
    print(f"   Best F1: {tuner.best_score:.4f}")
    print(f"   Tested {len(results_df)} parameter combinations")
    print(f"   Best params: {tuner.best_params}")
    print(f"   Model saved to: {model_path}")
    print(f"   Total time: ~15-25 minutes")
    
    return {
        'best_params': tuner.best_params,
        'best_score': tuner.best_score,
        'results': results_df,
        'mode': 'expanded_quick'
    }


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎯 Final Results:")
        print(f"   Best F1 Score: {results['best_score']:.4f}")
        print(f"   Strategy: {results['strategy']}")
        print(f"   Validation: {results['validation_method']}")