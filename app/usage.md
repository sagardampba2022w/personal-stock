# 1. Extract data (small test)
python run_data_extraction.py --mode small --num-tickers 20 --validate

# 2. Train model with default parameters
python run_model_training.py --mode basic

# 3. Generate predictions and recommendations
python run_predictions.py --mode recommend


📊 1. Data Extraction (run_data_extraction.py)

### Small Scale Testing (10-50 tickers)

# Test with 10 tickers
python run_data_extraction.py --mode small --num-tickers 10 --validate

# Test with custom ticker list
python run_data_extraction.py --mode custom --tickers AAPL GOOGL MSFT TSLA NVDA --validate


Production Scale (500+ tickers)
# Full pipeline with batching
python run_data_extraction.py --mode batch --max-tickers 500 --batch-size 50 --validate

# Smaller batch for testing batch logic
python run_data_extraction.py --mode batch --max-tickers 100 --batch-size 25


Available Options

--mode: small, batch, custom
--num-tickers: Number of tickers for small mode (default: 10)
--max-tickers: Maximum tickers for batch mode (default: 500)
--batch-size: Batch size for processing (default: 50)
--tickers: Custom ticker list for custom mode
--validate: Run data validation after extraction

Expected Output

Parquet files saved to data/ directory
Names like stock_data_combined_20241215_143022.parquet
Validation report showing feature categories and data quality


🤖 2. Model Training (run_model_training.py)
Basic Training (Recommended Start)

# Train with default hyperparameters
python run_model_training.py --mode basic

# Train with custom parameters
python run_model_training.py --mode basic --max-depth 20 --n-estimators 300 --class-weight balanced

# Use specific data file
python run_model_training.py --mode basic --data-file data/my_custom_data.parquet --start-date 2015-01-01


Hyperparameter Tuning
# Progressive tuning (recommended)
python run_model_training.py --mode tune --tune-strategy progressive --tune-validation walk_forward

# Quick tuning with static validation
python run_model_training.py --mode tune --tune-strategy coarse --tune-validation static --tune-metric precision

# Comprehensive random search
python run_model_training.py --mode tune --tune-strategy random --tune-validation walk_forward --tune-metric f1


Model Evaluation

# Evaluate existing model
python run_model_training.py --mode evaluate

# Evaluate with specific data
python run_model_training.py --mode evaluate --data-file data/my_test_data.parquet


Available Options

--mode: basic, tune, evaluate
--data-file: Specific data file (otherwise uses latest)
--start-date: Filter data from this date (YYYY-MM-DD)
--max-depth: Max tree depth for basic training (default: 17)
--n-estimators: Number of trees for basic training (default: 200)
--class-weight: Class weighting (None or balanced)
--tune-strategy: Tuning strategy (coarse, fine, progressive, random)
--tune-validation: Validation method (static, walk_forward)
--tune-metric: Primary optimization metric (f1, precision, recall, roc_auc)

Expected Output

Model files saved to artifacts/ directory
Training report with honest evaluation metrics
Production model ready for predictions

🎯 3. Predictions and Strategy Selection (run_predictions.py)


Generate All Predictions
# Generate all prediction strategies
python run_predictions.py --mode generate

Find Optimal Thresholds
# Find thresholds optimized for different objectives
python run_predictions.py --mode threshold

Compare Strategies

