# Detailed Usage Guide

> **📋 Overview**: Complete step-by-step guide for running the Stock Market Analytics system, from data extraction to live portfolio management.

**📖 Main Documentation**: [README.md](README.md)

## 📊 1. Data Extraction Pipeline

The data extraction pipeline fetches stock market data, applies technical analysis, and integrates macro economic indicators.

### Small Scale Testing (10-50 tickers)

Perfect for initial testing and development:

```bash
# Test with 10 tickers
python app/run_data_extraction.py --mode small --num-tickers 10 --validate

# Test with 20 tickers + validation report
python app/run_data_extraction.py --mode small --num-tickers 20 --validate

# Custom ticker list
python app/run_data_extraction.py --mode custom --tickers AAPL GOOGL MSFT TSLA NVDA --validate
```

### Production Scale (500+ tickers)

For full universe processing with intelligent batching:

```bash
# Full pipeline with automatic batching (recommended)
python app/run_data_extraction.py --mode batch --max-tickers 500 --batch-size 50 --validate

# Smaller batch for testing batch logic
python app/run_data_extraction.py --mode batch --max-tickers 100 --batch-size 25

# Use custom ticker file (from ticker universe)
python app/run_data_extraction.py --mode batch \
  --tickers-file ticker/spx_ndx_liq_top250_latest.csv \
  --batch-size 50
```

### Advanced Options

```bash
# Custom ticker universe from CSV
python app/run_data_extraction.py --mode custom \
  --tickers-file ticker/my_custom_universe.csv \
  --validate

# Batched processing of custom universe
python app/run_data_extraction.py --mode batch \
  --tickers-file ticker/my_custom_universe.csv \
  --max-tickers 200 --batch-size 40
```

**Expected Output:**
- Parquet files saved to `data/` directory
- Naming: `stock_data_combined_20240115_143022.parquet`
- Validation report showing feature categories and data quality

**⏱️ Processing Times:**
- Small (10 tickers): ~2-5 minutes
- Medium (100 tickers): ~15-30 minutes  
- Large (500 tickers): ~2-4 hours (with batching)

## 🤖 2. Model Training & Optimization

Train machine learning models with various configurations and hyperparameter optimization strategies.

### Basic Training (Recommended Start)

Quick model training with sensible defaults:

```bash
# Train with default hyperparameters
python app/run_model_training.py --mode basic

# Train with custom parameters
python app/run_model_training.py --mode basic \
  --max-depth 20 \
  --n-estimators 300 \
  --class-weight balanced

# Use specific data file and date range
python app/run_model_training.py --mode basic \
  --data-file data/my_custom_data.parquet \
  --start-date 2015-01-01
```

**Basic Training Options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-depth` | 17 | Maximum tree depth (None for unlimited) |
| `--n-estimators` | 200 | Number of trees in forest |
| `--class-weight` | None | Class balancing (None or balanced) |
| `--start-date` | auto | Filter data from this date (YYYY-MM-DD) |

### Hyperparameter Tuning

Systematic optimization for maximum performance:

```bash
# Progressive tuning (recommended for production)
python app/run_model_training.py --mode tune \
  --tune-strategy progressive \
  --tune-validation walk_forward \
  --tune-metric f1

# Quick tuning with static validation
python app/run_model_training.py --mode tune \
  --tune-strategy coarse \
  --tune-validation static \
  --tune-metric precision

# Comprehensive random search
python app/run_model_training.py --mode tune \
  --tune-strategy random \
  --tune-validation walk_forward \
  --tune-metric roc_auc

# Fine-tuning only (if you've run coarse first)
python app/run_model_training.py --mode tune \
  --tune-strategy fine \
  --tune-validation static \
  --tune-metric f1
```

**Tuning Strategy Guide:**

| Strategy | Time | Combinations | Best For |
|----------|------|--------------|----------|
| `coarse` | ~10 min | ~50 | Quick exploration |
| `fine` | ~30-60 min | ~200 | Detailed optimization |
| `progressive` | ~40-70 min | ~250 | Best overall performance |
| `random` | ~20-40 min | ~100 | Broad exploration |

**Validation Method Guide:**

| Method | Description | Best For | Speed |
|--------|-------------|----------|-------|
| `static` | Traditional train/valid split | Fast iteration | ⚡ Fast |
| `walk_forward` | Time-series aware validation | Production models | 🐌 Slower |

**Optimization Metrics:**

| Metric | Focus | When to Use |
|--------|--------|-------------|
| `roc_auc` | Overall discrimination | Balanced datasets |
| `f1` | Balanced precision/recall | General purpose |
| `precision` | Minimize false positives | Conservative trading |
| `recall` | Capture all opportunities | Aggressive trading |

### Model Evaluation

Evaluate existing trained models:

```bash
# Evaluate existing model
python app/run_model_training.py --mode evaluate

# Evaluate with specific data
python app/run_model_training.py --mode evaluate \
  --data-file data/my_test_data.parquet
```

**Expected Output:**
- Model files saved to `artifacts/` directory
- Training report with honest evaluation metrics
- Production model ready for predictions

**📁 Model Artifacts:**
```
artifacts/
├── best_rf_model.joblib              # Latest trained model
├── model_metadata.json               # Feature list + parameters
├── best_rf_model_20240115_143022.joblib  # Timestamped archive
└── rf_meta_20240115_143022.json       # Archived metadata
```

## 🎯 3. Prediction Generation & Strategy Selection

Generate multiple prediction strategies and select optimal approaches based on backtesting performance.

### Generate All Predictions

Create comprehensive strategy universe:

```bash
# Generate all prediction strategies
python app/run_predictions.py --mode generate

# Generate with specific data file
python app/run_predictions.py --mode generate \
  --data-file data/my_custom_data.parquet
```

**Strategy Types Created:**
- **7 Manual Rules**: Technical + macro-based strategies
- **5 ML Fixed Thresholds**: 21%, 50%, 65%, 80%, 90% probability cutoffs
- **3 ML Adaptive Thresholds**: Top 1%, 3%, 5% by validation performance
- **3 Daily Top-K**: Top 3, 5, 10 stocks per day
- **3 Ensemble Strategies**: Combined manual + ML approaches

### Find Optimal Thresholds

Optimize probability thresholds for different objectives:

```bash
# Find thresholds optimized for different metrics
python app/run_predictions.py --mode threshold

# Find optimal thresholds for specific metrics
python app/run_predictions.py --mode threshold \
  --validation-metrics precision f1 recall
```

**Threshold Optimization:**
- **Precision-focused**: Minimize false positives
- **F1-focused**: Balance precision and recall
- **Recall-focused**: Capture all opportunities
- **ROC-optimal**: Youden's J-statistic (sensitivity + specificity)

### Compare All Strategies

Comprehensive strategy evaluation and ranking:

```bash
# Compare all strategies on test set (default)
python app/run_predictions.py --mode compare

# Compare on validation set
python app/run_predictions.py --mode compare --split validation

# Save prediction probabilities for analysis
python app/run_predictions.py --mode compare --save-predictions
```

### Advanced Analysis

Deep dive into prediction patterns and correlations:

```bash
# Full analysis pipeline
python app/run_predictions.py --mode analyze --split test

# Generate final investment recommendations
python app/run_predictions.py --mode recommend
```

**Expected Strategy Output:**

```
TOP 10 STRATEGIES (ranked by PRECISION):

Strategy                          | Precision | F1    | Recall | Daily Improvement
pred15_rf_auto_rate_1p            | 61.3%     | 0.245 | 15.2%  | +8.7%
pred22_ens_manual2plus_and_auto3p | 58.9%     | 0.189 | 12.1%  | +7.2%
pred11_rf_thresh_50              | 52.4%     | 0.312 | 24.8%  | +5.1%
```

**Strategy Categories:**

| Type | Naming | Description |
|------|--------|-------------|
| Manual | `pred0-6_manual_*` | Rule-based technical/macro |
| ML Fixed | `pred10-14_rf_thresh_*` | Fixed probability thresholds |
| ML Adaptive | `pred15-17_rf_auto_rate_*` | Validation-optimized rates |
| Top-K | `pred30_top*_daily` | Daily ranking-based |
| Ensemble | `pred20-22_ens_*` | Combined approaches |

## 🎮 4. Trading Simulation & Backtesting

Run realistic trading simulations with transaction costs and capital requirements.

### Basic Simulation

```bash
# Run simulation with default settings
python app/run_simulations.py

# Use specific model file
python app/run_simulations.py \
  --model-file artifacts/my_specific_model.joblib

# Use custom artifacts directory
python app/run_simulations.py \
  --artifacts-dir /path/to/custom/artifacts \
  --results-dir /path/to/results
```

### Simulation Parameters

**Default Settings:**
- **Investment per trade**: $100
- **Transaction costs**: 0.2% round-trip (buy + sell)
- **Hold period**: 30 days (matches prediction horizon)
- **Capital calculation**: 75th percentile of concurrent positions

**Performance Metrics:**
- **Net P&L**: Profit/loss after transaction costs
- **Efficiency Score**: `net_pnl / (capital_required × (1 + |max_drawdown|))`
- **Max Drawdown**: Largest peak-to-trough decline
- **CAGR**: Compound annual growth rate

**Example Results:**

```
TOP 10 (by net_pnl):
Strategy                     | Trades | Net P&L | Efficiency | Max DD
pred15_rf_auto_rate_1p      | 45     | $1,247  | 0.089     | -8.2%
pred22_ens_manual2plus_...  | 28     | $894    | 0.076     | -5.1%

TOP 10 (by efficiency_score):
Strategy                     | Trades | Efficiency | Capital Req | Net P&L
pred30_top3_daily           | 156    | 0.124     | $8,400     | $1,567
pred15_rf_auto_rate_1p      | 45     | 0.089     | $4,500     | $1,247
```

## 📱 5. Portfolio Management & Live Trading

Integrate with Interactive Brokers for live portfolio tracking and automated management.

### Interactive Brokers Setup

**1. Configure Flex Queries in IB:**
- Create Trade Confirmation query (get `QUERY_ID`)
- Create Open Positions query (get `QUERY_ID_POSITIONS`)
- Generate Flex Token

**2. Set Environment Variables:**
```bash
# .env file
IBKR_FLEX_TOKEN=your_flex_token_here
IBKR_FLEX_QUERY_ID=your_trade_query_id
IBKR_FLEX_QUERY_POSITION_ID=your_position_query_id
```

### Real-Time Data Collection

**Daily Account Logging:**
```bash
# Log account summary and positions to Google Sheets
python portfolio-manage/log-journal.py
```

**Trade History Sync:**
```bash
# Download and sync trade history from IB Flex Queries
python portfolio-manage/log-trades.py
```

**Manual Portfolio Summary:**
```bash
# Send current portfolio summary via Telegram
python portfolio-manage/send-summary.py
```

### Portfolio Analysis & Alerts

**Daily Processing Pipeline:**
```bash
# 1. Update positions with latest data
python portfolio-manage/portfolio-positions-manage.py

# 2. Send automated daily alerts
python portfolio-manage/portfolio-alerts.py
```

**Google Sheets Integration:**

The system automatically maintains these worksheets:
- `portfolio`: Current positions from IB API
- `accountsummary`: Daily account metrics
- `flex-trades`: Trade history from Flex Queries
- `flex-positions`: Position snapshots
- `updated-positions`: Processed positions with analytics
- `portfolio-summary`: Time series of portfolio metrics

### Automated Monitoring

**Daily Telegram Alerts:**

**Portfolio Summary:**
```
📊 Portfolio Summary (2024-01-15)
• Value: $125,430.50
• Daily P/L: +1.2%
• Cumulative Return: +15.3%
• Drawdown: -2.1%
• 21-day Vol: 18.5%
• Sharpe: 1.85  Sortino: 2.12
```

**Position Analysis:**
```
📑 Position Summary (2024-01-15)
• Cash: 12.5%
• >10% positions: 2
• <1% positions: 8

• Top contributors: AAPL:+$450, NVDA:+$380
• Bottom contributors: META:-$120, TSLA:-$95
```

**Risk Alerts:**
- Concentration warnings (>10% position)
- Under-diversification flags (<1% positions)
- Drawdown notifications
- Volatility spike alerts

## 🔧 6. Advanced Configuration

### Custom Universe Creation

**1. Create ticker universe:**
```bash
# Fetch S&P 500 + NASDAQ 100 with liquidity ranking
python app/tickerfetch.py --liquidity --top 250
```

**2. Use custom universe:**
```bash
# Use the generated universe for data extraction
python app/run_data_extraction.py --mode batch \
  --tickers-file ticker/spx_ndx_liq_top250_latest.csv \
  --batch-size 50
```

### Feature Engineering Customization

**Technical Indicators (`app/technicals.py`):**
- Modify TA-Lib parameters
- Add custom indicators
- Adjust lookback periods

**Macro Features (`app/macros.py`):**
- Add new FRED economic series
- Modify growth rate calculations
- Include additional market indices

**Target Variables (`app/extract.py`):**
- Change prediction horizon (default: 30 days)
- Modify return thresholds (default: 1%)
- Create multi-class targets

### Model Customization

**Random Forest Parameters:**
```python
# In train_model_new.py, modify TrainConfig class
@dataclass
class TrainConfig:
    rf_max_depth: Optional[int] = 20        # Tree depth
    rf_n_estimators: int = 300              # Number of trees
    random_state: int = 42                  # Reproducibility
    n_jobs: int = 8                         # Parallel processing
```

**Alternative Models:**
- Replace RandomForestClassifier with GradientBoostingClassifier
- Add XGBoost or LightGBM implementations
- Implement ensemble methods

## 🚨 7. Troubleshooting

### Common Issues

**1. Data Extraction Failures:**
```bash
# Check internet connection and API limits
# Reduce batch size if getting timeouts
python app/run_data_extraction.py --mode batch --batch-size 25

# Validate specific tickers
python app/run_data_extraction.py --mode custom --tickers AAPL MSFT --validate
```

**2. Model Training Errors:**
```bash
# Check for sufficient data
python app/run_model_training.py --mode basic --start-date 2020-01-01

# Use smaller feature set for debugging
# Modify feature selection in train_model_new.py
```

**3. IB Connection Issues:**
```bash
# Ensure TWS/IB Gateway is running
# Check port configuration (default: 7496 for TWS, 4001 for Gateway)
# Verify API permissions in TWS
```

**4. Google Sheets Integration:**
```bash
# Verify service account permissions
# Check sheet name and worksheet names match
# Ensure GDRIVE_CREDS_JSON is properly formatted
```

### Performance Optimization

**Large Universe Processing:**
```bash
# Use SSD storage for data directory
# Increase batch size on powerful machines
python app/run_data_extraction.py --mode batch --batch-size 100

# Use multiprocessing for model training
# Modify n_jobs parameter in hyperparameter tuning
```

**Memory Management:**
```bash
# For large datasets, process in chunks
# Use data type optimization (int8 for dummies)
# Clear intermediate variables in pipelines
```

## 📈 8. Strategy Implementation & Decision Making

### Decision Framework

**1. Strategy Selection Process:**

```bash
# Step 1: Generate all strategies
python app/run_predictions.py --mode recommend

# Step 2: Run simulations for capital efficiency
python app/run_simulations.py

# Step 3: Select top performers by precision and efficiency
# Review results/strategy_comparison_*.csv
# Review results/simulations_*.csv
```

**2. Implementation Guidelines:**

| Risk Profile | Recommended Strategies | Characteristics |
|--------------|----------------------|-----------------|
| **Conservative** | `pred15_rf_auto_rate_1p` | High precision (>60%), low frequency |
| **Balanced** | `pred11_rf_thresh_50` | Moderate precision (~50%), regular signals |
| **Aggressive** | `pred30_top3_daily` | Higher frequency, moderate precision |

### Position Sizing

**Capital Allocation Rules:**
- **Maximum position size**: 10% of portfolio (concentration limit)
- **Minimum position size**: 1% of portfolio (diversification requirement)
- **Cash buffer**: 10-20% for opportunities and risk management
- **Rebalancing**: Weekly or on significant model updates

### Risk Management

**Automated Monitoring:**
```bash
# Daily risk check (run via cron/scheduled task)
python portfolio-manage/portfolio-alerts.py

# Weekly deep analysis
python portfolio-manage/portfolio-positions-manage.py
```

**Risk Limits:**
- Maximum portfolio drawdown: -15%
- Maximum single position loss: -5%
- Correlation limit: No more than 3 positions >20% correlated
- Sector concentration: Maximum 25% in any single sector

## 📊 9. Decision Support Notebooks

The system includes Jupyter notebooks for advanced decision analysis and model monitoring:

### Using the Decision Notebooks

In addition to the core pipeline, the `decision-notebooks/` folder provides examples of applying the models for decision support:

**1. make_decisions.ipynb**
- Load your trained model outputs
- Define decision thresholds or business rules
- Evaluate the impact of decisions based on predictions
- **Run:**
  ```bash
  jupyter notebook decision-notebooks/make_decisions.ipynb
  ```

**2. model-drift.ipynb**
- Detect drift in data distributions over time
- Compare historical vs. recent model performance
- Suggest retraining strategies
- **Run:**
  ```bash
  jupyter notebook decision-notebooks/model-drift.ipynb
  ```

**3. model-simulations.ipynb**
- Run scenario analysis and sensitivity testing
- Simulate the effect of hypothetical interventions
- Evaluate stability of decision-making under uncertainty
- **Run:**
  ```bash
  jupyter notebook decision-notebooks/model-simulations.ipynb
  ```

### Portfolio Decision Support

**Daily Decision Workflow:**

1. **Morning Analysis** (Market Open - 1 hour):
   ```bash
   # Get latest predictions
   python app/run_predictions.py --mode generate
   
   # Check portfolio status
   python portfolio-manage/portfolio-alerts.py
   ```

2. **Trade Execution** (During Market Hours):
   - Review top 3-5 highest probability predictions
   - Check position sizing limits
   - Execute trades via IB platform
   - Log trades automatically via Flex Queries

3. **Evening Review** (Market Close + 1 hour):
   ```bash
   # Update portfolio data
   python portfolio-manage/log-journal.py
   
   # Send daily summary
   python portfolio-manage/send-summary.py
   ```

**Weekly Analysis:**
```bash
# Full portfolio rebalancing analysis
python portfolio-manage/portfolio-positions-manage.py

# Model performance review
python app/run_predictions.py --mode analyze --split test

# Strategy performance update
python app/run_simulations.py
```

## 🔄 10. Automation & Scheduling

### Daily Automation (Linux/Mac)

**Crontab Setup:**
```bash
# Edit crontab
crontab -e

# Add daily tasks
# 6:30 AM - Morning data update
30 6 * * 1-5 cd /path/to/personal-stock && python portfolio-manage/log-journal.py

# 9:00 AM - Generate predictions for today
0 9 * * 1-5 cd /path/to/personal-stock && python app/run_predictions.py --mode generate

# 6:00 PM - Evening portfolio summary
0 18 * * 1-5 cd /path/to/personal-stock && python portfolio-manage/portfolio-alerts.py
```

### Windows Task Scheduler

**Create Batch Files:**

`daily_morning.bat`:
```batch
@echo off
cd C:\path\to\personal-stock
python portfolio-manage\log-journal.py
python app\run_predictions.py --mode generate
```

`daily_evening.bat`:
```batch
@echo off
cd C:\path\to\personal-stock
python portfolio-manage\portfolio-alerts.py
python portfolio-manage\send-summary.py
```

### Cloud Deployment (GitHub Actions)

**Example workflow (`.github/workflows/daily-trading.yml`):**
```yaml
name: Daily Trading Pipeline
on:
  schedule:
    - cron: '30 13 * * 1-5'  # 6:30 AM PST (1:30 PM UTC)

jobs:
  trading-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run predictions
        env:
          GDRIVE_CREDS_JSON: ${{ secrets.GDRIVE_CREDS_JSON }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python app/run_predictions.py --mode generate
```

## 📋 11. Maintenance & Updates

### Model Retraining Schedule

**Weekly Updates:**
```bash
# Update data with latest week
python app/run_data_extraction.py --mode small --num-tickers 50

# Quick model retrain
python app/run_model_training.py --mode basic
```

**Monthly Deep Updates:**
```bash
# Full universe refresh
python app/run_data_extraction.py --mode batch --max-tickers 500

# Full hyperparameter optimization
python app/run_model_training.py --mode tune --tune-strategy progressive

# Strategy re-evaluation
python app/run_predictions.py --mode recommend
```

**Quarterly Reviews:**
```bash
# Full pipeline refresh
python app/run_data_extraction.py --mode batch --max-tickers 500 --validate
python app/run_model_training.py --mode tune --tune-strategy progressive --tune-validation walk_forward
python app/run_predictions.py --mode recommend
python app/run_simulations.py

# Performance analysis in decision notebooks
jupyter notebook decision-notebooks/model-drift.ipynb
```

### Data Quality Monitoring

**Automated Checks:**
- Missing data validation
- Feature distribution monitoring
- Target variable balance tracking
- Prediction distribution analysis

**Manual Reviews:**
- Feature importance changes
- Strategy performance degradation
- Market regime shifts
- Economic indicator relevance

## 🎯 12. Success Metrics & KPIs

### Model Performance KPIs

| Metric | Target | Monitoring |
|--------|--------|------------|
| **Precision** | >50% | Daily via predictions pipeline |
| **Daily Hit Rate Improvement** | >5% | Weekly via simulation results |
| **Sharpe Ratio** | >1.5 | Monthly via portfolio analysis |
| **Maximum Drawdown** | <15% | Real-time via portfolio alerts |

### Trading Performance KPIs

| Metric | Target | Frequency |
|--------|--------|-----------|
| **Win Rate** | >55% | Daily |
| **Average Win/Loss Ratio** | >1.2 | Weekly |
| **Portfolio Volatility** | <20% annualized | Daily |
| **Correlation to Market** | <0.7 | Monthly |

### Operational KPIs

| Metric | Target | Monitoring |
|--------|--------|------------|
| **Data Pipeline Uptime** | >99% | Automated alerts |
| **Model Prediction Latency** | <5 minutes | System logs |
| **Alert Delivery Success** | >95% | Telegram delivery reports |
| **Portfolio Sync Accuracy** | >99% | Daily reconciliation |

---

## 📞 Support & Contributing

**Issues & Bug Reports:**
- Open GitHub issues for bugs or feature requests
- Include system info, error logs, and reproduction steps

**Feature Contributions:**
- Fork the repository
- Create feature branch
- Submit pull request with tests and documentation

**Community:**
- Share successful strategies and configurations
- Contribute new technical indicators or macro features
- Improve documentation and usage examples

---

*Happy Trading! 📈*