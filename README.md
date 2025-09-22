# Stock Market Analytics & Trading System

A comprehensive quantitative trading system that combines machine learning predictions, technical analysis, macro economic indicators, and automated portfolio management for systematic trading decisions.

## 🎯 Objective

This system provides a complete end-to-end pipeline for:

- **Predictive Modeling**: ML-driven stock selection using Random Forest classification
- **Risk Management**: Portfolio analytics, drawdown monitoring, and concentration analysis  
- **Automated Trading**: Integration with Interactive Brokers for live portfolio tracking
- **Performance Monitoring**: Real-time alerts and comprehensive reporting via Telegram and Google Sheets

The core goal is to systematically identify stocks with positive 30-day forward returns while managing risk through diversification, position sizing, and automated monitoring.

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  ML Training    │    │  Live Trading   │
│                 │    │                 │    │                 │
│ • Stock Data    │───▶│ • Feature Eng.  │───▶│ • IB Integration│
│ • Technical     │    │ • Model Training│    │ • Portfolio Mgmt│
│ • Macro Data    │    │ • Backtesting   │    │ • Risk Alerts   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Storage  │    │   Model Store   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Parquet Files │    │ • Trained Models│    │ • Google Sheets │
│ • Google Sheets │    │ • Metadata      │    │ • Telegram Bot  │
│ • Time Series   │    │ • Thresholds    │    │ • Daily Reports │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 Data Pipeline & Transformations

### 1. Data Extraction (`app/extract.py`)

**Stock Market Data:**
- Fetches OHLCV data for S&P 500 + NASDAQ 100 universe
- Creates 30+ fundamental features including:
  - Historical growth rates (1d, 3d, 7d, 30d, 90d, 252d, 365d)
  - Moving averages (SMA10, SMA20) and crossover signals
  - Volatility and Sharpe ratios
  - Volume analytics (ln_volume)

**Target Variable Creation:**
- Binary classification: `is_positive_growth_30d_future`
- Threshold: 1% gain over 30-day forward period
- Prevents data leakage by excluding future information

### 2. Technical Analysis (`app/technicals.py`)

**TA-Lib Integration:**
- **Momentum Indicators**: RSI, MACD, ADX, CCI, Stochastic, Williams %R
- **Volume Indicators**: OBV, A/D Line, Money Flow Index
- **Volatility Indicators**: ATR, Bollinger Bands, Average True Range
- **Candlestick Patterns**: 50+ patterns (Doji, Hammer, Engulfing, etc.)
- **Trend Indicators**: Moving averages, trend strength, directional movement

**Data Processing:**
```python
# Example transformation flow
stock_data → technical_indicators → pattern_recognition → feature_matrix
```

### 3. Macro Economic Features (`app/macros.py`)

**Data Sources:**
- **FRED Economic Data**: Fed Funds Rate, Treasury yields (1Y, 5Y, 10Y), CPI
- **Market Indices**: VIX, S&P 500, DAX  
- **Commodities**: Gold, Oil (Brent/WTI)
- **Crypto**: Bitcoin as risk sentiment indicator

**Feature Engineering:**
- Year-over-year growth rates (`_yoy`)
- Quarter-over-quarter changes (`_qoq`) 
- 30/90-day momentum indicators
- Forward-filled for trading day alignment

### 4. Feature Engineering Pipeline

**Data Augmentation:**
```python
raw_data → basic_features → technical_analysis → macro_integration → ml_ready_dataset
```

**Feature Categories:**
- **Growth Features**: 7 lookback periods × multiple timeframes = 35+ features
- **Technical Features**: 100+ TA-Lib indicators across all categories
- **Macro Features**: 15+ economic indicators with growth rates
- **Time Features**: Month, weekday, week-of-month encoding
- **Dummy Variables**: Ticker encoding for stock-specific effects

**Data Quality:**
- Infinite value handling and imputation
- Temporal alignment across data sources
- Missing data forward-filling strategies
- Outlier detection and winsorization

## 🤖 Machine Learning Pipeline

### Model Architecture

**Random Forest Classifier:**
- Optimized for binary classification (positive/negative 30d returns)
- Hyperparameter tuning with walk-forward validation
- Feature importance analysis for model interpretability
- Multiple threshold optimization strategies

### Training Process (`app/train_model_new.py`)

**Data Splits:**
```python
# Temporal splits to prevent lookahead bias
train_split: 70% (earliest dates)
validation: 15% (middle period)  
test_split: 15% (most recent data)
```

**Feature Selection:**
- Removes any features containing "future" to prevent data leakage
- Includes 200+ engineered features
- Automatic dummy variable creation for categorical data
- Numeric type validation and cleaning

**Model Training Options:**
```bash
# Basic training with default parameters
python run_model_training.py --mode basic

# Hyperparameter tuning with walk-forward validation
python run_model_training.py --mode tune --tune-strategy progressive --tune-validation walk_forward
```

### Hyperparameter Optimization (`app/hyperparameter_tuning.py`)

**Tuning Strategies:**
- **Coarse Grid**: Quick exploration (~10 minutes)
- **Fine Grid**: Detailed search (~30-60 minutes) 
- **Progressive**: Coarse → Fine sequential optimization
- **Random Search**: Broad parameter space exploration

**Validation Methods:**
- **Static**: Traditional train/validation split
- **Walk-Forward**: Time-series aware validation with expanding window

**Optimization Metrics:**
- Primary: ROC-AUC (default), F1-Score, Precision, Recall
- Custom: Youden's J-statistic for ROC curve optimization

## 📊 Prediction & Strategy Engine

### Strategy Generation (`app/predictions.py`)

**Multiple Prediction Approaches:**

1. **Rule-Based Strategies:**
   - Technical momentum (CCI > 200, 30d growth > 1%)
   - Macro environment (declining rates, VIX contrarian signals)
   - Multi-factor combinations

2. **ML Probability-Based:**
   - Fixed thresholds (21%, 50%, 65%, 80%, 90%)
   - Adaptive thresholds (top 1%, 3%, 5% by validation performance)
   - Daily top-K selection (top 3, 5, 10 stocks per day)

3. **Ensemble Strategies:**
   - ML + momentum confirmation
   - Multiple rule consensus
   - Risk-adjusted combinations

### Strategy Evaluation

**Metrics-Focused Ranking:**
- **Primary**: Precision (minimizes false positives)
- **Secondary**: F1-Score, Recall, Daily improvement over baseline
- **Risk Metrics**: Prediction frequency, concentration analysis

**Performance Analysis:**
```python
# Strategy comparison on test set
results = comparator.evaluate_all_predictions(split="test")
```

## 🎯 Trading Simulation & Backtesting

### Simulation Engine (`app/run_simulations.py`)

**Per-Trade Simulation:**
- Investment amount: $100 per signal
- Transaction costs: 0.2% round-trip fees
- Hold period: 30 days (matches prediction horizon)
- Capital efficiency scoring

**Risk Metrics:**
- Maximum drawdown from daily P&L equity curve
- Capital requirement estimation (75th percentile concurrent positions)
- Efficiency score: `net_pnl / (capital_required × (1 + |max_drawdown|))`

**Performance Ranking:**
1. **Net P&L**: Absolute profit/loss after fees
2. **Efficiency Score**: Risk-adjusted returns per dollar of capital

## 📱 Portfolio Management & Monitoring

### Interactive Brokers Integration

**Real-Time Data Collection (`portfolio-manage/log-journal.py`):**
- Account summary metrics (cash, buying power, unrealized P&L)
- Position-level tracking (holdings, market values, returns)
- Automated Google Sheets logging every market day

**Trade Execution Tracking (`portfolio-manage/log-trades.py`):**
- IBKR Flex Query integration for trade history
- Position snapshot management
- Automated trade reconciliation

### Portfolio Analytics (`portfolio-manage/portfolio-positions-manage.py`)

**Position Analysis:**
- Concentration flags (positions > 10% of portfolio)
- Under-diversification alerts (positions < 1% of portfolio)
- Unrealized P&L and return calculations
- Cash allocation tracking

**Time Series Metrics:**
- Daily portfolio value and returns
- Rolling volatility (21-day)
- Drawdown analysis from running peaks
- Sharpe and Sortino ratio calculations

### Automated Alerting (`portfolio-manage/portfolio-alerts.py`)

**Daily Telegram Reports:**

1. **Portfolio Summary:**
   ```
   📊 Portfolio Summary (2024-01-15)
   • Value: $125,430.50
   • Daily P/L: +1.2%
   • Cumulative Return: +15.3%
   • 21-day Vol: 18.5%
   • Sharpe: 1.85
   ```

2. **Position Analysis:**
   ```
   📑 Position Summary (2024-01-15)
   • Cash: 12.5%
   • >10% positions: 2
   • <1% positions: 8
   • Top contributors: AAPL:+$450, NVDA:+$380
   • Bottom contributors: META:-$120, TSLA:-$95
   ```

## 🛠️ Installation & Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (technical analysis)
# On Windows: download from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# On Linux/Mac: 
conda install -c conda-forge ta-lib
```

### Environment Configuration

Create `.env` file:
```bash
# Interactive Brokers
IBKR_FLEX_TOKEN=your_flex_token
IBKR_FLEX_QUERY_ID=your_query_id

# Google Sheets  
GDRIVE_CREDS_JSON=your_service_account_json

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Project Structure

```
personal-stock/
├── .github/                      # GitHub Actions & Automation
│   └── workflows/                # Automated workflow definitions
│       ├── portfolio-summary.yml     # Daily portfolio alerts & analysis
│       └── portfolio-summary copy.txt # Backup/alternative workflow
├── app/                          # Core ML pipeline
│   ├── run_data_extraction.py    # Data pipeline orchestration
│   ├── run_model_training.py     # ML training orchestration
│   ├── run_predictions.py        # Strategy generation & evaluation
│   ├── run_simulations.py        # Backtesting & simulation
│   ├── extract.py                # Stock data fetching
│   ├── technicals.py             # TA-Lib integration
│   ├── macros.py                 # Economic data pipeline
│   ├── train_model_new.py        # ML training engine
│   ├── hyperparameter_tuning.py  # Model optimization
│   ├── predictions.py            # Strategy engine
│   ├── stock_pipeline.py         # Complete pipeline orchestration
│   ├── tickerfetch.py            # Universe fetching & liquidity ranking
│   └── augment.py                # Data augmentation utilities
├── portfolio-manage/             # Live trading integration
│   ├── log-journal.py            # IB account logging
│   ├── log-trades.py             # Trade execution tracking
│   ├── portfolio-alerts.py       # Automated monitoring
│   ├── portfolio-positions-manage.py  # Portfolio analytics
│   └── send-summary.py           # Manual reporting
├── decision-notebooks/           # Decision support & analysis
│   ├── make_decisions.ipynb      # Apply models to decision making
│   ├── model-drift.ipynb         # Monitor model performance over time
│   └── model-simulations.ipynb   # Scenario analysis & stress testing
├── data/                         # Processed datasets
│   ├── stock_data_combined_*.parquet    # Full universe datasets
│   ├── stock_data_small_*.parquet       # Test datasets
│   └── stock_data_batch_*.parquet       # Batch processing results
├── artifacts/                    # Trained models & metadata
│   ├── best_rf_model.joblib      # Latest trained model
│   ├── model_metadata.json       # Model configuration & features
│   └── *_timestamped.joblib      # Archived model versions
├── results/                      # Strategy results & analysis
│   ├── strategy_comparison_*.csv # Strategy performance rankings
│   ├── simulations_*.csv         # Trading simulation results
│   └── prediction_*.parquet      # Full prediction datasets
├── tuning_results/              # Hyperparameter optimization results
│   └── tuning_results_*.csv     # Grid search & optimization logs
├── ticker/                      # Universe definition & liquidity data
│   ├── tickers_combined_latest.csv     # S&P 500 + NASDAQ 100
│   └── spx_ndx_liq_top250_latest.csv   # Liquidity-ranked universe
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md                   # This file
```

## 🚀 Quick Start

**1. Extract Data (Small Test):**
```bash
python app/run_data_extraction.py --mode small --num-tickers 20 --validate
```

**2. Train Model:**
```bash  
python app/run_model_training.py --mode basic
```

**3. Generate Predictions:**
```bash
python app/run_predictions.py --mode recommend
```

**4. Run Simulations:**
```bash
python app/run_simulations.py
```

**5. Live Portfolio Monitoring:**
```bash
python portfolio-manage/log-journal.py        # Daily data collection
python portfolio-manage/portfolio-alerts.py   # Send alerts
```

## 📊 Decision Support Notebooks

This project includes a comprehensive set of decision-focused notebooks located in the `decision-notebooks/` directory. These notebooks demonstrate how the models and analysis can be applied to real-world decision-making scenarios:

### 📋 Available Notebooks

- **make_decisions.ipynb**  
  Walks through applying the trained models to make concrete trading decisions based on prediction outputs. Demonstrates threshold selection, position sizing, and risk management rules.

- **model-drift.ipynb**  
  Monitors model performance over time and demonstrates techniques to detect and handle model drift. Includes feature distribution analysis, performance degradation detection, and retraining triggers.

- **model-simulations.ipynb**  
  Runs comprehensive simulations with the trained models under different market scenarios to evaluate robustness and stress-test assumptions. Includes sensitivity analysis, scenario planning, and uncertainty quantification.

### 🎯 Usage

```bash
# Start Jupyter Lab to access decision notebooks
jupyter lab decision-notebooks/

# Or run individual notebooks
jupyter notebook decision-notebooks/make_decisions.ipynb
jupyter notebook decision-notebooks/model-drift.ipynb
jupyter notebook decision-notebooks/model-simulations.ipynb
```

These notebooks provide practical examples of converting model predictions into actionable investment decisions, monitoring system health, and stress-testing strategies under various market conditions.

## 📈 Results & Performance

The system generates comprehensive performance metrics including:

- **Strategy Precision**: Typically 40-65% for top strategies
- **Daily Hit Rate Improvement**: 5-15% above baseline
- **Risk-Adjusted Returns**: Sharpe ratios and maximum drawdown analysis
- **Capital Efficiency**: Optimized position sizing and timing

**Example Top Strategy Output:**
```
🏆 BEST STRATEGY: pred15_rf_auto_rate_1p
  Precision: 61.3%
  F1-Score: 0.245
  Daily Improvement: +8.7%
  Prediction Rate: 1.1% of time
```

## 🔗 Links

- **Detailed Usage Guide**: [usage.md](usage.md)
- **API Documentation**: See individual module docstrings
- **Contributing**: Open issues for bugs or feature requests

## ⚖️ Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Always conduct your own research and consider your risk tolerance before making investment decisions.

---

*Built with Python, scikit-learn, TA-Lib, Interactive Brokers API, and lots of coffee ☕*