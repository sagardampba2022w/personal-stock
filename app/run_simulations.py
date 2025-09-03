# run_simulations.py

import os
import numpy as np
import pandas as pd
from datetime import datetime

# Import your existing pipeline utilities
from train_model_new import TrainModel
from predictions import (
    load_latest_data,
    load_model_and_features,
    PredictionComparator,
    ARTIFACTS_DIR,
    RESULTS_DIR,
)

# ----------------------------
# Config (feel free to tweak)
# ----------------------------
SPLIT = "test"
INVEST_PER_TRADE = 100.0          # $100 per positive signal
FEE_RATE = 0.002                  # 0.2% round-trip (buy+sell)
HOLD_DAYS_FOR_CAPITAL = 30        # how many days capital is tied up per position
USE_RANK_BASED_AUTO = False       # toggle for rank-based auto-rate selectors
AUTO_TARGET_RATES = (0.01, 0.03, 0.05)
TOPK_LIST = (3, 5, 10)            # daily Top-K strategies
PROBA_COL = "rf_prob_30d"         # probability column added by add_ml_predictions
GROWTH_COL = "growth_future_30d"  # realized forward growth used in P&L
DATE_COL = "Date"
SPLIT_COL = "split"


# ----------------------------------------
# Helpers: rank-based "auto-rate" selectors
# ----------------------------------------
def add_rank_based_auto_rates(df: pd.DataFrame,
                              proba_col: str = PROBA_COL,
                              split_col: str = SPLIT_COL,
                              split_name: str = "validation",
                              target_rates=AUTO_TARGET_RATES) -> list:
    """
    Create selection columns that choose the top X% by rank on `split_name`
    (e.g., validation). This avoids brittle absolute thresholds like == 1.0.
    Returns the list of newly created column names.
    """
    if proba_col not in df.columns:
        print(f"[rank-auto] Missing probability column: {proba_col}")
        return []

    sub = df[df[split_col] == split_name][[proba_col]].copy()
    if sub.empty:
        print(f"[rank-auto] No rows in split '{split_name}'.")
        return []

    # Rank descending by probability; ties broken by order-of-appearance
    sub["_rank"] = sub[proba_col].rank(method="first", ascending=False)
    n = len(sub)

    created = []
    for r in target_rates:
        k = max(1, int(np.floor(n * float(r))))
        col = f"pred16_rank_auto_{int(r*100)}p"

        # Select exact top-k rows within the split
        top_idx = sub.nsmallest(k, "_rank").index  # smaller rank => higher prob
        df[col] = 0
        df.loc[top_idx, col] = 1

        v_rate = float(df.loc[df[split_col] == "validation", col].mean())
        t_rate = float(df.loc[df[split_col] == "test", col].mean())
        print(f"[rank-auto] {col}: val rate={v_rate:.2%} | test rate={t_rate:.2%}")
        created.append(col)

    return created


# ----------------------------------------
# Helpers: risk — max drawdown from daily PnL
# ----------------------------------------
def _max_drawdown_from_daily_pnl(daily_pnl: pd.Series, starting_capital: float) -> tuple[float, float]:
    """
    Build an equity curve starting at starting_capital and adding daily P&L.
    Returns (max_drawdown_abs, max_drawdown_pct) — drawdown pct is negative.
    """
    if daily_pnl.empty:
        return 0.0, 0.0

    equity = starting_capital + daily_pnl.cumsum()
    running_peak = equity.cummax()
    drawdowns = equity - running_peak
    max_dd_abs = float(drawdowns.min())  # negative value
    max_dd_pct = float(max_dd_abs / running_peak.max()) if running_peak.max() > 0 else 0.0
    return max_dd_abs, max_dd_pct


# ----------------------------------------
# Core simulation
# ----------------------------------------
def simulate_strategies(df: pd.DataFrame,
                        strategies: list[str],
                        split: str = SPLIT,
                        invest_per_trade: float = INVEST_PER_TRADE,
                        fee_rate: float = FEE_RATE,
                        growth_col: str = GROWTH_COL,
                        date_col: str = DATE_COL,
                        split_col: str = SPLIT_COL,
                        hold_days_for_capital: int = HOLD_DAYS_FOR_CAPITAL) -> pd.DataFrame:
    """
    Simple per-trade simulation:
      - Invest $INVEST_PER_TRADE when strategy==1 on the chosen split
      - Gross P&L = $ * (growth - 1)
      - Fees = -$ * fee_rate
      - Net P&L = Gross + Fees
      - Capital required ≈ $ * hold_days_for_capital * 75th percentile of daily concurrent positions
      - Max drawdown computed from daily P&L equity curve (starting at capital_required)
      - Efficiency score = net_pnl / (capital_required * (1 + |max_drawdown_pct|))
    """
    results = []

    need_cols = {split_col, date_col, growth_col}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for simulation: {missing}")

    for strat in strategies:
        if strat not in df.columns:
            print(f"[simulate] Skipping missing strategy column: {strat}")
            continue

        sub = df[(df[split_col] == split) & (df[strat] == 1)].copy()
        n_trades = len(sub)
        if n_trades == 0:
            # No trades — record zeros and continue
            results.append({
                "strategy": strat, "split": split, "n_trades": 0,
                "gross_pnl": 0.0, "fees": 0.0, "net_pnl": 0.0,
                "avg_pnl_per_trade": 0.0,
                "avg_positions_per_day": 0.0, "q75_positions_per_day": 0.0,
                "capital_required": 0.0, "cagr": 0.0,
                "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
                "efficiency_score": 0.0,
            })
            continue

        # Per-trade P&L
        sub["gross"] = invest_per_trade * (pd.to_numeric(sub[growth_col], errors="coerce") - 1.0)
        sub["fees"] = -invest_per_trade * fee_rate
        sub["net"] = sub["gross"] + sub["fees"]

        gross_pnl = float(sub["gross"].sum())
        fees_total = float(sub["fees"].sum())
        net_pnl = float(sub["net"].sum())
        avg_pnl_per_trade = float(net_pnl / n_trades)

        # Daily concurrency
        per_day = sub.groupby(date_col)[strat].count()
        avg_pos_per_day = float(per_day.mean()) if len(per_day) else 0.0
        q75_pos_per_day = float(per_day.quantile(0.75)) if len(per_day) else 0.0

        # Capital estimate — $ * HOLD_DAYS * Q75 concurrent positions
        capital_required = float(invest_per_trade * hold_days_for_capital * q75_pos_per_day)

        # CAGR over ~4 years if we assume split spans ~4y (adjust if needed)
        # If capital_required is zero (unlikely), set CAGR to 0
        if capital_required > 0:
            years = 4.0
            cagr = float(((capital_required + net_pnl) / capital_required) ** (1.0 / years))
        else:
            cagr = 0.0

        # Daily P&L -> Max Drawdown on equity curve
        daily_pnl = sub.groupby(date_col)["net"].sum()
        max_dd_abs, max_dd_pct = _max_drawdown_from_daily_pnl(daily_pnl, starting_capital=capital_required)

        # Efficiency score (risk & capital adjusted)
        efficiency = (net_pnl / capital_required / (1.0 + abs(max_dd_pct))) if capital_required > 0 else 0.0

        results.append({
            "strategy": strat,
            "split": split,
            "n_trades": int(n_trades),
            "gross_pnl": gross_pnl,
            "fees": fees_total,
            "net_pnl": net_pnl,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "avg_positions_per_day": avg_pos_per_day,
            "q75_positions_per_day": q75_pos_per_day,
            "capital_required": capital_required,
            "cagr": cagr,
            "max_drawdown": max_dd_abs,        # negative
            "max_drawdown_pct": max_dd_pct,    # negative fraction
            "efficiency_score": efficiency,    # NEW
        })

    cols = [
        "strategy", "split", "n_trades", "gross_pnl", "fees", "net_pnl",
        "avg_pnl_per_trade", "avg_positions_per_day", "q75_positions_per_day",
        "capital_required", "cagr", "max_drawdown", "max_drawdown_pct",
        "efficiency_score"
    ]
    return pd.DataFrame(results, columns=cols)


# ----------------------------------------
# Main
# ----------------------------------------
def main():
    print("\n" + "=" * 60)
    print("SIMULATION: LOADING DATA + MODEL AND RECREATING STRATEGIES")
    print("=" * 60)

    # 1) Load data (no training)
    df = load_latest_data()
    print(f"✓ Data loaded: {df.shape}")

    # 2) Prepare splits/targets (no training)
    tm = TrainModel(type("Adapter", (), {"transformed_df": df})())
    tm.prepare_dataframe(start_date="2000-01-01")

    # 3) Load trained model + feature order (no training)
    sk_model, feature_cols, target_col_from_model = load_model_and_features(ARTIFACTS_DIR)
    print(f"Target column: {getattr(tm, 'target_col', target_col_from_model)}")

    # --- PATCH: align ticker dummies to the training schema (silent zero-fill) ---
    expected = list(feature_cols)
    df = tm.df_full  # this already has dummies from TrainModel._define_dummies()

    # 1) Create any missing Ticker_* dummies (fill with 0, cast to int8)
    missing_feats = [c for c in expected if c not in df.columns]
    missing_ticker_dummies = [c for c in missing_feats if c.startswith("Ticker_")]
    if missing_ticker_dummies:
        for c in missing_ticker_dummies:
            df[c] = 0
        df[missing_ticker_dummies] = df[missing_ticker_dummies].astype("int8")
        # keep only non-ticker missing for later diagnostics (if any)
        missing_feats = [c for c in missing_feats if not c.startswith("Ticker_")]

    # 2) Drop any extra Ticker_* dummies not used by the model
    extra_ticker_dummies = [c for c in df.columns if c.startswith("Ticker_") and c not in expected]
    if extra_ticker_dummies:
        df.drop(columns=extra_ticker_dummies, inplace=True)

    # (ensure comparator sees patched frame)
    tm.df_full = df
    # --- END PATCH ---

    # 4) Build strategies via PredictionComparator (manual + ML)
    comparator = PredictionComparator(tm.df_full, tm.target_col)
    print("Creating manual rule-based predictions...")
    comparator.add_manual_predictions()

    # Add ML probs + fixed thresholds
    comparator.add_ml_predictions(sk_model, feature_cols, thresholds=(0.21, 0.50, 0.65, 0.80, 0.90))

    # --- DIAGNOSTIC: print ONLY if there are issues ---
    all_cols = set(comparator.df.columns)
    missing_feats = [c for c in feature_cols if c not in all_cols]
    extra_feats = [c for c in comparator.df.columns if c not in feature_cols and c.isidentifier()]

    if missing_feats:
        print(f"\n🔎 Inference diagnostic: {len(missing_feats)} expected features are MISSING (showing up to 20):")
        for name in missing_feats[:20]:
            print("     •", name)
    if extra_feats:
        print(f"\nℹ️  {len(extra_feats)} EXTRA columns exist in data that the model doesn’t use (showing up to 20):")
        for name in extra_feats[:20]:
            print("     •", name)

    # Choose auto-selectors: rank-based vs quantile-based
    if USE_RANK_BASED_AUTO:
        print("🎯 Creating rank-based auto-rate selectors (validation split)…")
        new_cols = add_rank_based_auto_rates(
            comparator.df,
            proba_col=PROBA_COL,
            split_col=SPLIT_COL,
            split_name="validation",
            target_rates=AUTO_TARGET_RATES,
        )
        comparator.prediction_cols.extend(new_cols)
    else:
        print("🎯 Creating quantile-based auto thresholds (validation split)…")
        comparator.add_ml_thresholds_from_validation(PROBA_COL, target_rates=AUTO_TARGET_RATES)

    # Daily Top-K strategies
    print("[Top-K] Creating daily Top-K selectors…")
    for k in TOPK_LIST:
        comparator.add_daily_topn(proba_col=PROBA_COL, n=k, date_col=DATE_COL)

    # Ensembles
    print("Creating ensemble predictions...")
    comparator.add_ensemble_predictions()
    print(f"✓ Strategies created: {len(comparator.prediction_cols)}")

    # 5) Run simulations on the chosen split
    print("\n" + "=" * 60)
    print(f"RUNNING SIMULATIONS on split={SPLIT}")
    print("=" * 60)

    # Guard: require growth column for P&L
    if GROWTH_COL not in comparator.df.columns:
        raise ValueError(
            f"Required growth column '{GROWTH_COL}' not found in dataframe. "
            "Ensure your pipeline produced it."
        )

    # Run the simulator
    sim_df = simulate_strategies(
        comparator.df,
        strategies=comparator.prediction_cols,
        split=SPLIT,
        invest_per_trade=INVEST_PER_TRADE,
        fee_rate=FEE_RATE,
        growth_col=GROWTH_COL,
        date_col=DATE_COL,
        split_col=SPLIT_COL,
        hold_days_for_capital=HOLD_DAYS_FOR_CAPITAL,
    )

    # 6) Print leaderboards + save CSV
    top_net = sim_df.sort_values("net_pnl", ascending=False).head(10)
    print("\nTOP 10 (by net_pnl):")
    print(top_net.to_string(index=False))

    top_eff = sim_df.sort_values("efficiency_score", ascending=False).head(10)
    print("\nTOP 10 (by efficiency_score):")
    print(top_eff.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR, f"simulations_{SPLIT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    sim_df.to_csv(out_path, index=False)
    print(f"\n💾 Saved full simulation results to: {out_path}")


if __name__ == "__main__":
    main()
