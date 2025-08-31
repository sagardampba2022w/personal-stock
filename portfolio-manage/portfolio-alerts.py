# portfolio_alerts.py
"""
Send two Telegram messages each day using existing Google‑Sheet data.
Verbose debug prints show exactly where data might be missing.

1. **Portfolio Summary** – aggregate portfolio metrics.
2. **Position Summary**  – cash %, concentration flags, top/bottom PnL.

No data is written back to Sheets.

Worksheets required
-------------------
• `updated-positions`   – security‑level holdings (incl. CASH rows).
• `portfolio-summary`   – daily time‑series metrics (value, return, vol…).

Env‑vars needed
---------------
• `GDRIVE_CREDS_JSON`  (or local `gdrive-creds.json`)  
• `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import gspread
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from gspread_dataframe import get_as_dataframe

# ────────────────────────────────────────────────────────────
# 0.  Environment & Constants
# ────────────────────────────────────────────────────────────
load_dotenv()
TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SHEET   = "myportfolio"
WS_POS  = "updated-positions"
WS_TS   = "portfolio-summary"

NUMERIC_TS_COLS = [
    "PortfolioValue", "DailyReturn", "CumulativeReturn", "Drawdown",
    "RollingVol_21d", "Sharpe", "Sortino",
]

# ────────────────────────────────────────────────────────────
# 1.  Google‑Sheets helpers
# ────────────────────────────────────────────────────────────

def gclient() -> gspread.Client:
    """Create and return authenticated gspread client."""
    try:
        creds = os.getenv("GDRIVE_CREDS_JSON")
        if creds:
            print("[debug] Using creds from env var GDRIVE_CREDS_JSON")
            return gspread.service_account_from_dict(json.loads(creds))
        print("[debug] Using creds from gdrive-creds.json file")
        return gspread.service_account(filename="gdrive-creds.json")
    except Exception as e:
        print(f"[error] Failed to authenticate with Google Sheets: {e}")
        raise


def load_ws(name: str) -> pd.DataFrame:
    """Load worksheet data and return as DataFrame."""
    try:
        print(f"[debug] Loading worksheet '{name}' …")
        client = gclient()
        spreadsheet = client.open(SHEET)
        worksheet = spreadsheet.worksheet(name)
        
        df = get_as_dataframe(
            worksheet,
            evaluate_formulas=True,
            na_filter=False,
        )
        
        # Remove completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        
        # Remove rows where all values are empty strings
        df = df.replace('', np.nan).dropna(how="all")
        
        print(f"[debug] → rows: {df.shape[0]}, cols: {df.shape[1]}")
        print(f"[debug] → columns: {list(df.columns)}")
        
        if df.empty:
            print(f"[warn] Worksheet '{name}' is empty after cleaning")
        
        return df
        
    except gspread.WorksheetNotFound:
        print(f"[error] Worksheet '{name}' not found in spreadsheet '{SHEET}'")
        raise
    except Exception as e:
        print(f"[error] Failed to load worksheet '{name}': {e}")
        raise

# ────────────────────────────────────────────────────────────
# 2.  Data utilities
# ────────────────────────────────────────────────────────────

def to_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert column to datetime, handling various formats."""
    if col not in df.columns:
        print(f"[warn] Column '{col}' not found in DataFrame")
        return df
    
    out = df.copy()
    original_count = out[col].notna().sum()
    
    # Try multiple datetime formats
    out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=False)
    
    converted_count = out[col].notna().sum()
    failed_count = original_count - converted_count
    
    if failed_count > 0:
        print(f"[warn] Failed to convert {failed_count} datetime values in column '{col}'")
        print(f"[debug] Sample problematic values: {df[col][out[col].isna() & df[col].notna()].head().tolist()}")
    
    return out


def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert specified columns to numeric, with detailed error reporting."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            print(f"[warn] Column '{col}' not found in DataFrame")
            continue
            
        original_count = out[col].notna().sum()
        
        # Clean common formatting issues
        if out[col].dtype == 'object':
            # Remove currency symbols, commas, parentheses
            out[col] = out[col].astype(str).str.replace(r'[$,()]', '', regex=True)
            # Handle percentage signs
            pct_mask = out[col].str.contains('%', na=False)
            if pct_mask.any():
                out.loc[pct_mask, col] = pd.to_numeric(
                    out.loc[pct_mask, col].str.replace('%', ''), errors='coerce'
                ) / 100
                out.loc[~pct_mask, col] = pd.to_numeric(out.loc[~pct_mask, col], errors='coerce')
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        
        converted_count = out[col].notna().sum()
        failed_count = original_count - converted_count
        
        if failed_count > 0:
            print(f"[warn] Failed to convert {failed_count} numeric values in column '{col}'")
            problematic_values = df[col][out[col].isna() & df[col].notna()].head().tolist()
            print(f"[debug] Sample problematic values: {problematic_values}")
    
    return out


def add_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position weights as percentage of portfolio value."""
    out = df.copy()
    
    if "positionValue" not in out.columns:
        print("[error] Column 'positionValue' not found for weight calculation")
        return out
    
    out["positionValue"] = pd.to_numeric(out["positionValue"], errors="coerce")
    
    # Group by date and calculate weights
    def calc_weights(group):
        total_value = group["positionValue"].sum()
        if total_value == 0:
            print(f"[warn] Zero total portfolio value found for date group")
            return group["positionValue"] * 0
        return (group["positionValue"] / total_value * 100).fillna(0)
    
    if "reportDate" in out.columns:
        out["WeightInPortfolio"] = out.groupby("reportDate")["positionValue"].transform(calc_weights)
    else:
        print("[warn] 'reportDate' column not found, calculating weights for entire dataset")
        total_value = out["positionValue"].sum()
        if total_value != 0:
            out["WeightInPortfolio"] = (out["positionValue"] / total_value * 100).fillna(0)
        else:
            out["WeightInPortfolio"] = 0
    
    return out


def add_flags(df: pd.DataFrame, hi=10.0, lo=1.0) -> pd.DataFrame:
    """Add concentration and diversification flags."""
    out = df.copy()
    
    if "WeightInPortfolio" not in out.columns:
        print("[warn] 'WeightInPortfolio' column not found for flag calculation")
        out["ConcentrationFlag"] = False
        out["UnderDiversifiedFlag"] = False
        return out
    
    out["ConcentrationFlag"] = out["WeightInPortfolio"] > hi
    out["UnderDiversifiedFlag"] = out["WeightInPortfolio"] < lo
    
    return out


def top_bottom(df: pd.DataFrame, date: pd.Timestamp, n=3) -> Tuple[str, str]:
    """Get top and bottom performers by PnL for a specific date."""
    try:
        day = df[df["reportDate"] == date].copy()
        
        if day.empty:
            print(f"[warn] No position data found for date {date.date()}")
            return "—", "—"
        
        if "UnrealizedPnL" not in day.columns:
            print("[warn] 'UnrealizedPnL' column not found")
            return "—", "—"
        
        day["UnrealizedPnL"] = pd.to_numeric(day["UnrealizedPnL"], errors="coerce").fillna(0)
        
        # Filter out CASH and zero PnL positions for cleaner display
        day_filtered = day[(day["symbol"] != "CASH") & (day["UnrealizedPnL"] != 0)]
        
        if day_filtered.empty:
            return "—", "—"
        
        # Get top performers
        top_positions = day_filtered.nlargest(n, "UnrealizedPnL")
        top = ", ".join(f"{row['symbol']}:{row['UnrealizedPnL']:+.0f}" 
                       for _, row in top_positions.iterrows()) or "—"
        
        # Get bottom performers  
        bottom_positions = day_filtered.nsmallest(n, "UnrealizedPnL")
        bot = ", ".join(f"{row['symbol']}:{row['UnrealizedPnL']:+.0f}" 
                       for _, row in bottom_positions.iterrows()) or "—"
        
        return top, bot
        
    except Exception as e:
        print(f"[error] Error in top_bottom calculation: {e}")
        return "—", "—"

# ────────────────────────────────────────────────────────────
# 3.  Message builder
# ────────────────────────────────────────────────────────────

def pct(x):
    """Format percentage values."""
    if pd.isna(x) or x is None:
        return "—"
    try:
        return f"{float(x):+.2%}"
    except (ValueError, TypeError):
        return "—"

def num(x):
    """Format numeric values."""
    if pd.isna(x) or x is None:
        return "—"
    try:
        return f"{float(x):,.2f}"
    except (ValueError, TypeError):
        return "—"


def build_messages(ts_row: pd.Series, pos_df: pd.DataFrame, hi=10.0, lo=1.0) -> Tuple[str, str]:
    """Build both Telegram messages from the latest data."""
    try:
        date_dt = pd.to_datetime(ts_row["Date"])
        date = date_dt.date()
        print(f"[debug] Building messages for {date} …")

        # Slice positions for the specific date
        day_pos = pos_df[pos_df["reportDate"] == date_dt].copy()
        print(f"[debug]   positions rows: {day_pos.shape[0]}")
        
        if day_pos.empty:
            print(f"[warn] No position data found for {date}")
            # Create minimal messages with available portfolio data
            msg1 = (
                f"📊 *Portfolio Summary* ({date})\n\n"
                f"• Value:   ${num(ts_row.get('PortfolioValue'))}\n"
                f"• Daily P/L: {pct(ts_row.get('DailyReturn'))}\n"
                f"• Cum. Ret:  {pct(ts_row.get('CumulativeReturn'))}\n"
                f"• Drawdown:  {pct(ts_row.get('Drawdown'))}\n"
                f"• 21‑day Vol: {pct(ts_row.get('RollingVol_21d'))}\n"
                f"• Sharpe: {num(ts_row.get('Sharpe'))}  Sortino: {num(ts_row.get('Sortino'))}"
            )
            msg2 = f"📑 *Position Summary* ({date})\n\nNo position data available for this date."
            return msg1, msg2

        # Ensure weights are calculated
        if "WeightInPortfolio" not in day_pos.columns or day_pos["WeightInPortfolio"].isna().all():
            day_pos = add_weights(day_pos)
            print("[debug]   Weights calculated.")
        
        day_pos = add_flags(day_pos, hi, lo)

        # Calculate metrics
        cash_positions = day_pos[day_pos["symbol"] == "CASH"]
        cash_w = cash_positions["WeightInPortfolio"].sum() if not cash_positions.empty else 0
        
        n_conc = int(day_pos["ConcentrationFlag"].sum())
        n_under = int(day_pos["UnderDiversifiedFlag"].sum())
        top_str, bot_str = top_bottom(pos_df, date_dt)

        # Build messages
        msg1 = (
            f"📊 *Portfolio Summary* ({date})\n\n"
            f"• Value:   ${num(ts_row.get('PortfolioValue'))}\n"
            f"• Daily P/L: {pct(ts_row.get('DailyReturn'))}\n"
            f"• Cum. Ret:  {pct(ts_row.get('CumulativeReturn'))}\n"
            f"• Drawdown:  {pct(ts_row.get('Drawdown'))}\n"
            f"• 21‑day Vol: {pct(ts_row.get('RollingVol_21d'))}\n"
            f"• Sharpe: {num(ts_row.get('Sharpe'))}  Sortino: {num(ts_row.get('Sortino'))}"
        )
        
        msg2 = (
            f"📑 *Position Summary* ({date})\n\n"
            f"• Cash: {num(cash_w)}%\n"
            f"• >{hi}% positions: {n_conc}\n"
            f"• <{lo}% positions: {n_under}\n\n"
            f"• Top contributors: {top_str}\n"
            f"• Bottom contributors: {bot_str}"
        )
        
        return msg1, msg2
        
    except Exception as e:
        print(f"[error] Error building messages: {e}")
        # Return basic error messages
        date_str = str(ts_row.get("Date", "Unknown"))
        msg1 = f"📊 *Portfolio Summary* ({date_str})\n\nError processing data. Please check logs."
        msg2 = f"📑 *Position Summary* ({date_str})\n\nError processing data. Please check logs."
        return msg1, msg2

# ────────────────────────────────────────────────────────────
# 4.  Telegram helper
# ────────────────────────────────────────────────────────────

def send_tg(msg: str, label: str) -> bool:
    """Send message to Telegram and return success status."""
    if not TOKEN or not CHAT_ID:
        print("[warn] Telegram creds missing – skip send.")
        return False
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"[info] Sent {label} alert ✔")
        return True
    except requests.exceptions.RequestException as e:
        print(f"[error] Failed {label} alert: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[error] Response: {e.response.text}")
        return False

# ────────────────────────────────────────────────────────────
# 5.  Main
# ────────────────────────────────────────────────────────────

def main():
    """Main execution function."""
    try:
        print("[info] Starting portfolio alerts generation...")
        
        # Validate environment variables
        if not TOKEN:
            print("[error] TELEGRAM_BOT_TOKEN not found in environment")
        if not CHAT_ID:
            print("[error] TELEGRAM_CHAT_ID not found in environment")
        
        # Load data
        print("[info] Loading portfolio data...")
        ts_raw = load_ws(WS_TS)
        pos_raw = load_ws(WS_POS)
        
        if ts_raw.empty:
            print("[error] Portfolio summary data is empty")
            return
        
        if pos_raw.empty:
            print("[warn] Position data is empty")
        
        print("[debug] portfolio-summary head:")
        print(ts_raw.head())
        print("\n[debug] updated-positions head:")
        print(pos_raw.head())

        # Process time series data
        ts_df = to_dt(ts_raw, "Date")
        ts_df = to_numeric(ts_df, NUMERIC_TS_COLS)
        ts_df = ts_df.dropna(subset=["Date"])
        
        # Validate PortfolioValue exists
        if "PortfolioValue" not in ts_df.columns:
            print("[error] PortfolioValue column not found in portfolio-summary")
            return
        
        ts_df = ts_df.dropna(subset=["PortfolioValue"])
        
        print(f"[debug] Cleaned portfolio-summary rows: {ts_df.shape[0]}")
        
        if ts_df.empty:
            print("[error] No valid rows in portfolio-summary – abort.")
            return

        # Process positions data
        pos_df = to_dt(pos_raw, "reportDate") if not pos_raw.empty else pd.DataFrame()

        # Get latest data
        latest = ts_df.sort_values("Date").iloc[-1]
        latest_date = latest["Date"]
        print(f"[info] Latest portfolio data date: {latest_date}")
        
        # Build and send messages
        msg1, msg2 = build_messages(latest, pos_df)

        print("\n——— Debug output ———————————————")
        print("Portfolio Summary:")
        print(msg1)
        print("\n———————————————————————————————")
        print("Position Summary:")
        print(msg2)
        print("————————— end debug ————————————\n")

        # Send messages
        success_count = 0
        for label, message in [("Portfolio", msg1), ("Positions", msg2)]:
            if send_tg(message, label):
                success_count += 1

        print(f"[info] Successfully sent {success_count}/2 messages")
        
    except Exception as e:
        print(f"[error] Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()