import gspread
import pandas as pd
import json
import os
import numpy as np
import yfinance as yf
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import pandas as pd

# ------- Load Data--------
def get_gspread_client():
    """Get gspread client using either file or environment variable"""
    gdrive_creds_json = os.getenv("GDRIVE_CREDS_JSON")
    if gdrive_creds_json:
        print("Using environment variable for credentials")
        creds_dict = json.loads(gdrive_creds_json)
        return gspread.service_account_from_dict(creds_dict)
    else:
        print("Using file for credentials")
        return gspread.service_account(filename='gdrive-creds.json')

# def load_positions_from_gsheet(sheet_name="myportfolio", worksheet_name="flex-positions"):
#     """Load positions from Google Sheet as DataFrame"""
#     gc = get_gspread_client()
#     sh = gc.open(sheet_name)
#     ws_positions = sh.worksheet(worksheet_name)
#     positions = get_as_dataframe(ws_positions, evaluate_formulas=True, na_filter=False)
#     # Drop empty rows
#     positions = positions.dropna(how='all')
#     return positions

# def load_trades_from_gsheet(sheet_name="myportfolio", worksheet_name="flex-trades"):
#     """Load trades from Google Sheet as DataFrame"""
#     gc = get_gspread_client()
#     sh = gc.open(sheet_name)
#     ws_trades = sh.worksheet(worksheet_name)
#     trades = get_as_dataframe(ws_trades, evaluate_formulas=True, na_filter=False)
#     trades = trades.dropna(how='all')
#     return trades


def load_gsheet_worksheet(sheet_name, worksheet_name):
    """
    Load any worksheet from a Google Sheet as a pandas DataFrame.
    """
    gc = get_gspread_client()
    sh = gc.open(sheet_name)
    ws = sh.worksheet(worksheet_name)
    df = get_as_dataframe(ws, evaluate_formulas=True, na_filter=False)
    df = df.dropna(how='all')
    return df


# Or, if you want to support CSV fallback
def load_positions_from_csv(path):
    """Load positions from CSV if needed"""
    return pd.read_csv(path)




# ------- Preprocess Positions---------
def preprocess_positions(positions):
    """Preprocess IBKR positions Flex DataFrame"""
    import numpy as np
    # Parse openDateTime from IBKR format
    def parse_ibkr_datetime(val):
        try:
            if pd.isnull(val):
                return pd.NaT
            val = str(val)
            if ';' in val and len(val) >= 15:
                return pd.to_datetime(val, format='%Y%m%d;%H%M%S')
            elif len(val) == 8 and val.isdigit():
                return pd.to_datetime(val, format='%Y%m%d')
            else:
                return pd.NaT
        except Exception:
            return pd.NaT

    positions = positions.copy()
    positions['openDateTime'] = positions['openDateTime'].astype(str).replace('nan', '')
    positions['openDateTime'] = positions['openDateTime'].apply(parse_ibkr_datetime)

    positions['levelOfDetail'] = positions['levelOfDetail'].str.strip().str.upper()
    positions['levelOfDetail_order'] = positions['levelOfDetail'].map({'LOT': 0, 'SUMMARY': 1, '': 2})

    # Ensure numeric columns
    num_cols = [
        'positionValue', 'markPrice', 'costBasisPrice', 'position',
        'multiplier', 'fxRateToBase', 'fifoPnlUnrealized'
    ]
    for col in num_cols:
        if col in positions.columns:
            positions[col] = pd.to_numeric(positions[col], errors='coerce')

    # Sort so LOTs before SUMMARY by symbol
    positions = positions.sort_values(by=['symbol', 'levelOfDetail_order'])

    # Backfill openDateTime from LOT to SUMMARY by symbol
    positions['openDateTime'] = positions.groupby('symbol')['openDateTime'].ffill()

    # Filter only SUMMARY rows
    positions_summary = positions[positions['levelOfDetail'] == 'SUMMARY'].copy()

    # Add portfolio metrics
    positions_summary['WeightInPortfolio'] = (positions_summary['positionValue'] / positions_summary['positionValue'].sum()) * 100
    positions_summary['UnrealizedReturnPct'] = ((positions_summary['markPrice'] - positions_summary['costBasisPrice']) / positions_summary['costBasisPrice']) * 100
    positions_summary['UnrealizedPnL'] = (positions_summary['markPrice'] - positions_summary['costBasisPrice']) * positions_summary['position'] * positions_summary['multiplier']
    positions_summary['days_held'] = (pd.Timestamp.today() - positions_summary['openDateTime']).dt.days

    positions_summary['AnnualizedReturn'] = np.where(
        positions_summary['days_held'] > 0,
        ((positions_summary['markPrice'] / positions_summary['costBasisPrice']) ** (365 / positions_summary['days_held']) - 1) * 100,
        np.nan
    )
    positions_summary['AnnualizedReturn'] = positions_summary['AnnualizedReturn'].replace([np.inf, -np.inf], np.nan)

    calc_cols = ['WeightInPortfolio', 'UnrealizedReturnPct', 'AnnualizedReturn', 'UnrealizedPnL']
    for col in calc_cols:
        positions_summary[col] = pd.to_numeric(positions_summary[col], errors='coerce').round(2)

    positions_summary = positions_summary.drop(columns=['levelOfDetail_order'], errors='ignore')
    return positions_summary


def get_cash_by_report_date(account_df, positions_summary, tag='TotalCashValue'):
    # Convert datetimes to date (not datetime)
    account_df = account_df.copy()
    account_df['DateTime'] = pd.to_datetime(account_df['DateTime'])
    # Truncate time so only the date matches reportDate
    account_df['reportDate'] = account_df['DateTime'].dt.strftime('%Y%m%d')
    # Filter for the cash tag and relevant account (if needed)
    cash_df = account_df[account_df['Tag'] == tag]
    # Keep only latest cash value for each reportDate
    latest_cash = (cash_df.sort_values('DateTime')
                            .groupby('reportDate')
                            .tail(1))
    # Map to positions_summary dates
    cash_map = latest_cash.set_index('reportDate')['Value'].astype(float)
    return cash_map



# def add_cash_rows(positions_summary, cash_map):
#     """Add a cash row for each report date (if missing) using cash_map Series"""
#     positions_list = [positions_summary]
#     # For each unique reportDate, add a cash row if not already present
#     for date, cash_value in cash_map.items():
#         mask = (positions_summary['reportDate'] == date) & (positions_summary['symbol'] == 'CASH')
#         if not mask.any():  # only add if not present
#             row = {col: None for col in positions_summary.columns}
#             row.update({'symbol': 'CASH', 'positionValue': cash_value, 'reportDate': date})
#             positions_list.append(pd.DataFrame([row]))
#     # Combine all together
#     return pd.concat(positions_list, ignore_index=True)

def add_cash_for_latest_position_date(positions_summary, account_summary, tag='TotalCashValue'):
    """
    For the last date in positions, add a cash row using the account summary.
    Returns a DataFrame with positions + cash row for that date only.
    """
    positions_summary = positions_summary.copy()
    positions_summary['reportDate'] = pd.to_datetime(positions_summary['reportDate'], format='%Y%m%d')
    account_summary = account_summary.copy()
    account_summary['DateTime'] = pd.to_datetime(account_summary['DateTime'])
    account_summary['reportDate'] = account_summary['DateTime'].dt.normalize()
    
    # Find last date in positions
    last_pos_date = positions_summary['reportDate'].max()
    # Get latest cash row for <= last_pos_date
    cash_candidates = account_summary[(account_summary['Tag'] == tag) & 
                                      (account_summary['reportDate'] <= last_pos_date)]
    if cash_candidates.empty:
        print(f"No cash value found for {last_pos_date.date()}.")
        cash_value = 0
    else:
        # Find the latest cash on or before last position date
        last_cash_row = cash_candidates.sort_values('reportDate').iloc[-1]
        cash_value = float(last_cash_row['Value'])

    # Get all positions for last_pos_date
    day_positions = positions_summary[positions_summary['reportDate'] == last_pos_date].copy()
    

    cash_row = {col: None for col in day_positions.columns}
    cash_row.update({
        'symbol': 'CASH',
        'positionValue': cash_value,
        'reportDate': last_pos_date,
        'position_key': f'CASH_{last_pos_date.strftime("%Y%m%d")}'
    })


    result = pd.concat([day_positions, pd.DataFrame([cash_row])], ignore_index=True)
    return result


def add_cash_rows_all_dates(positions_summary, account_summary, tag='TotalCashValue'):
    """
    For every unique reportDate in positions, add a cash row using the best-matching date from account summary.
    """
    positions_summary = positions_summary.copy()
    positions_summary['reportDate'] = pd.to_datetime(positions_summary['reportDate'], format='%Y%m%d')
    account_summary = account_summary.copy()
    account_summary['DateTime'] = pd.to_datetime(account_summary['DateTime'])
    account_summary['reportDate'] = account_summary['DateTime'].dt.normalize()
    
    result = []
    unique_dates = positions_summary['reportDate'].unique()
    for date in unique_dates:
        day_positions = positions_summary[positions_summary['reportDate'] == date].copy()
        # Find latest cash <= this date
        cash_candidates = account_summary[(account_summary['Tag'] == tag) & 
                                          (account_summary['reportDate'] <= date)]
        if cash_candidates.empty:
            cash_value = 0
        else:
            last_cash_row = cash_candidates.sort_values('reportDate').iloc[-1]
            cash_value = float(last_cash_row['Value'])
        # Add cash row
        # cash_row = {col: None for col in day_positions.columns}
        # cash_row.update({
        #     'symbol': 'CASH',
        #     'positionValue': cash_value,
        #     'reportDate': date
        # })

        # cash_row = {col: None for col in day_positions.columns}
        # cash_row.update({
        #     'symbol': 'CASH',
        #     'positionValue': cash_value,
        #     'reportDate': date,
        #     'position_key': f'CASH_{date.strftime("%Y%m%d")}'
        # })
        cash_row = {col: None for col in day_positions.columns}
        cash_row.update({
            'symbol': 'CASH',
            'positionValue': cash_value,
            'reportDate': date,
            'position_key': f'CASH_{date.strftime("%Y%m%d")}',
            # Fill in typical fields to avoid dtype issues:
            'currency': 'USD',
            'accountId': day_positions['accountId'].iloc[0] if 'accountId' in day_positions and not day_positions.empty else None,
            # Add other fields if you want (not required)
        })


        day_with_cash = pd.concat([day_positions, pd.DataFrame([cash_row])], ignore_index=True)
        result.append(day_with_cash)
        result['positionValue'] = pd.to_numeric(result['positionValue'], errors='coerce')
    return pd.concat(result, ignore_index=True)



def add_cash_rows_all_dates(positions_summary, account_summary, tag='TotalCashValue'):
    """
    For every unique reportDate in positions, add a cash row using the best-matching date from account summary.
    """
    positions_summary = positions_summary.copy()
    positions_summary['reportDate'] = pd.to_datetime(positions_summary['reportDate'], format='%Y%m%d')
    account_summary = account_summary.copy()
    account_summary['DateTime'] = pd.to_datetime(account_summary['DateTime'])
    account_summary['reportDate'] = account_summary['DateTime'].dt.normalize()
    
    result = []
    unique_dates = positions_summary['reportDate'].unique()
    for date in unique_dates:
        day_positions = positions_summary[positions_summary['reportDate'] == date].copy()
        # Find latest cash <= this date
        cash_candidates = account_summary[(account_summary['Tag'] == tag) & 
                                          (account_summary['reportDate'] <= date)]
        if cash_candidates.empty:
            cash_value = 0
        else:
            last_cash_row = cash_candidates.sort_values('reportDate').iloc[-1]
            cash_value = float(last_cash_row['Value'])
        # Add cash row
        cash_row = {col: None for col in day_positions.columns}
        cash_row.update({
            'symbol': 'CASH',
            'positionValue': cash_value,
            'reportDate': date,
            'position_key': f'CASH_{date.strftime("%Y%m%d")}',
            'currency': 'USD',
            'accountId': day_positions['accountId'].iloc[0] if 'accountId' in day_positions and not day_positions.empty else None,
        })

        day_with_cash = pd.concat([day_positions, pd.DataFrame([cash_row])], ignore_index=True)
        result.append(day_with_cash)
    
    # Fix: concat after the loop and convert to numeric
    result_df = pd.concat(result, ignore_index=True)
    result_df['positionValue'] = pd.to_numeric(result_df['positionValue'], errors='coerce')
    return result_df


# --------------------
# Data Preparation
# --------------------

def ensure_reportdate_datetime(df, date_col='reportDate'):
    """Ensure the reportDate column is datetime."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
    return df

# --------------------
# Portfolio Time Series
# --------------------

def build_portfolio_history(positions, date_col='reportDate', value_col='positionValue'):
    """Aggregate total portfolio value by report date."""
    hist = (
        positions.groupby(date_col, as_index=False)[value_col].sum()
        .rename(columns={date_col: 'Date', value_col: 'PortfolioValue'})
        .sort_values('Date')
        .reset_index(drop=True)
    )
    return hist

def add_timeseries_metrics(history_df):
    """Add returns, rolling vol, drawdown, etc. to time series."""
    df = history_df.copy()
    df['DailyReturn'] = df['PortfolioValue'].pct_change()
    df['CumulativeReturn'] = (1 + df['DailyReturn']).cumprod() - 1
    df['RollingVol_21d'] = df['DailyReturn'].rolling(21).std() * np.sqrt(252)
    df['RollingMax'] = df['PortfolioValue'].cummax()
    df['Drawdown'] = df['PortfolioValue'] / df['RollingMax'] - 1
    return df

def add_pnl_to_history(history_df, positions, date_col='reportDate'):
    """Merge UnrealizedPnL by date if available."""
    if 'UnrealizedPnL' in positions.columns:
        pnl_by_date = positions.groupby(date_col)['UnrealizedPnL'].sum().rename('UnrealizedPnL')
        return history_df.merge(pnl_by_date, left_on='Date', right_index=True, how='left')
    return history_df

def compute_sharpe_sortino(df, risk_free_rate=0.05):
    """Compute Sharpe and Sortino ratio for time series."""
    mean_return = df['DailyReturn'].mean()
    std_return = df['DailyReturn'].std()
    downside_std = df['DailyReturn'][df['DailyReturn'] < 0].std()
    sharpe = (mean_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
    sortino = (mean_return * 252 - risk_free_rate) / (downside_std * np.sqrt(252))
    return sharpe, sortino

# --------------------
# Cross Sectional Analytics
# --------------------

def get_top_bottom_contributors(positions, N=5, pnl_col='UnrealizedPnL'):
    """Return top and bottom N contributors for the latest date."""
    latest_date = positions['reportDate'].max()
    latest = positions[positions['reportDate'] == latest_date]
    top_n = latest[latest[pnl_col] > 0].nlargest(N, pnl_col)[['symbol', pnl_col]]
    bottom_n = latest[latest[pnl_col] < 0].nsmallest(N, pnl_col)[['symbol', pnl_col]]
    return latest_date, top_n, bottom_n

def add_position_flags(positions, weight_col='WeightInPortfolio', high=10, low=1):
    """Add overconcentration and underdiversified flags."""
    positions = positions.copy()
    positions['ConcentrationFlag'] = positions[weight_col] > high
    positions['UnderDiversifiedFlag'] = positions[weight_col] < low
    return positions

def add_weight_column(positions, value_col='positionValue'):
    """Add WeightInPortfolio column."""
    total = positions[value_col].sum()
    positions = positions.copy()
    positions['WeightInPortfolio'] = positions[value_col] / total * 100
    return positions


def recalc_weight_by_date(df, value_col='positionValue'):
    df = df.copy()
    df['WeightInPortfolio'] = (
        df.groupby('reportDate')[value_col]
        .transform(lambda x: x / x.sum() * 100)
    )
    return df

def recalc_weight_by_date(df, value_col='positionValue'):
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df['WeightInPortfolio'] = (
        df.groupby('reportDate')[value_col]
        .transform(lambda x: x / x.sum() * 100)
    )
    # Debug: Print all cash rows after calc
    cash_rows = df[df['symbol'] == 'CASH']
    print("\nAfter recalc, cash rows:")
    print(cash_rows[['reportDate', 'positionValue', 'WeightInPortfolio']])
    # Optionally: Fill 0 for missing values
    # df['WeightInPortfolio'] = df['WeightInPortfolio'].fillna(0)
    return df




def build_portfolio_summary_table(portfolio_history):
    """Create a summary DataFrame with key metrics for Google Sheets."""
    sharpe, sortino = compute_sharpe_sortino(portfolio_history)
    summary = portfolio_history.copy()
    # Add Sharpe/Sortino to all rows, or just latest (optional)
    summary["Sharpe"] = sharpe
    summary["Sortino"] = sortino
    # Fill missing columns if needed
    if 'UnrealizedPnL' not in summary.columns:
        summary['UnrealizedPnL'] = np.nan
    # Choose columns to keep
    keep = [
        'Date', 'PortfolioValue', 'DailyReturn', 'CumulativeReturn', 
        'RollingVol_21d', 'Drawdown', 'Sharpe', 'Sortino', 'UnrealizedPnL'
    ]
    return summary[keep]



# --------------------
# append positions and sumamry to sheets
# --------------------

# def append_positions_to_gsheet(new_positions_df, sheet_name="myportfolio", history_sheet_name="updated-positions"):

#     gc = get_gspread_client()
#     sh = gc.open(sheet_name)
    
#     # Try to load the historical sheet, or create if not exists
#     try:
#         ws_hist = sh.worksheet(history_sheet_name)
#         hist_df = get_as_dataframe(ws_hist, evaluate_formulas=True, na_filter=False).dropna(how="all")
#     except gspread.exceptions.WorksheetNotFound:
#         print(f"Creating new worksheet: {history_sheet_name}")
#         ws_hist = sh.add_worksheet(title=history_sheet_name, rows=1000, cols=50)
#         hist_df = pd.DataFrame(columns=new_positions_df.columns)
    
#     # Remove duplicates using position_key
#     existing_keys = set(hist_df['position_key']) if not hist_df.empty else set()
#     new_rows = new_positions_df[~new_positions_df['position_key'].isin(existing_keys)]
    
#     if new_rows.empty:
#         print("No new positions to append.")
#         return
    
#     # Append new rows
#     updated_hist = pd.concat([hist_df, new_rows], ignore_index=True)
#     set_with_dataframe(ws_hist, updated_hist, include_index=False)
#     print(f"Appended {len(new_rows)} new positions to '{history_sheet_name}'.")

def append_positions_to_gsheet(new_positions_df, sheet_name="myportfolio", history_sheet_name="updated-positions"):

    gc = get_gspread_client()
    sh = gc.open(sheet_name)
    
    # Try to load the historical sheet, or create if not exists
    try:
        ws_hist = sh.worksheet(history_sheet_name)
        hist_df = get_as_dataframe(ws_hist, evaluate_formulas=True, na_filter=False).dropna(how="all")

        # If the sheet is empty (no columns) or missing 'position_key', initialize it with new_positions_df columns
        if hist_df.empty or 'position_key' not in hist_df.columns:
            hist_df = pd.DataFrame(columns=new_positions_df.columns)
            set_with_dataframe(ws_hist, hist_df, include_index=False)  # Write just header row

    except gspread.exceptions.WorksheetNotFound:
        print(f"Creating new worksheet: {history_sheet_name}")
        ws_hist = sh.add_worksheet(title=history_sheet_name, rows=1000, cols=50)
        hist_df = pd.DataFrame(columns=new_positions_df.columns)
        set_with_dataframe(ws_hist, hist_df, include_index=False)  # Write just header row

    # Remove duplicates using position_key
    existing_keys = set(hist_df['position_key']) if not hist_df.empty else set()
    new_rows = new_positions_df[~new_positions_df['position_key'].isin(existing_keys)]
    
    if new_rows.empty:
        print("No new positions to append.")
        return
    
    # Append new rows
    updated_hist = pd.concat([hist_df, new_rows], ignore_index=True)
    set_with_dataframe(ws_hist, updated_hist, include_index=False)
    print(f"Appended {len(new_rows)} new positions to '{history_sheet_name}'.")


def upsert_portfolio_summary_to_gsheet(summary_df, sheet_name="myportfolio", summary_sheet_name="portfolio-summary"):
    gc = get_gspread_client()
    sh = gc.open(sheet_name)
    try:
        ws_summary = sh.worksheet(summary_sheet_name)
        existing = get_as_dataframe(ws_summary, evaluate_formulas=True, na_filter=False).dropna(how="all")
        # If sheet is empty or missing 'Date' column, initialize
        if existing.empty or 'Date' not in existing.columns:
            existing = pd.DataFrame(columns=summary_df.columns)
            set_with_dataframe(ws_summary, existing, include_index=False)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Creating new worksheet: {summary_sheet_name}")
        ws_summary = sh.add_worksheet(title=summary_sheet_name, rows=1000, cols=len(summary_df.columns))
        existing = pd.DataFrame(columns=summary_df.columns)
        set_with_dataframe(ws_summary, existing, include_index=False)
    # Only append new dates
    existing_dates = set(pd.to_datetime(existing['Date']).dt.date) if not existing.empty else set()
    new_rows = summary_df[~pd.to_datetime(summary_df['Date']).dt.date.isin(existing_dates)]
    if new_rows.empty:
        print("No new summary rows to append.")
        return
    updated = pd.concat([existing, new_rows], ignore_index=True)
    set_with_dataframe(ws_summary, updated, include_index=False)
    print(f"Appended {len(new_rows)} new summary rows to '{summary_sheet_name}'.")




# --------------------
# Example main script
# --------------------

# def main(positions_with_cash):
#     # 1. Ensure dates
#     positions_with_cash = ensure_reportdate_datetime(positions_with_cash)

#     # 2. Build time series
#     portfolio_history = build_portfolio_history(positions_with_cash)
#     portfolio_history = add_timeseries_metrics(portfolio_history)
#     portfolio_history = add_pnl_to_history(portfolio_history, positions_with_cash)


#     # 4. Performance ratios
#     sharpe, sortino = compute_sharpe_sortino(portfolio_history)
#     print(f"Sharpe Ratio: {sharpe:.2f}, Sortino Ratio: {sortino:.2f}")

#     # 5. Top/bottom contributors (latest date)
#     latest_date, top_n, bottom_n = get_top_bottom_contributors(positions_with_cash)
#     print(f"\nTop contributors for {latest_date.date()}:\n", top_n)
#     print(f"Top detractors for {latest_date.date()}:\n", bottom_n)

#     # 6. Concentration/Underdiversified flags (latest date)
#     latest = positions_with_cash[positions_with_cash['reportDate'] == latest_date]
#     latest = add_weight_column(latest)
#     latest = add_position_flags(latest)
#     print("\nConcentration/UnderDiversified Flags (latest date):")
#     print(latest[['symbol', 'WeightInPortfolio', 'ConcentrationFlag', 'UnderDiversifiedFlag']])
#     return portfolio_history  # <<<< ADD THIS


# def main(positions_with_cash):
#     # 1. Ensure dates
#     positions_with_cash = ensure_reportdate_datetime(positions_with_cash)
#     positions_with_cash['positionValue'] = pd.to_numeric(positions_with_cash['positionValue'], errors='coerce')

    
#     # 2. Recalculate weights *after* adding cash (do this early!)
#     positions_with_cash = recalc_weight_by_date(positions_with_cash)

#     # 3. Build time series
#     portfolio_history = build_portfolio_history(positions_with_cash)
#     portfolio_history = add_timeseries_metrics(portfolio_history)
#     portfolio_history = add_pnl_to_history(portfolio_history, positions_with_cash)

#     # 4. Performance ratios
#     sharpe, sortino = compute_sharpe_sortino(portfolio_history)
#     print(f"Sharpe Ratio: {sharpe:.2f}, Sortino Ratio: {sortino:.2f}")

#     # 5. Top/bottom contributors (latest date)
#     latest_date, top_n, bottom_n = get_top_bottom_contributors(positions_with_cash)
#     print(f"\nTop contributors for {latest_date.date()}:\n", top_n)
#     print(f"Top detractors for {latest_date.date()}:\n", bottom_n)

#     # 6. Concentration/Underdiversified flags (latest date)
#     latest = positions_with_cash[positions_with_cash['reportDate'] == latest_date]
#     latest = add_position_flags(latest)  # Now WeightInPortfolio is correct
#     print("\nConcentration/UnderDiversified Flags (latest date):")
#     print(latest[['symbol', 'WeightInPortfolio', 'ConcentrationFlag', 'UnderDiversifiedFlag']])
#     return portfolio_history


def main(positions_with_cash):
    """
    • Re-calculates weights (after CASH rows are in)
    • Builds the performance time-series
    • Prints analytics
    • RETURNS BOTH the history table *and* the updated positions DataFrame
    """
    # ------------------------------------------------------------------
    # 1. Clean up / type-cast
    # ------------------------------------------------------------------
    positions_with_cash = ensure_reportdate_datetime(positions_with_cash)
    positions_with_cash['positionValue'] = pd.to_numeric(
        positions_with_cash['positionValue'], errors='coerce'
    )

    # ------------------------------------------------------------------
    # 2. Recalculate weights *after* cash has been added
    # ------------------------------------------------------------------
    positions_with_cash = recalc_weight_by_date(positions_with_cash)

    # ------------------------------------------------------------------
    # 3. Build portfolio time series + metrics
    # ------------------------------------------------------------------
    portfolio_history = build_portfolio_history(positions_with_cash)
    portfolio_history = add_timeseries_metrics(portfolio_history)
    portfolio_history = add_pnl_to_history(portfolio_history, positions_with_cash)

    sharpe, sortino = compute_sharpe_sortino(portfolio_history)
    print(f"Sharpe Ratio: {sharpe:.2f}, Sortino Ratio: {sortino:.2f}")

    # ------------------------------------------------------------------
    # 4. Other one-off analytics (top/bottom, flags) –- optional
    # ------------------------------------------------------------------
    latest_date, top_n, bottom_n = get_top_bottom_contributors(positions_with_cash)
    print(f"\nTop contributors for {latest_date.date()}:\n", top_n)
    print(f"Top detractors for {latest_date.date()}:\n", bottom_n)

    latest = positions_with_cash[positions_with_cash['reportDate'] == latest_date]
    latest = add_position_flags(latest)
    print("\nConcentration/UnderDiversified Flags (latest date):")
    print(latest[['symbol', 'WeightInPortfolio',
                  'ConcentrationFlag', 'UnderDiversifiedFlag']])

    # ------------------------------------------------------------------
    # 5. Return BOTH artefacts
    # ------------------------------------------------------------------
    return portfolio_history, positions_with_cash




if __name__ == "__main__":
    # ---- Load positions from Google Sheets ----
    positions_raw = load_gsheet_worksheet("myportfolio", "flex-positions")
    print("\nRaw positions loaded from Google Sheet:")
    print(positions_raw.head())

    # ---- Preprocess positions ----
    positions_summary = preprocess_positions(positions_raw)
    print("\nPreprocessed positions summary (after metrics):")
    print(positions_summary.head())

    # ---- Load account summary (for cash by date) ----
    account_df = load_gsheet_worksheet("myportfolio", "accountsummary")
    print("\nAccount summary loaded from Google Sheet:")
    print(account_df.head())

    # ---- Add a cash row for the latest position date ----
    #positions_with_cash = add_cash_for_latest_position_date(positions_summary, account_df, tag='TotalCashValue')
    positions_with_cash = add_cash_rows_all_dates(positions_summary, account_df, tag='TotalCashValue')

    # print("\nPositions with cash for latest date added:")
    # print(positions_with_cash)
    print("\nPositions with cash (all dates):")

    print(positions_with_cash)

    print("\nSanity check - sample positions with cash and weights:")
    print(positions_with_cash[['symbol', 'positionValue', 'reportDate', 'position_key', 'WeightInPortfolio']])



    # ---- Run main analytics ----
    print("\nRunning main analytics pipeline...\n")
    #main(positions_with_cash)
    #portfolio_history = main(positions_with_cash)
    portfolio_history, positions_with_cash = main(positions_with_cash)



    # Assume positions_with_cash is your DataFrame for today
    append_positions_to_gsheet(positions_with_cash, sheet_name="myportfolio", history_sheet_name="updated-positions")

    summary_df = build_portfolio_summary_table(portfolio_history)
    upsert_portfolio_summary_to_gsheet(summary_df)

