import nest_asyncio
nest_asyncio.apply()
from ib_insync import *
import pandas as pd
from datetime import datetime
import gspread
from gspread_dataframe import set_with_dataframe
import json
import os

# Define your shortlist of key tags
CORE_TAGS = [
    'NetLiquidation',
    'TotalCashValue', 
    'AvailableFunds',
    'BuyingPower',
    'UnrealizedPnL',
    'RealizedPnL',
    'GrossPositionValue',  # Added useful tag
    'EquityWithLoanValue'  # Added useful tag
]

def get_gspread_client():
    """Get gspread client using either file or environment variable"""
    gdrive_creds_json = os.getenv("GDRIVE_CREDS_JSON")
    
    if gdrive_creds_json:
        print("Using environment variable for credentials")
        try:
            creds_dict = json.loads(gdrive_creds_json)
            return gspread.service_account_from_dict(creds_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing GDRIVE_CREDS_JSON: {e}")
            raise
    else:
        print("Using file for credentials")
        return gspread.service_account(filename='gdrive-creds.json')

def extract_ib_portfolio(ib):
    """Extract portfolio positions with better error handling"""
    portfolio = ib.portfolio()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = []
    
    for pos in portfolio:
        # Only log positions that exist (non-zero)
        if abs(pos.position) > 0:
            row = {
                'DateTime': now,
                'Symbol': pos.contract.symbol,
                'SecType': pos.contract.secType,  # Added security type
                'Exchange': pos.contract.primaryExchange,
                'Currency': pos.contract.currency,
                'Position': pos.position,
                'Market Price': pos.marketPrice if pos.marketPrice and pos.marketPrice > 0 else None,
                'Market Value': pos.marketValue,
                'Average Cost': pos.averageCost,
                'Unrealized PnL': pos.unrealizedPNL,
                'Realized PnL': pos.realizedPNL,
                'Account': pos.account
            }
            data.append(row)
    
    return pd.DataFrame(data)

def extract_core_account_summary(ib):
    """Extract account summary with better data handling"""
    summary = ib.accountSummary()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = []
    
    for item in summary:
        if item.tag in CORE_TAGS:
            # Convert to float if possible, otherwise keep as string
            try:
                value = float(item.value) if item.value else 0.0
            except (ValueError, TypeError):
                value = item.value
                
            row = {
                'DateTime': now,
                'Tag': item.tag,
                'Value': value,
                'Currency': item.currency,
                'Account': item.account
            }
            data.append(row)
    
    return pd.DataFrame(data)

def get_or_create_worksheet(sh, title, rows=1000, cols=26):
    """Get worksheet by name, or create if it does not exist."""
    try:
        return sh.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Creating new worksheet: {title}")
        return sh.add_worksheet(title=title, rows=rows, cols=cols)
    
def append_df_to_worksheet(worksheet, df):
    """Appends df to the end of a Google Sheet tab (worksheet)."""
    import gspread_dataframe as gd

    if df.empty:
        print("No data to append")
        return

    # Read existing data (if any)
    try:
        existing = gd.get_as_dataframe(worksheet)
        # Drop entirely empty columns/rows (gspread quirk)
        existing = existing.dropna(how='all').dropna(axis=1, how='all')
    except Exception as e:
        print(f"Could not read existing data: {e}")
        existing = pd.DataFrame()

    # If sheet is empty, write with header
    if existing.empty:
        print("Writing new data with headers")
        gd.set_with_dataframe(worksheet, df, include_column_header=True)
    else:
        # Append without header (start after last row)
        start_row = len(existing) + 2  # +1 for 1-based, +1 for header
        print(f"Appending {len(df)} rows starting at row {start_row}")
        gd.set_with_dataframe(worksheet, df, row=start_row, include_column_header=False)

def connect_to_ib(host='127.0.0.1', port=7496, client_id=2, timeout=10):
    """Connect to IB with better error handling"""
    ib = IB()
    try:
        print(f"Connecting to IB at {host}:{port}...")
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        print("Connected successfully!")
        return ib
    except Exception as e:
        print(f"Failed to connect to IB: {e}")
        raise

def main():
    ib = None
    try:
        # Connect to IBKR
        ib = connect_to_ib()
        
        # Wait a moment for data to be ready
        ib.sleep(2)
        
        # Extract data
        print("Extracting portfolio data...")
        portfolio_df = extract_ib_portfolio(ib)
        print(f"Found {len(portfolio_df)} positions")
        
        print("Extracting account summary...")
        account_df = extract_core_account_summary(ib)
        print(f"Found {len(account_df)} account metrics")

        # Connect to Google Sheets
        print("Connecting to Google Sheets...")
        gc = get_gspread_client()
        sh = gc.open('myportfolio')

        # Portfolio tab
        if not portfolio_df.empty:
            portfolio_ws = get_or_create_worksheet(sh, 'portfolio')
            append_df_to_worksheet(portfolio_ws, portfolio_df)
            print("Portfolio data uploaded!")
        else:
            print("No portfolio positions to upload")

        # Account summary tab
        if not account_df.empty:
            account_ws = get_or_create_worksheet(sh, 'accountsummary')
            append_df_to_worksheet(account_ws, account_df)
            print("Account summary uploaded!")
        else:
            print("No account data to upload")

        print("✅ Data logging completed successfully!")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise
    finally:
        # Always disconnect
        if ib and ib.isConnected():
            ib.disconnect()
            print("Disconnected from IB")

if __name__ == "__main__":
    main()