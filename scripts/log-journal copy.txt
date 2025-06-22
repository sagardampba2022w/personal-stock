import nest_asyncio
nest_asyncio.apply()
from ib_insync import *
import pandas as pd
from datetime import datetime
import gspread
from gspread_dataframe import set_with_dataframe

# Define your shortlist of key tags
CORE_TAGS = [
    'NetLiquidation',
    'TotalCashValue',
    'AvailableFunds',
    'BuyingPower',
    'UnrealizedPnL',
    'RealizedPnL'
]

def extract_ib_portfolio(ib):
    portfolio = ib.portfolio()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = []
    for pos in portfolio:
        row = {
            'DateTime': now,
            'Symbol': pos.contract.symbol,
            'Exchange': pos.contract.primaryExchange,
            'Currency': pos.contract.currency,
            'Position': pos.position,
            'Market Price': pos.marketPrice,
            'Market Value': pos.marketValue,
            'Average Cost': pos.averageCost,
            'Unrealized PnL': pos.unrealizedPNL,
            'Realized PnL': pos.realizedPNL,
            'Account': pos.account
        }
        data.append(row)
    return pd.DataFrame(data)

def extract_core_account_summary(ib):
    summary = ib.accountSummary()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = []
    for item in summary:
        if item.tag in CORE_TAGS:
            row = {
                'DateTime': now,
                'Tag': item.tag,
                'Value': item.value,
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
        return sh.add_worksheet(title=title, rows=rows, cols=cols)
    
def append_df_to_worksheet(worksheet, df):
    """Appends df to the end of a Google Sheet tab (worksheet)."""
    import gspread_dataframe as gd

    # Read existing data (if any)
    try:
        existing = gd.get_as_dataframe(worksheet)
        # Drop entirely empty columns/rows (gspread quirk)
        existing = existing.dropna(how='all').dropna(axis=1, how='all')
    except Exception:
        existing = pd.DataFrame()

    # If sheet is empty, write with header
    if existing.empty:
        gd.set_with_dataframe(worksheet, df, include_column_header=True)
    else:
        # Append without header (start after last row)
        start_row = len(existing) + 2  # +1 for 1-based, +1 for header
        gd.set_with_dataframe(worksheet, df, row=start_row, include_column_header=False)



def main():
    # Connect to IBKR
    ib = IB()
    ib.connect('127.0.0.1', 7496, clientId=2)

    # Extract data
    portfolio_df = extract_ib_portfolio(ib)
    account_df = extract_core_account_summary(ib)

    print('Core portfolio and account summary extracted.')

    ib.disconnect()

    # Connect to Google Sheets
    gc = gspread.service_account(filename='gdrive-creds.json')
    sh = gc.open('myportfolio')

    # Portfolio tab
    portfolio_ws = get_or_create_worksheet(sh, 'portfolio')
    append_df_to_worksheet(portfolio_ws, portfolio_df)

    #portfolio_ws.clear()
    #set_with_dataframe(portfolio_ws, portfolio_df)

    # Account summary tab
    account_ws = get_or_create_worksheet(sh, 'accountsummary')
    append_df_to_worksheet(account_ws, account_df)

    #account_ws.clear()
    #set_with_dataframe(account_ws, account_df)

    print("Portfolio and account summary uploaded to Google Sheets!")


if __name__ == "__main__":
    main()
