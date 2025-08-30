import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe
import yfinance as yf
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def get_gspread_client():
    """Get gspread client using either file or environment variable"""
    gdrive_creds_json = os.getenv("GDRIVE_CREDS_JSON")
    
    if gdrive_creds_json:
        # Use environment variable (for GitHub Actions)
        print("Using environment variable for credentials")
        try:
            creds_dict = json.loads(gdrive_creds_json)
            return gspread.service_account_from_dict(creds_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing GDRIVE_CREDS_JSON: {e}")
            raise
    else:
        # Use file (for local development)
        print("Using file for credentials")
        return gspread.service_account(filename='gdrive-creds.json')


def get_open_positions_from_gsheet(sheet_name='myportfolio', worksheet_name='portfolio'):
    gc = get_gspread_client()
    #gc = gspread.service_account(filename='gdrive-creds.json')
    sh = gc.open(sheet_name)
    ws = sh.worksheet(worksheet_name)
    df = get_as_dataframe(ws)

    # If DataFrame is empty or has no 'Symbol' column, return empty DataFrame
    if df.empty or 'Symbol' not in df.columns:
        print("Portfolio sheet is empty or missing the 'Symbol' column.")
        return pd.DataFrame()  # Return empty DataFrame

    df = df.dropna(subset=['Symbol'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['Symbol', 'DateTime'])
    latest_df = df.groupby('Symbol').tail(1).reset_index(drop=True)
    return latest_df

    


def get_latest_prices(symbols):
    price_dict = {}
    price_date_dict = {}
    for symbol in symbols:
        try:
            data = yf.Ticker(symbol)
            latest = data.history(period="1d")['Close']
            if not latest.empty:
                price_dict[symbol] = latest.iloc[-1]
                price_date_dict[symbol] = latest.index[-1].strftime('%Y-%m-%d')
            else:
                price_dict[symbol] = None
                price_date_dict[symbol] = None
        except Exception as e:
            price_dict[symbol] = None
            price_date_dict[symbol] = None
    return price_dict, price_date_dict


# def calculate_yfinance_pnl(df, price_dict):
#     results = []
#     for _, row in df.iterrows():
#         symbol = row['Symbol']
#         qty = float(row['Position'])
#         avg_cost = float(row['Average Cost'])
#         last_price = price_dict.get(symbol)
#         if last_price is not None:
#             unrealized = (last_price - avg_cost) * qty
#             pct = 100 * (last_price - avg_cost) / avg_cost if avg_cost != 0 else 0
#             results.append({
#                 'Symbol': symbol,
#                 'Qty': qty,
#                 'Avg Cost': avg_cost,
#                 'Last Price': last_price,
#                 'Unrealized PnL': unrealized,
#                 'PnL %': pct
#             })
#     return pd.DataFrame(results)


def calculate_yfinance_pnl(df, price_dict):
    results = []
    for _, row in df.iterrows():
        symbol = row['Symbol']
        formatted_symbol = row['FormattedSymbol']
        qty = float(row['Position'])
        avg_cost = float(row['Average Cost'])
        last_price = price_dict.get(formatted_symbol)
        if last_price is not None:
            unrealized = (last_price - avg_cost) * qty
            pct = 100 * (last_price - avg_cost) / avg_cost if avg_cost != 0 else 0
            results.append({
                'Symbol': symbol,
                'Qty': qty,
                'Avg Cost': avg_cost,
                'Last Price': last_price,
                'Unrealized PnL': unrealized,
                'PnL %': pct
            })
    return pd.DataFrame(results)



def summarize_yfinance_portfolio(pnl_df,price_date):
    summary_lines = []
    summary_lines.append("📊 *PORTFOLIO SUMMARY*")
    summary_lines.append(f"📅 _Prices as of: {price_date}_\n")
    summary_lines.append("`Symbol  | Qty    | Cost   | Price  | P&L    | %`")
    summary_lines.append("`" + "-" * 37 + "`")
    for _, row in pnl_df.iterrows():
        pnl_emoji = "🟢" if row['Unrealized PnL'] > 0 else "🔴" if row['Unrealized PnL'] < 0 else "⚪"
        summary_lines.append(
            f"`{row['Symbol']:<6} | {row['Qty']:>5.1f} | "
            f"{row['Avg Cost']:>6.0f} | {row['Last Price']:>6.0f} | "
            f"{row['Unrealized PnL']:>6.0f} | {row['PnL %']:>4.1f}%` {pnl_emoji}"
        )
    summary_lines.append("`" + "-" * 37 + "`")
    total_unrlz_pnl = pnl_df['Unrealized PnL'].sum()
    total_invested = (pnl_df['Avg Cost'] * pnl_df['Qty']).sum()
    total_pnl_pct = 100 * total_unrlz_pnl / total_invested if total_invested != 0 else 0
    total_emoji = "🚀" if total_unrlz_pnl > 0 else "📉" if total_unrlz_pnl < 0 else "➖"
    summary_lines.append(f"\n💰 *TOTAL P&L:* `${total_unrlz_pnl:.2f}` {total_emoji}")
    summary_lines.append(f"📈 *TOTAL %:* `{total_pnl_pct:.2f}%`")
    return "\n".join(summary_lines)

def get_account_summary_from_gsheet(sheet_name='myportfolio', worksheet_name='accountsummary'):
    gc = get_gspread_client()
    #gc = gspread.service_account(filename='gdrive-creds.json')
    sh = gc.open(sheet_name)
    ws = sh.worksheet(worksheet_name)
    df = get_as_dataframe(ws)
    if df.empty or 'Tag' not in df.columns:
        print("Account summary sheet is empty or missing the 'Tag' column.")
        return pd.DataFrame()
    df = df.dropna(subset=['Tag'])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['Tag', 'DateTime'])
    latest_df = df.groupby('Tag').tail(1).reset_index(drop=True)
    return latest_df



def summarize_account_summary(df):
    lines = []
    latest_datetime = df['DateTime'].max()
    lines.append("🏦 *ACCOUNT SUMMARY*")
    lines.append(f"📅 _As of: {latest_datetime}_\n")
    lines.append("`Tag                  | Value      | Curr`")
    lines.append("`" + "-" * 37 + "`")
    for _, row in df.iterrows():
        if 'Cash' in row['Tag']:
            emoji = "💵"
        elif 'PnL' in row['Tag']:
            emoji = "📊"
        elif 'Buying' in row['Tag']:
            emoji = "💳"
        else:
            emoji = "📋"
        lines.append(
            f"`{row['Tag']:<20} | {row['Value']:>10} | {row['Currency']:<4}` {emoji}"
        )
    lines.append("`" + "-" * 37 + "`")
    return "\n".join(lines)


def send_telegram_message(token, chat_id, message):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("Telegram message sent!")
    else:
        print("Failed to send:", response.text)


EXCHANGE_SUFFIX = {
    "NASDAQ": "",
    "NYSE": "",
    "AMEX": "",
    "LSE": ".L",
    "LSEETF": ".L",  # <-- support non-standard LSEETF label
    # add others like "ASX": ".AX", "PA": ".PA", etc.
}


def format_symbol(raw_symbol, exchange):
    suffix = EXCHANGE_SUFFIX.get(exchange.upper())
    if suffix is None:
        raise ValueError(f"Unknown exchange: {exchange}")
    return raw_symbol + suffix





if __name__ == "__main__":
    df = get_open_positions_from_gsheet()
    if df.empty:
        print("No portfolio positions found in Google Sheet.")
        pnl_summary_msg = "No positions found in database"
        price_dict, price_date_dict = {}, {}
        pnl_df = pd.DataFrame()
        price_date = "N/A"
    else:
                # Format symbols using Exchange column
        df["FormattedSymbol"] = df.apply(lambda r: format_symbol(r["Symbol"], r["Exchange"]), axis=1)
        unique_syms = df["FormattedSymbol"].unique().tolist()
        price_dict, price_date_dict = get_latest_prices(unique_syms)
        price_dates = list(price_date_dict.values())
        unique_dates = set([d for d in price_dates if d is not None])
        price_date = unique_dates.pop() if len(unique_dates) == 1 else "Mixed"
        pnl_df = calculate_yfinance_pnl(df, price_dict)
        pnl_summary_msg = summarize_yfinance_portfolio(pnl_df, price_date)

    account_df = get_account_summary_from_gsheet()
    if account_df.empty:
        print("No account summary found in Google Sheet.")
        account_summary_msg = "No account summary found in database."
    else:
        account_summary_msg = summarize_account_summary(account_df)

    print(pnl_summary_msg)
    print(account_summary_msg)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, pnl_summary_msg)
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, account_summary_msg)
    else:
        print("Telegram bot token or chat ID missing, not sending message.")


