import requests
import time
import pandas as pd
import xml.etree.ElementTree as ET
from io import BytesIO
import nest_asyncio
nest_asyncio.apply()
from ib_insync import *
import pandas as pd
from datetime import datetime
import gspread
from gspread_dataframe import set_with_dataframe
import json
import os
import gspread_dataframe as gd


import os

# If using dotenv:
from dotenv import load_dotenv
load_dotenv()  # This will load .env if present

# Then get from environment
FLEX_TOKEN = os.environ.get("IBKR_FLEX_TOKEN")
QUERY_ID = os.environ.get("IBKR_FLEX_QUERY_ID")
QUERY_ID_POSITIONS = os.environ.get("IBKR_FLEX_QUERY_POSITION_ID")

if not FLEX_TOKEN or not QUERY_ID:
    raise Exception("IBKR_FLEX_TOKEN or IBKR_FLEX_QUERY_ID not set in environment.")


def request_flex_report(token, query_id, version=3):
    """
    Request an IBKR Flex report and return the reference code and pickup url.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
    params = {"t": token, "q": query_id, "v": str(version)}
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"SendRequest error: {resp.status_code}")
    root = ET.fromstring(resp.content)
    status = root.findtext("Status")
    if status != "Success":
        raise Exception(f"Flex request failed: {status}, details: {resp.text}")
    reference_code = root.findtext("ReferenceCode")
    pickup_url = root.findtext("Url")
    return reference_code, pickup_url

def download_flex_statement(token, reference_code, pickup_url, xml_filename=None, version=3, wait=4):
    """
    Download the Flex statement XML file using the reference code and pickup URL.
    If xml_filename is provided, saves to file. Always returns the XML content as bytes.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"t": token, "q": reference_code, "v": str(version)}
    time.sleep(wait)
    resp = requests.get(pickup_url, params=params, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"GetStatement error: {resp.status_code}")
    xml_content = resp.content
    if xml_filename:
        with open(xml_filename, "wb") as f:
            f.write(xml_content)
    return xml_content

def flex_xml_to_trades_df(xml_source, trade_tag='Trade'):
    """
    Parse Flex XML from a filename, XML string, or bytes and return a DataFrame of trades.
    """
    if isinstance(xml_source, str):
        try:
            # Try as file path first
            tree = ET.parse(xml_source)
        except (FileNotFoundError, OSError):
            # If fails, treat as XML string
            tree = ET.ElementTree(ET.fromstring(xml_source))
    elif isinstance(xml_source, bytes):
        tree = ET.parse(BytesIO(xml_source))
    else:
        raise ValueError("xml_source must be filename (str), xml string (str), or bytes.")
    root = tree.getroot()
    trades = [trade.attrib for trade in root.iter(trade_tag)]
    return pd.DataFrame(trades)


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
    

def get_or_create_worksheet(sh, title, rows=1000, cols=26):
    """Get worksheet by name, or create if it does not exist."""
    try:
        return sh.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        print(f"Creating new worksheet: {title}")
        return sh.add_worksheet(title=title, rows=rows, cols=cols)
    



# def append_new_trades_to_worksheet(worksheet, df, unique_col='tradeID'):
#     """
#     Appends only new trades (by unique_col) to the worksheet.
#     Handles column name whitespace, type, and shows debug info if things don't match.
#     """
#     import gspread_dataframe as gd

#     if df.empty:
#         print("No data to append")
#         return

#     # Standardize columns in new DataFrame
#     df = df.copy()
#     df.columns = [c.strip() for c in df.columns]

#     # Read existing data (if any) and standardize columns
#     try:
#         existing = gd.get_as_dataframe(worksheet)
#         existing = existing.dropna(how='all').dropna(axis=1, how='all')
#         existing.columns = [str(c).strip() for c in existing.columns]
#         print("Existing worksheet columns:", list(existing.columns))
#     except Exception as e:
#         print(f"Could not read existing data: {e}")
#         existing = pd.DataFrame()

#     # Ensure the unique column exists in the DataFrame
#     if unique_col not in df.columns:
#         raise ValueError(f"'{unique_col}' column missing in the new DataFrame.")

#     # If sheet is empty, or column is missing in existing, write all with header
#     if existing.empty or unique_col not in existing.columns:
#         print("Writing new data with headers")
#         gd.set_with_dataframe(worksheet, df, include_column_header=True)
#         print(f"Wrote {len(df)} new rows.")
#         return

#     # Print some samples for debugging
#     print("Sample tradeIDs in new df:", df[unique_col].head().tolist())
#     print("Sample tradeIDs in existing sheet:", existing[unique_col].head().tolist())

#     def to_clean_str(x):
#         if pd.isnull(x):
#             return ''
#         s = str(x).strip()
#         if s.endswith('.0'):
#             s = s[:-2]
#         return s

#     # Standardize unique_col values for comparison
#     old_ids = set(existing[unique_col].apply(to_clean_str))
#     mask_new = ~df[unique_col].apply(to_clean_str).isin(old_ids)
#     df_new = df[mask_new]

#     print(f"{len(df_new)} truly new rows found to append")
#     if df_new.empty:
#         print("No new trades to append.")
#         return

#     # Find where to start appending (after last row)
#     start_row = len(existing) + 2  # +1 for header, +1 for 1-based indexing
#     gd.set_with_dataframe(worksheet, df_new, row=start_row, include_column_header=False)
#     print(f"Appended {len(df_new)} new rows.")



def generic_append_new_rows_to_worksheet(worksheet, df, unique_cols):
    """
    Appends only new rows (by composite unique_cols) to the worksheet.
    :param worksheet: gspread worksheet object
    :param df: new data as pandas DataFrame
    :param unique_cols: list of column names forming the unique key
    """
    import gspread_dataframe as gd

    if df.empty:
        print("No data to append")
        return

    # Standardize columns in new DataFrame
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Read existing data (if any) and standardize columns
    try:
        existing = gd.get_as_dataframe(worksheet)
        existing = existing.dropna(how='all').dropna(axis=1, how='all')
        existing.columns = [str(c).strip() for c in existing.columns]
    except Exception as e:
        print(f"Could not read existing data: {e}")
        existing = pd.DataFrame()

    # Ensure the unique columns exist
    for col in unique_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' column missing in the new DataFrame.")

    if existing.empty or any(col not in existing.columns for col in unique_cols):
        print("Writing all data with headers (sheet empty or columns missing)")
        gd.set_with_dataframe(worksheet, df, include_column_header=True)
        print(f"Wrote {len(df)} new rows.")
        return

    # Create unique key columns for comparison
    def make_key(df):
        return df[unique_cols].astype(str).agg('|'.join, axis=1)

    df['_key'] = make_key(df)
    existing['_key'] = make_key(existing)

    mask_new = ~df['_key'].isin(existing['_key'])
    df_new = df[mask_new].drop(columns=['_key'])

    print(f"{len(df_new)} truly new rows to append")
    if df_new.empty:
        print("No new data to append.")
        return

    start_row = len(existing) + 2  # 1 for header, 1 for 1-based index
    gd.set_with_dataframe(worksheet, df_new, row=start_row, include_column_header=False)
    print(f"Appended {len(df_new)} new rows.")

def generic_append_new_rows_to_worksheet(worksheet, df, unique_cols):
    import gspread_dataframe as gd

    if df.empty:
        print("No data to append")
        return

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    try:
        existing = gd.get_as_dataframe(worksheet)
        existing = existing.dropna(how='all').dropna(axis=1, how='all')
        existing.columns = [str(c).strip() for c in existing.columns]
    except Exception as e:
        print(f"Could not read existing data: {e}")
        existing = pd.DataFrame()

    for col in unique_cols:
        if col not in df.columns:
            raise ValueError(f"'{col}' column missing in the new DataFrame.")

    # Normalize all key fields (remove .0, strip spaces, etc)
    def clean_str(x):
        s = str(x).strip()
        return s[:-2] if s.endswith('.0') else s

    # def make_key(df):
    #     return df[unique_cols].map(clean_str).agg('|'.join, axis=1)
    def make_key(df):
    # Clean every column in unique_cols
        for col in unique_cols:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        return df[unique_cols].agg('|'.join, axis=1)


    df['_key'] = make_key(df)
    if not existing.empty and all(col in existing.columns for col in unique_cols):
        existing['_key'] = make_key(existing)
    else:
        existing['_key'] = []

    if existing.empty or any(col not in existing.columns for col in unique_cols):
        print("Writing all data with headers (sheet empty or columns missing)")
        gd.set_with_dataframe(worksheet, df.drop(columns=['_key']), include_column_header=True)
        print(f"Wrote {len(df)} new rows.")
        return

    mask_new = ~df['_key'].isin(existing['_key'])
    df_new = df[mask_new].drop(columns=['_key'])

    print(f"{len(df_new)} truly new rows to append")
    if df_new.empty:
        print("No new data to append.")
        return

    start_row = len(existing) + 2
    gd.set_with_dataframe(worksheet, df_new, row=start_row, include_column_header=False)
    print(f"Appended {len(df_new)} new rows.")



if __name__ == "__main__":
    # --- CONFIG ---

    GOOGLE_SHEET_NAME = "myportfolio"
    TRADES_WORKSHEET_NAME = "flex-trades"
    POSITIONS_WORKSHEET_NAME="flex-positions"

    # # --- Download and parse trades ---
    # ref_code, pickup_url = request_flex_report(FLEX_TOKEN, QUERY_ID)
    # xml_bytes = download_flex_statement(FLEX_TOKEN, ref_code, pickup_url)
    # df_trades = flex_xml_to_trades_df(xml_bytes, trade_tag='Trade')
    # print(f"Downloaded {len(df_trades)} trades.")

    # # --- Upload to Google Sheets ---
    # gc = get_gspread_client()
    # sh = gc.open(GOOGLE_SHEET_NAME)
    # ws = get_or_create_worksheet(sh, WORKSHEET_NAME)
    # append_new_trades_to_worksheet(ws, df_trades, unique_col='tradeID')


        # --- Download and parse trades ---
    ref_code, pickup_url = request_flex_report(FLEX_TOKEN, QUERY_ID)
    xml_bytes = download_flex_statement(FLEX_TOKEN, ref_code, pickup_url)
    df_trades = flex_xml_to_trades_df(xml_bytes, trade_tag='Trade')
    print(f"Downloaded {len(df_trades)} trades.")

    # --- Download and parse positions ---
    ref_code_pos, pickup_url_pos = request_flex_report(FLEX_TOKEN, QUERY_ID_POSITIONS)
    xml_bytes_pos = download_flex_statement(FLEX_TOKEN, ref_code_pos, pickup_url_pos)
    df_positions = flex_xml_to_trades_df(xml_bytes_pos, trade_tag='OpenPosition')
    print(f"Downloaded {len(df_positions)} positions.")

    # --- Upload to Google Sheets ---
    gc = get_gspread_client()
    sh = gc.open(GOOGLE_SHEET_NAME)

    # Trades tab
    ws_trades = get_or_create_worksheet(sh, TRADES_WORKSHEET_NAME)
    #generic_append_new_rows_to_worksheet(ws_trades, df_trades, unique_col='tradeID')
    generic_append_new_rows_to_worksheet(ws_trades, df_trades, unique_cols=['tradeID'])


    # Positions tab: choose your uniqueness key!
    # Option 1: By contract id (conid) + reportDate for daily snapshots
    df_positions['position_key'] = df_positions['accountId'] + "_" + df_positions['conid'] + '_' + df_positions['reportDate']
    ws_positions = get_or_create_worksheet(sh, POSITIONS_WORKSHEET_NAME)
    #generic_append_new_rows_to_worksheet(ws_positions, df_positions, unique_col='position_key')
    generic_append_new_rows_to_worksheet(ws_positions, df_positions, unique_cols=['position_key'])


    print("----Done----")

