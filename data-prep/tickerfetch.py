"""
Ticker Fetcher Module
====================

Fetches S&P 500 and NASDAQ 100 ticker symbols from Wikipedia
and creates a combined, deduplicated universe.

Usage:
    from ticker_fetcher import get_combined_universe, save_tickers
    tickers = get_combined_universe()
    save_tickers(tickers, "../data/tickers.csv")
"""

import pandas as pd
import warnings
from typing import List, Dict, Optional
from datetime import datetime

def fetch_sp500_tickers() -> List[str]:
    """
    Fetch S&P 500 ticker symbols from Wikipedia
    
    Returns:
    --------
    List[str]: List of S&P 500 ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]  # First table contains current constituents
        
        # Clean ticker symbols (remove periods, handle special cases)
        tickers = df['Symbol'].astype(str).str.strip().tolist()
        
        # Handle common issues with ticker symbols
        clean_tickers = []
        for ticker in tickers:
            # Replace periods with hyphens for yfinance compatibility
            clean_ticker = ticker.replace('.', '-')
            clean_tickers.append(clean_ticker)
        
        print(f"Successfully fetched {len(clean_tickers)} S&P 500 tickers")
        return clean_tickers
        
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

# In your tickerfetch.py file, modify the functions:

def fetch_sp500_tickers() -> List[str]:
    try:
        import requests
        from io import StringIO
        
        # Add headers to avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Use requests + pandas instead of direct pd.read_html
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        tickers = df['Symbol'].astype(str).str.strip().tolist()
        clean_tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"Successfully fetched {len(clean_tickers)} S&P 500 tickers")
        return clean_tickers
        
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []
    
def fetch_nasdaq100_tickers() -> List[str]:
    """
    Fetch NASDAQ 100 ticker symbols from Wikipedia
    
    Returns:
    --------
    List[str]: List of NASDAQ 100 ticker symbols
    """
    try:
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        
        # Search for table with ticker/symbol column
        for i, df in enumerate(tables):
            # Safely lowercase column names
            lower_cols = [str(col).lower() for col in df.columns]
            
            if "ticker" in lower_cols or "symbol" in lower_cols:
                # Get the actual column name
                if "ticker" in lower_cols:
                    orig_col = df.columns[lower_cols.index("ticker")]
                else:
                    orig_col = df.columns[lower_cols.index("symbol")]
                
                print(f"Using table #{i} with column '{orig_col}'")
                
                # Clean and return tickers
                tickers = df[orig_col].astype(str).str.strip().tolist()
                
                # Filter out any invalid entries
                valid_tickers = [t for t in tickers if t and t != 'nan' and len(t) <= 5]
                
                print(f"Successfully fetched {len(valid_tickers)} NASDAQ 100 tickers")
                return valid_tickers
        
        raise ValueError("Could not find a 'Ticker' or 'Symbol' column in any table")
        
    except Exception as e:
        print(f"Error fetching NASDAQ 100 tickers: {e}")
        return []

import requests
from io import StringIO
import time

def fetch_nasdaq100_tickers() -> List[str]:
    """
    Fetch NASDAQ 100 ticker symbols with improved headers and retry logic
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }
        
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        
        # Add small delay to avoid rate limiting
        time.sleep(1)
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse HTML content
        tables = pd.read_html(StringIO(response.text))
        
        for i, df in enumerate(tables):
            lower_cols = [str(col).lower() for col in df.columns]
            
            if "ticker" in lower_cols or "symbol" in lower_cols:
                if "ticker" in lower_cols:
                    orig_col = df.columns[lower_cols.index("ticker")]
                else:
                    orig_col = df.columns[lower_cols.index("symbol")]
                
                print(f"Using NASDAQ table #{i} with column '{orig_col}'")
                
                tickers = df[orig_col].astype(str).str.strip().tolist()
                
                # Clean and validate tickers
                valid_tickers = []
                for ticker in tickers:
                    if ticker and ticker != 'nan' and len(ticker) <= 6 and ticker.isalpha():
                        valid_tickers.append(ticker)
                
                if len(valid_tickers) > 80:  # NASDAQ 100 should have ~100 stocks
                    print(f"Successfully fetched {len(valid_tickers)} NASDAQ 100 tickers")
                    return valid_tickers
        
        raise ValueError("Could not find valid ticker column with sufficient data")
        
    except Exception as e:
        print(f"Error fetching NASDAQ 100 tickers: {e}")
        return []

def get_combined_universe(include_sp500: bool = True, 
                         include_nasdaq100: bool = True,
                         save_components: bool = False) -> pd.DataFrame:
    """
    Get combined, deduplicated universe of tickers
    
    Parameters:
    -----------
    include_sp500 : bool
        Include S&P 500 tickers
    include_nasdaq100 : bool  
        Include NASDAQ 100 tickers
    save_components : bool
        Save individual components to see overlap
        
    Returns:
    --------
    pd.DataFrame: Combined ticker universe with metadata
    """
    
    all_tickers = []
    sources = []
    
    if include_sp500:
        sp500_tickers = fetch_sp500_tickers()
        for ticker in sp500_tickers:
            all_tickers.append(ticker)
            sources.append("S&P 500")
    
    if include_nasdaq100:
        nasdaq100_tickers = fetch_nasdaq100_tickers()
        for ticker in nasdaq100_tickers:
            all_tickers.append(ticker)
            sources.append("NASDAQ 100")
    
    if not all_tickers:
        print("No tickers fetched!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame({
        'Ticker': all_tickers,
        'Source': sources
    })
    
    # Add overlap information before deduplication
    ticker_counts = df['Ticker'].value_counts()
    df['In_Both'] = df['Ticker'].map(ticker_counts > 1)
    
    # Deduplicate while preserving source info
    unique_df = (df.groupby('Ticker')
                   .agg({
                       'Source': lambda x: ' + '.join(sorted(set(x))),
                       'In_Both': 'first'
                   })
                   .reset_index())
    
    # Add metadata
    unique_df['Fetched_Date'] = datetime.now().strftime('%Y-%m-%d')
    unique_df['Total_Count'] = len(unique_df)
    
    print(f"\nCombined Universe Summary:")
    print(f"Total unique tickers: {len(unique_df)}")
    print(f"S&P 500 only: {len(unique_df[unique_df['Source'] == 'S&P 500'])}")
    print(f"NASDAQ 100 only: {len(unique_df[unique_df['Source'] == 'NASDAQ 100'])}")
    print(f"In both indices: {len(unique_df[unique_df['In_Both']])}")
    
    if save_components:
        # Save the component analysis
        import os
        os.makedirs("../data", exist_ok=True)
        
        overlap_analysis = {
            'total_sp500': len(sp500_tickers) if include_sp500 else 0,
            'total_nasdaq100': len(nasdaq100_tickers) if include_nasdaq100 else 0,
            'total_combined': len(all_tickers),
            'unique_tickers': len(unique_df),
            'overlap_count': len(unique_df[unique_df['In_Both']]),
            'fetch_date': datetime.now().isoformat()
        }
        
        pd.Series(overlap_analysis).to_csv("../data/ticker_analysis.csv")
        print("Saved analysis to ../data/ticker_analysis.csv")
    
    return unique_df

def save_tickers(ticker_df: pd.DataFrame, filepath: str = None) -> str:
    """
    Save ticker universe to file
    
    Parameters:
    -----------
    ticker_df : pd.DataFrame
        DataFrame with ticker universe
    filepath : str, optional
        Custom file path, defaults to ../data/tickers_YYYYMMDD.csv
        
    Returns:
    --------
    str: Path where file was saved
    """
    
    if filepath is None:
        import os
        os.makedirs("../data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        filepath = f"../data/tickers_{timestamp}.csv"
    
    ticker_df.to_csv(filepath, index=False)
    print(f"Saved {len(ticker_df)} tickers to {filepath}")
    
    return filepath

def load_tickers(filepath: str) -> pd.DataFrame:
    """
    Load previously saved ticker universe
    
    Parameters:
    -----------
    filepath : str
        Path to saved ticker file
        
    Returns:
    --------
    pd.DataFrame: Loaded ticker universe
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} tickers from {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()

# Main execution
if __name__ == "__main__":
    # Fetch combined universe
    universe = get_combined_universe(
        include_sp500=True,
        include_nasdaq100=True, 
        save_components=True
    )
    
    if not universe.empty:
        # Save to file
        saved_path = save_tickers(universe)
        
        # Show sample
        print(f"\nSample tickers:")
        print(universe[['Ticker', 'Source', 'In_Both']].head(10))
        
        # Show tickers in both indices
        both_indices = universe[universe['In_Both']]
        print(f"\nTickers in both indices ({len(both_indices)}):")
        print(both_indices['Ticker'].tolist()[:10], "..." if len(both_indices) > 10 else "")
    
    else:
        print("Failed to fetch any tickers!")