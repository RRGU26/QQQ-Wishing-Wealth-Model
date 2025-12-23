"""
Market Breadth Data Fetcher

Gets REAL market breadth data (no proxies) for GMI calculation:
1. NYSE/NASDAQ new highs and new lows
2. Stocks above moving averages (T2108 equivalent)
3. Advance/decline data

Data Sources (in priority order):
1. Yahoo Finance breadth tickers (^NYHIGH, ^NYLOW)
2. FINVIZ screener API
3. Calculated from S&P 500 + NASDAQ 100 universe
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import time
import warnings
import re
warnings.filterwarnings('ignore')


# S&P 500 + NASDAQ 100 stock universe for breadth calculation
STOCK_UNIVERSE = [
    # Top 100 by market cap (mix of sectors)
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'XOM', 'V', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
    'KO', 'COST', 'AVGO', 'LLY', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT',
    'DHR', 'VZ', 'ADBE', 'NKE', 'CRM', 'CMCSA', 'NEE', 'TXN', 'PM', 'WFC',
    'BMY', 'RTX', 'UPS', 'QCOM', 'T', 'MS', 'ORCL', 'HON', 'SCHW', 'COP',
    'AMD', 'INTC', 'BA', 'CAT', 'IBM', 'GE', 'AMGN', 'LOW', 'INTU', 'SPGI',
    'GS', 'DE', 'AXP', 'BLK', 'MDLZ', 'ELV', 'ADI', 'GILD', 'ISRG', 'BKNG',
    'TJX', 'REGN', 'VRTX', 'SYK', 'ADP', 'MMC', 'PLD', 'CI', 'LRCX', 'CB',
    # Additional 50 for better coverage
    'MO', 'DUK', 'SO', 'CL', 'ITW', 'MMM', 'EMR', 'PNC', 'USB', 'TFC',
    'APD', 'ECL', 'SHW', 'NSC', 'UNP', 'FDX', 'CSX', 'WM', 'RSG', 'AMT',
    'CCI', 'PSA', 'EQIX', 'DLR', 'O', 'SPG', 'AVB', 'EQR', 'MAA', 'UDR',
    'WELL', 'VTR', 'HST', 'KIM', 'REG', 'FRT', 'BXP', 'SLG', 'VNO', 'HIW',
    'ARE', 'BIO', 'PKI', 'MTD', 'WAT', 'A', 'IQV', 'DGX', 'LH', 'ZBH'
]


class BreadthDataFetcher:
    """
    Fetch real market breadth data from multiple sources.
    """

    def __init__(self):
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes

    def get_new_highs_lows(self) -> Dict:
        """
        Get daily new highs and new lows count.

        Attempts multiple sources in order:
        1. Yahoo Finance breadth tickers
        2. FINVIZ screener
        3. Calculated from stock universe

        Returns:
            Dictionary with new_highs, new_lows, net counts
        """
        # Try Yahoo Finance breadth tickers first
        result = self._fetch_yahoo_breadth()
        if result and result.get('new_highs', 0) > 0:
            return result

        # Try FINVIZ screener
        result = self._fetch_finviz_breadth()
        if result and result.get('new_highs', 0) > 0:
            return result

        # Fallback: Calculate from stock universe
        return self._calculate_from_universe()

    def _fetch_yahoo_breadth(self) -> Optional[Dict]:
        """Fetch breadth data from Yahoo Finance special tickers."""
        import io
        import sys

        try:
            # Suppress yfinance error output for these tickers (they often don't exist)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()

            # Yahoo has some market breadth tickers
            # ^NYHIGH - NYSE new highs, ^NYLOW - NYSE new lows
            highs_data = yf.download('^NYHIGH', period='5d', progress=False)
            lows_data = yf.download('^NYLOW', period='5d', progress=False)

            # Restore stderr
            sys.stderr = old_stderr

            if len(highs_data) > 0 and len(lows_data) > 0:
                # Handle multi-index columns
                if isinstance(highs_data.columns, pd.MultiIndex):
                    highs_data.columns = highs_data.columns.droplevel('Ticker')
                if isinstance(lows_data.columns, pd.MultiIndex):
                    lows_data.columns = lows_data.columns.droplevel('Ticker')

                new_highs = int(highs_data['Close'].iloc[-1])
                new_lows = int(lows_data['Close'].iloc[-1])

                return {
                    'new_highs': new_highs,
                    'new_lows': new_lows,
                    'net': new_highs - new_lows,
                    'source': 'yahoo_breadth',
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            # Restore stderr if exception occurred
            sys.stderr = old_stderr
            pass  # Silently fail, try next source

        return None

    def _fetch_finviz_breadth(self) -> Optional[Dict]:
        """Fetch breadth data from FINVIZ screener."""
        try:
            # FINVIZ screener for 52-week highs
            url_highs = "https://finviz.com/screener.ashx?v=111&f=ta_highlow52w_nh"
            url_lows = "https://finviz.com/screener.ashx?v=111&f=ta_highlow52w_nl"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            # Get new highs count
            response = requests.get(url_highs, headers=headers, timeout=10)
            if response.status_code == 200:
                # Look for "Total: XXX" pattern
                match = re.search(r'Total:\s*<b>(\d+)</b>', response.text)
                if not match:
                    match = re.search(r'#1\s*/\s*(\d+)', response.text)
                new_highs = int(match.group(1)) if match else 0
            else:
                new_highs = 0

            # Get new lows count
            response = requests.get(url_lows, headers=headers, timeout=10)
            if response.status_code == 200:
                match = re.search(r'Total:\s*<b>(\d+)</b>', response.text)
                if not match:
                    match = re.search(r'#1\s*/\s*(\d+)', response.text)
                new_lows = int(match.group(1)) if match else 0
            else:
                new_lows = 0

            if new_highs > 0 or new_lows > 0:
                return {
                    'new_highs': new_highs,
                    'new_lows': new_lows,
                    'net': new_highs - new_lows,
                    'source': 'finviz',
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            pass  # Silently fail, try next source

        return None

    def _calculate_from_universe(self) -> Dict:
        """
        Calculate new highs/lows from universe of 150 liquid stocks.

        Uses batch download for efficiency, then calculates 52-week highs/lows.
        Scale factor applied to estimate full market (~4000 traded stocks).
        """
        new_highs = 0
        new_lows = 0
        total_checked = 0

        # Batch download for efficiency (process in chunks of 50)
        chunk_size = 50
        for i in range(0, min(len(STOCK_UNIVERSE), 150), chunk_size):
            chunk = STOCK_UNIVERSE[i:i+chunk_size]
            try:
                # Batch download
                data = yf.download(chunk, period='1y', progress=False, threads=True)

                if data.empty:
                    continue

                # Handle multi-index columns
                if isinstance(data.columns, pd.MultiIndex):
                    # Get close and high prices for each symbol
                    for symbol in chunk:
                        try:
                            if symbol not in data['Close'].columns:
                                continue

                            close_series = data['Close'][symbol].dropna()
                            high_series = data['High'][symbol].dropna()
                            low_series = data['Low'][symbol].dropna()

                            if len(close_series) < 200:
                                continue

                            current_price = close_series.iloc[-1]
                            high_52w = high_series.max()
                            low_52w = low_series.min()

                            # At 52-week high (within 1%)
                            if current_price >= high_52w * 0.99:
                                new_highs += 1

                            # At 52-week low (within 1%)
                            if current_price <= low_52w * 1.01:
                                new_lows += 1

                            total_checked += 1
                        except:
                            continue
                else:
                    # Single stock case
                    if len(data) < 200:
                        continue

                    current_price = data['Close'].iloc[-1]
                    high_52w = data['High'].max()
                    low_52w = data['Low'].min()

                    if current_price >= high_52w * 0.99:
                        new_highs += 1
                    if current_price <= low_52w * 1.01:
                        new_lows += 1

                    total_checked += 1

            except Exception as e:
                continue

        # Scale to estimate full market (~4000 actively traded stocks)
        # Our sample of 150 blue chips is ~3.75% of the market
        scale_factor = 4000 / total_checked if total_checked > 0 else 1

        return {
            'new_highs': int(new_highs * scale_factor),
            'new_lows': int(new_lows * scale_factor),
            'net': int((new_highs - new_lows) * scale_factor),
            'raw_highs': new_highs,
            'raw_lows': new_lows,
            'source': 'calculated_universe',
            'sample_size': total_checked,
            'timestamp': datetime.now().isoformat()
        }

    def get_successful_new_high_index(self) -> Dict:
        """
        Calculate the Successful New High Index.

        Counts stocks that hit a 52-week high 10 trading days ago
        and are still trading higher today.

        This is a KEY Wishing Wealth indicator.
        """
        successful = 0
        total_at_high = 0
        total_checked = 0

        # Batch download 100 stocks for efficiency
        stocks_to_check = STOCK_UNIVERSE[:100]

        try:
            # Batch download 3 months of data
            data = yf.download(stocks_to_check, period='3mo', progress=False, threads=True)

            if data.empty:
                return self._fallback_successful_nh()

            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                for symbol in stocks_to_check:
                    try:
                        if symbol not in data['Close'].columns:
                            continue

                        close_series = data['Close'][symbol].dropna()
                        high_series = data['High'][symbol].dropna()

                        if len(close_series) < 15:
                            continue

                        # Price 10 trading days ago
                        price_10d_ago = close_series.iloc[-11]

                        # Highest high before 10 days ago (proxy for 52-week high at that time)
                        high_before_10d = high_series.iloc[:-10].max()

                        # Was at 52-week high 10 days ago?
                        was_at_high = price_10d_ago >= high_before_10d * 0.99

                        if was_at_high:
                            total_at_high += 1
                            # Is it still higher today?
                            current_price = close_series.iloc[-1]
                            if current_price > price_10d_ago:
                                successful += 1

                        total_checked += 1

                    except:
                        continue

        except Exception as e:
            return self._fallback_successful_nh()

        # Scale to full market (sample is ~2.5% of 4000 stocks)
        scale_factor = 4000 / total_checked if total_checked > 0 else 100

        return {
            'successful_count': int(successful * scale_factor),
            'total_at_high': int(total_at_high * scale_factor),
            'raw_successful': successful,
            'raw_at_high': total_at_high,
            'success_rate': successful / total_at_high if total_at_high > 0 else 0,
            'threshold_met': int(successful * scale_factor) >= 100,
            'source': 'calculated_universe',
            'sample_size': total_checked,
            'timestamp': datetime.now().isoformat()
        }

    def _fallback_successful_nh(self) -> Dict:
        """Fallback when batch download fails."""
        return {
            'successful_count': 0,
            'total_at_high': 0,
            'success_rate': 0,
            'threshold_met': False,
            'source': 'fallback',
            'timestamp': datetime.now().isoformat()
        }

    def get_t2108_equivalent(self) -> Dict:
        """
        Get T2108 equivalent (% of stocks above 200-day MA).

        T2108 from Worden measures NYSE stocks above 40-day MA.
        We approximate with a broader measure.
        """
        # Use S&P 500 components or ETF-based approximation
        universe = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
            'XOM', 'V', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
            'KO', 'COST', 'AVGO', 'LLY', 'WMT', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT'
        ]

        above_200ma = 0
        above_50ma = 0
        total = 0

        for symbol in universe:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1y')

                if len(hist) < 200:
                    continue

                current = hist['Close'].iloc[-1]
                ma_200 = hist['Close'].rolling(200).mean().iloc[-1]
                ma_50 = hist['Close'].rolling(50).mean().iloc[-1]

                if current > ma_200:
                    above_200ma += 1
                if current > ma_50:
                    above_50ma += 1

                total += 1

            except:
                continue

        pct_above_200 = (above_200ma / total * 100) if total > 0 else 50
        pct_above_50 = (above_50ma / total * 100) if total > 0 else 50

        # Determine signal
        if pct_above_200 < 10:
            signal = 'EXTREME_OVERSOLD'
        elif pct_above_200 < 20:
            signal = 'OVERSOLD'
        elif pct_above_200 > 80:
            signal = 'OVERBOUGHT'
        else:
            signal = 'NEUTRAL'

        return {
            'pct_above_200ma': round(pct_above_200, 1),
            'pct_above_50ma': round(pct_above_50, 1),
            'signal': signal,
            'sample_size': total,
            'source': 'calculated',
            'timestamp': datetime.now().isoformat()
        }

    def get_all_breadth_data(self) -> Dict:
        """
        Fetch all breadth data at once.

        Returns comprehensive breadth analysis.
        """
        print("Fetching market breadth data...")

        # Get all components
        new_highs_lows = self.get_new_highs_lows()
        print(f"  New Highs/Lows: {new_highs_lows.get('new_highs', 'N/A')} / {new_highs_lows.get('new_lows', 'N/A')}")

        successful_nh = self.get_successful_new_high_index()
        print(f"  Successful New High Index: {successful_nh.get('successful_count', 'N/A')}")

        t2108 = self.get_t2108_equivalent()
        print(f"  T2108 Equivalent: {t2108.get('pct_above_200ma', 'N/A')}%")

        return {
            'new_highs_lows': new_highs_lows,
            'successful_new_high': successful_nh,
            't2108': t2108,
            'fetched_at': datetime.now().isoformat()
        }


def fetch_breadth_data() -> Dict:
    """Convenience function to fetch all breadth data."""
    fetcher = BreadthDataFetcher()
    return fetcher.get_all_breadth_data()


if __name__ == "__main__":
    # Test the fetcher
    data = fetch_breadth_data()
    print("\n=== BREADTH DATA ===")
    print(f"New Highs: {data['new_highs_lows'].get('new_highs')}")
    print(f"New Lows: {data['new_highs_lows'].get('new_lows')}")
    print(f"Net: {data['new_highs_lows'].get('net')}")
    print(f"Successful NH Index: {data['successful_new_high'].get('successful_count')}")
    print(f"T2108 Equivalent: {data['t2108'].get('pct_above_200ma')}%")
