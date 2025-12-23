"""
Stock Screener - Wishing Wealth Methodology

Scans for technically strong stocks when GMI is GREEN using
patterns from Dr. Eric Wish's methodology:

1. GLB (Green Line Breakout) - Breaking to new all-time highs after consolidation
2. Blue Dot - Oversold bounce candidates (stochastic oversold, near support)
3. RWB (Red White Blue) - MAs aligned in bullish order (10 > 30 > 50)

Only scan when GMI >= 5 (GREEN) - that's when the market favors breakouts.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import concurrent.futures


@dataclass
class StockSetup:
    """A stock setup/pattern match."""
    symbol: str
    pattern: str  # GLB, BLUE_DOT, RWB
    price: float
    score: int  # 1-10 strength score
    details: Dict


class StockScreener:
    """
    Scan for technically strong stocks using Wishing Wealth patterns.
    """

    # Default universe - QQQ top holdings + growth favorites
    DEFAULT_UNIVERSE = [
        # QQQ Top Holdings
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX',
        'AMD', 'ADBE', 'PEP', 'CSCO', 'TMUS', 'INTC', 'CMCSA', 'TXN', 'QCOM', 'AMGN',
        'INTU', 'AMAT', 'ISRG', 'HON', 'BKNG', 'LRCX', 'VRTX', 'REGN', 'ADI', 'MU',
        'PANW', 'KLAC', 'SNPS', 'CDNS', 'ASML', 'MELI', 'CRWD', 'MRVL', 'FTNT', 'DXCM',
        # Additional Growth Stocks
        'NOW', 'SNOW', 'PLTR', 'NET', 'DDOG', 'ZS', 'OKTA', 'TEAM', 'MDB', 'TTD',
        'ABNB', 'UBER', 'SHOP', 'SE', 'COIN', 'ROKU', 'PINS', 'SNAP', 'RBLX',
        # Sector Leaders
        'LLY', 'UNH', 'JNJ', 'V', 'MA', 'JPM', 'BAC', 'GS', 'MS', 'BLK',
        'XOM', 'CVX', 'CAT', 'DE', 'BA', 'RTX', 'LMT', 'GE', 'MMM', 'HD'
    ]

    def __init__(self, universe: List[str] = None):
        """
        Initialize screener with stock universe.

        Args:
            universe: List of stock symbols to scan (default: QQQ + growth stocks)
        """
        self.universe = universe or self.DEFAULT_UNIVERSE
        self.data_cache = {}

    def fetch_stock_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a stock."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if len(data) < 50:
                return None

            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel('Ticker')

            return data
        except:
            return None

    def detect_glb(self, symbol: str, data: pd.DataFrame) -> Optional[StockSetup]:
        """
        Detect GLB (Green Line Breakout) pattern.

        GLB criteria:
        - Stock breaks above prior all-time high (the "green line")
        - Consolidation period of 3+ months before breakout
        - Volume confirmation on breakout
        """
        if len(data) < 100:
            return None

        current_price = data['Close'].iloc[-1]
        current_high = data['High'].iloc[-1]

        # Find the all-time high before the last 20 days
        prior_ath = data['High'].iloc[:-20].max()
        prior_ath_date = data['High'].iloc[:-20].idxmax()

        # Check if breaking out now (within last 5 days)
        recent_highs = data['High'].iloc[-5:]
        breaking_out = (recent_highs > prior_ath).any()

        if not breaking_out:
            return None

        # Check for consolidation (price stayed within 15% of ATH for 3+ months)
        days_since_ath = (data.index[-1] - prior_ath_date).days
        if days_since_ath < 60:  # Need at least 2 months consolidation
            return None

        # Check consolidation quality (tight range)
        consolidation_data = data.loc[prior_ath_date:]
        consolidation_range = (consolidation_data['High'].max() - consolidation_data['Low'].min()) / prior_ath

        if consolidation_range > 0.25:  # Too volatile, not a clean consolidation
            return None

        # Volume confirmation
        avg_volume = data['Volume'].iloc[-20:-1].mean()
        recent_volume = data['Volume'].iloc[-5:].mean()
        volume_surge = recent_volume > avg_volume * 1.2

        # Calculate score
        score = 5
        if volume_surge:
            score += 2
        if days_since_ath > 90:
            score += 1
        if consolidation_range < 0.15:
            score += 1
        if current_price > prior_ath * 1.02:
            score += 1

        return StockSetup(
            symbol=symbol,
            pattern='GLB',
            price=round(current_price, 2),
            score=min(10, score),
            details={
                'prior_ath': round(prior_ath, 2),
                'breakout_pct': round((current_price / prior_ath - 1) * 100, 2),
                'consolidation_days': days_since_ath,
                'volume_surge': volume_surge
            }
        )

    def detect_blue_dot(self, symbol: str, data: pd.DataFrame) -> Optional[StockSetup]:
        """
        Detect Blue Dot (oversold bounce) pattern.

        Blue Dot criteria:
        - Stochastic oversold (K < 20)
        - Price near 50-day or 200-day MA (support)
        - In longer-term uptrend (above 200-day MA or was recently)
        - Not in freefall (some stabilization)
        """
        if len(data) < 200:
            return None

        current_price = data['Close'].iloc[-1]

        # Calculate Stochastic (10.4.4)
        low_10 = data['Low'].rolling(10).min()
        high_10 = data['High'].rolling(10).max()
        stoch_k = 100 * (data['Close'] - low_10) / (high_10 - low_10 + 0.0001)
        stoch_k = stoch_k.rolling(4).mean()

        current_stoch = stoch_k.iloc[-1]

        if current_stoch > 25:  # Not oversold enough
            return None

        # Calculate MAs
        ma_50 = data['Close'].rolling(50).mean().iloc[-1]
        ma_200 = data['Close'].rolling(200).mean().iloc[-1]

        # Check if near support (within 5% of 50-day or 200-day MA)
        near_ma50 = abs(current_price / ma_50 - 1) < 0.05
        near_ma200 = abs(current_price / ma_200 - 1) < 0.05

        if not (near_ma50 or near_ma200):
            return None

        # Check for uptrend context (was above 200-day in last 30 days)
        recent_above_200 = (data['Close'].iloc[-30:] > data['Close'].rolling(200).mean().iloc[-30:]).any()

        if not recent_above_200:
            return None

        # Check for stabilization (not crashing - last 3 days not all red)
        last_3_days = data['Close'].iloc[-3:]
        all_red = all(last_3_days.iloc[i] < last_3_days.iloc[i-1] for i in range(1, len(last_3_days)))

        if all_red:
            return None

        # Calculate score
        score = 5
        if current_stoch < 15:
            score += 1
        if near_ma200:
            score += 1
        if ma_50 > ma_200:  # 50 above 200 = stronger uptrend
            score += 2

        # Recent price action
        pct_off_high = (data['High'].iloc[-20:].max() - current_price) / current_price * 100
        if pct_off_high > 10:
            score += 1

        return StockSetup(
            symbol=symbol,
            pattern='BLUE_DOT',
            price=round(current_price, 2),
            score=min(10, score),
            details={
                'stochastic_k': round(current_stoch, 1),
                'pct_vs_ma50': round((current_price / ma_50 - 1) * 100, 2),
                'pct_vs_ma200': round((current_price / ma_200 - 1) * 100, 2),
                'pct_off_high': round(pct_off_high, 1)
            }
        )

    def detect_rwb(self, symbol: str, data: pd.DataFrame) -> Optional[StockSetup]:
        """
        Detect RWB (Red White Blue) pattern.

        RWB criteria:
        - Moving averages aligned: 10 > 30 > 50 > 100 > 200
        - Price above all MAs
        - Steady uptrend (not parabolic)
        """
        if len(data) < 200:
            return None

        current_price = data['Close'].iloc[-1]

        # Calculate MAs
        ma_10 = data['Close'].rolling(10).mean().iloc[-1]
        ma_30 = data['Close'].rolling(30).mean().iloc[-1]
        ma_50 = data['Close'].rolling(50).mean().iloc[-1]
        ma_100 = data['Close'].rolling(100).mean().iloc[-1]
        ma_200 = data['Close'].rolling(200).mean().iloc[-1]

        # Check alignment
        aligned = (ma_10 > ma_30 > ma_50 > ma_100 > ma_200)

        if not aligned:
            return None

        # Price above all MAs
        if current_price < ma_10:
            return None

        # Not too extended (within 10% of 10-day MA)
        if current_price > ma_10 * 1.10:
            return None

        # Calculate trend strength
        ma_spread = (ma_10 - ma_200) / ma_200 * 100

        # Calculate score
        score = 6  # Base score for having alignment

        if current_price > ma_10:
            score += 1
        if ma_spread > 10:
            score += 1
        if ma_spread > 20:
            score += 1

        # Check for recent pullback to MA (better entry)
        recent_low = data['Low'].iloc[-5:].min()
        if recent_low <= ma_10 * 1.02:
            score += 1

        return StockSetup(
            symbol=symbol,
            pattern='RWB',
            price=round(current_price, 2),
            score=min(10, score),
            details={
                'ma_10': round(ma_10, 2),
                'ma_50': round(ma_50, 2),
                'ma_200': round(ma_200, 2),
                'ma_spread_pct': round(ma_spread, 1),
                'pct_above_ma10': round((current_price / ma_10 - 1) * 100, 2)
            }
        )

    def scan_stock(self, symbol: str) -> List[StockSetup]:
        """Scan a single stock for all patterns."""
        setups = []

        data = self.fetch_stock_data(symbol)
        if data is None:
            return setups

        # Check each pattern
        glb = self.detect_glb(symbol, data)
        if glb:
            setups.append(glb)

        blue_dot = self.detect_blue_dot(symbol, data)
        if blue_dot:
            setups.append(blue_dot)

        rwb = self.detect_rwb(symbol, data)
        if rwb:
            setups.append(rwb)

        return setups

    def scan_universe(self, patterns: List[str] = None) -> Dict[str, List[StockSetup]]:
        """
        Scan entire universe for patterns.

        Args:
            patterns: List of patterns to scan for (default: all)

        Returns:
            Dictionary with pattern names as keys and lists of setups as values
        """
        patterns = patterns or ['GLB', 'BLUE_DOT', 'RWB']

        all_setups = {p: [] for p in patterns}

        print(f"Scanning {len(self.universe)} stocks...")

        # Use threading for faster scanning
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.scan_stock, symbol): symbol
                for symbol in self.universe
            }

            completed = 0
            for future in concurrent.futures.as_completed(future_to_symbol):
                completed += 1
                if completed % 20 == 0:
                    print(f"  Scanned {completed}/{len(self.universe)} stocks...")

                symbol = future_to_symbol[future]
                try:
                    setups = future.result()
                    for setup in setups:
                        if setup.pattern in all_setups:
                            all_setups[setup.pattern].append(setup)
                except Exception as e:
                    pass

        # Sort each pattern's results by score
        for pattern in all_setups:
            all_setups[pattern].sort(key=lambda x: x.score, reverse=True)

        return all_setups

    def generate_report(self, setups: Dict[str, List[StockSetup]], gmi_score: int) -> str:
        """Generate formatted stock screening report."""
        lines = []
        lines.append("=" * 65)
        lines.append("STOCK SCREENER - WISHING WEALTH METHODOLOGY")
        lines.append("=" * 65)
        lines.append("")

        if gmi_score < 5:
            lines.append(f"GMI: {gmi_score}/6 - NOT GREEN")
            lines.append("")
            lines.append("NOTE: Stock scanning works best when GMI >= 5 (GREEN).")
            lines.append("      Current conditions do not favor new breakout entries.")
            lines.append("      Consider waiting for market to improve.")
            lines.append("")
        else:
            lines.append(f"GMI: {gmi_score}/6 (GREEN) - FAVORABLE FOR BREAKOUTS")
            lines.append("")

        total_setups = sum(len(s) for s in setups.values())
        lines.append(f"Total Setups Found: {total_setups}")
        lines.append("")

        # GLB - Green Line Breakouts
        lines.append("-" * 65)
        lines.append("GLB (GREEN LINE BREAKOUTS) - New All-Time Highs")
        lines.append("-" * 65)

        if setups.get('GLB'):
            lines.append(f"{'Symbol':<8} {'Price':>10} {'Score':>6} {'Breakout %':>12} {'Consol Days':>12}")
            lines.append("-" * 65)
            for s in setups['GLB'][:10]:  # Top 10
                lines.append(
                    f"{s.symbol:<8} ${s.price:>9.2f} {s.score:>6}/10 "
                    f"{s.details['breakout_pct']:>+11.1f}% {s.details['consolidation_days']:>12}"
                )
        else:
            lines.append("  No GLB setups found")
        lines.append("")

        # Blue Dot - Oversold Bounces
        lines.append("-" * 65)
        lines.append("BLUE DOT (OVERSOLD BOUNCE) - Pullback to Support")
        lines.append("-" * 65)

        if setups.get('BLUE_DOT'):
            lines.append(f"{'Symbol':<8} {'Price':>10} {'Score':>6} {'Stoch':>8} {'vs MA50':>10} {'Off High':>10}")
            lines.append("-" * 65)
            for s in setups['BLUE_DOT'][:10]:
                lines.append(
                    f"{s.symbol:<8} ${s.price:>9.2f} {s.score:>6}/10 "
                    f"{s.details['stochastic_k']:>7.0f} {s.details['pct_vs_ma50']:>+9.1f}% "
                    f"{s.details['pct_off_high']:>9.1f}%"
                )
        else:
            lines.append("  No Blue Dot setups found")
        lines.append("")

        # RWB - Trend Aligned
        lines.append("-" * 65)
        lines.append("RWB (RED WHITE BLUE) - MAs Aligned in Uptrend")
        lines.append("-" * 65)

        if setups.get('RWB'):
            lines.append(f"{'Symbol':<8} {'Price':>10} {'Score':>6} {'MA Spread':>10} {'vs MA10':>10}")
            lines.append("-" * 65)
            for s in setups['RWB'][:10]:
                lines.append(
                    f"{s.symbol:<8} ${s.price:>9.2f} {s.score:>6}/10 "
                    f"{s.details['ma_spread_pct']:>+9.1f}% {s.details['pct_above_ma10']:>+9.1f}%"
                )
        else:
            lines.append("  No RWB setups found")
        lines.append("")

        lines.append("=" * 65)
        lines.append("PATTERN DEFINITIONS:")
        lines.append("  GLB: Breaking above prior all-time high after 3+ month base")
        lines.append("  BLUE DOT: Oversold (Stoch < 25) pullback to 50/200-day MA")
        lines.append("  RWB: Moving averages aligned (10 > 30 > 50 > 100 > 200)")
        lines.append("=" * 65)

        return "\n".join(lines)


def scan_for_setups(gmi_score: int = 6) -> str:
    """
    Convenience function to scan for setups and return report.

    Args:
        gmi_score: Current GMI score (0-6)

    Returns:
        Formatted report string
    """
    screener = StockScreener()
    setups = screener.scan_universe()
    return screener.generate_report(setups, gmi_score)


if __name__ == "__main__":
    # Test run
    print("Running stock screener...")
    report = scan_for_setups(gmi_score=6)
    print(report)
