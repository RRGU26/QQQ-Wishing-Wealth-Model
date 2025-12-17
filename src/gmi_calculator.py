"""
GMI (General Market Index) Calculator
Based on Dr. Eric Wish's Wishing Wealth methodology
https://wishingwealthblog.com/

The GMI is a 6-component index (0-6) that signals market conditions:
- GMI 5-6 (GREEN): Aggressively long
- GMI 3-4 (YELLOW): Defensive/nimble
- GMI 0-2 (RED): Cash or short
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta


class GMICalculator:
    """
    Calculate the General Market Index (GMI) based on Wishing Wealth methodology.

    Components:
    1. Successful New High Index - stocks at 52-week high 10 days ago still higher
    2. Daily New Highs Count - >100 new highs in stock universe
    3. Daily QQQ Index - QQQ above 10-week MA with positive technicals
    4. Daily SPY Index - SPY above 10-week MA
    5. Weekly QQQ Index - 10-week MA > 30-week MA
    6. IBD Mutual Fund Index - Growth fund proxy above 50-day MA
    """

    def __init__(self):
        self.new_high_threshold = 100
        self.lookback_days = 10

    def calculate_successful_new_high_index(
        self,
        universe_data: pd.DataFrame,
        current_date: datetime
    ) -> Tuple[bool, int]:
        """
        Component 1: Successful New High Index

        Counts stocks that:
        - Hit a 52-week high 10 trading days ago
        - Are still trading higher than that high today

        Args:
            universe_data: DataFrame with columns ['symbol', 'date', 'close', 'high_52w']
            current_date: Date to calculate for

        Returns:
            Tuple of (is_positive, count)
        """
        if universe_data is None or universe_data.empty:
            # Fallback: use simplified calculation based on QQQ behavior
            return False, 0

        lookback_date = current_date - timedelta(days=14)  # ~10 trading days

        # Find stocks that hit 52-week highs 10 days ago
        past_data = universe_data[universe_data['date'] == lookback_date]
        current_data = universe_data[universe_data['date'] == current_date]

        if past_data.empty or current_data.empty:
            return False, 0

        # Stocks at 52-week high 10 days ago
        at_high_10d_ago = past_data[past_data['close'] >= past_data['high_52w'] * 0.99]

        # Check if they're still higher
        merged = at_high_10d_ago.merge(
            current_data[['symbol', 'close']],
            on='symbol',
            suffixes=('_past', '_current')
        )

        successful = (merged['close_current'] > merged['close_past']).sum()

        return successful >= self.new_high_threshold, successful

    def calculate_daily_new_highs(
        self,
        new_highs_count: int
    ) -> Tuple[bool, int]:
        """
        Component 2: Daily New Highs Count

        Positive when >100 stocks hit new 52-week highs

        Args:
            new_highs_count: Number of stocks at new 52-week highs today

        Returns:
            Tuple of (is_positive, count)
        """
        return new_highs_count >= self.new_high_threshold, new_highs_count

    def calculate_daily_qqq_index(
        self,
        qqq_data: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Component 3: Daily QQQ Index

        Positive when QQQ is above its 10-week (50-day) moving average
        with positive technical momentum.

        Args:
            qqq_data: DataFrame with OHLCV data for QQQ

        Returns:
            Tuple of (is_positive, details_dict)
        """
        if len(qqq_data) < 50:
            return False, {"error": "Insufficient data"}

        current_close = qqq_data['Close'].iloc[-1]
        ma_50 = qqq_data['Close'].rolling(50).mean().iloc[-1]
        ma_10 = qqq_data['Close'].rolling(10).mean().iloc[-1]

        # QQQ above 50-day MA and 10-day MA trending up
        is_positive = (current_close > ma_50) and (ma_10 > ma_50)

        details = {
            "close": current_close,
            "ma_50": ma_50,
            "ma_10": ma_10,
            "pct_above_ma50": ((current_close / ma_50) - 1) * 100
        }

        return is_positive, details

    def calculate_daily_spy_index(
        self,
        spy_data: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Component 4: Daily SPY Index

        Positive when SPY is above its 10-week (50-day) moving average.

        Args:
            spy_data: DataFrame with OHLCV data for SPY

        Returns:
            Tuple of (is_positive, details_dict)
        """
        if len(spy_data) < 50:
            return False, {"error": "Insufficient data"}

        current_close = spy_data['Close'].iloc[-1]
        ma_50 = spy_data['Close'].rolling(50).mean().iloc[-1]

        is_positive = current_close > ma_50

        details = {
            "close": current_close,
            "ma_50": ma_50,
            "pct_above_ma50": ((current_close / ma_50) - 1) * 100
        }

        return is_positive, details

    def calculate_weekly_qqq_index(
        self,
        qqq_data: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Component 5: Weekly QQQ Index

        Positive when 10-week MA is above 30-week MA.
        This is the key characteristic of tradable advances.

        Args:
            qqq_data: Daily DataFrame with OHLCV data for QQQ

        Returns:
            Tuple of (is_positive, details_dict)
        """
        if len(qqq_data) < 150:  # Need ~30 weeks of data
            return False, {"error": "Insufficient data"}

        # Ensure datetime index without timezone
        df = qqq_data.copy()
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

        # Convert to weekly (using Friday close or last day of week)
        weekly = df.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if len(weekly) < 30:
            return False, {"error": "Insufficient weekly data"}

        ma_10w = weekly['Close'].rolling(10).mean().iloc[-1]
        ma_30w = weekly['Close'].rolling(30).mean().iloc[-1]
        current_close = weekly['Close'].iloc[-1]

        is_positive = ma_10w > ma_30w

        details = {
            "close_weekly": current_close,
            "ma_10w": ma_10w,
            "ma_30w": ma_30w,
            "ma_spread_pct": ((ma_10w / ma_30w) - 1) * 100
        }

        return is_positive, details

    def calculate_ibd_fund_index(
        self,
        fund_data: pd.DataFrame
    ) -> Tuple[bool, Dict]:
        """
        Component 6: IBD Mutual Fund Index

        Uses XLK (Technology Select Sector SPDR) as a proxy for growth fund health.
        Positive when above its 50-day moving average.

        Args:
            fund_data: DataFrame with OHLCV data for XLK or similar growth fund

        Returns:
            Tuple of (is_positive, details_dict)
        """
        if len(fund_data) < 50:
            return False, {"error": "Insufficient data"}

        current_close = fund_data['Close'].iloc[-1]
        ma_50 = fund_data['Close'].rolling(50).mean().iloc[-1]

        is_positive = current_close > ma_50

        details = {
            "close": current_close,
            "ma_50": ma_50,
            "pct_above_ma50": ((current_close / ma_50) - 1) * 100
        }

        return is_positive, details

    def calculate_gmi(
        self,
        qqq_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        fund_data: pd.DataFrame,
        new_highs_count: Optional[int] = None,
        successful_nh_count: Optional[int] = None,
        universe_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calculate the complete GMI score.

        Args:
            qqq_data: Daily OHLCV data for QQQ
            spy_data: Daily OHLCV data for SPY
            fund_data: Daily OHLCV data for growth fund proxy (XLK)
            new_highs_count: Optional count of daily new highs (Component 2)
            successful_nh_count: Optional count of successful new highs (Component 1)
            universe_data: Optional DataFrame for successful new high calculation

        Returns:
            Dictionary with GMI score, signal, and component details
        """
        components = {}
        gmi_score = 0

        # Component 1: Successful New High Index
        if successful_nh_count is not None:
            # Use pre-calculated successful new high count from breadth data
            c1_positive = successful_nh_count >= self.new_high_threshold
            c1_count = successful_nh_count
        elif universe_data is not None:
            c1_positive, c1_count = self.calculate_successful_new_high_index(
                universe_data,
                qqq_data.index[-1]
            )
        else:
            # Proxy: use market momentum as substitute
            c1_positive = self._proxy_successful_new_high(qqq_data)
            c1_count = None
        components['successful_new_high'] = {
            'positive': c1_positive,
            'count': c1_count,
            'using_proxy': successful_nh_count is None and universe_data is None
        }
        if c1_positive:
            gmi_score += 1

        # Component 2: Daily New Highs
        if new_highs_count is not None:
            c2_positive, c2_count = self.calculate_daily_new_highs(new_highs_count)
        else:
            # Proxy: use market breadth estimate
            c2_positive = self._proxy_daily_new_highs(qqq_data)
            c2_count = None
        components['daily_new_highs'] = {
            'positive': c2_positive,
            'count': c2_count,
            'using_proxy': new_highs_count is None
        }
        if c2_positive:
            gmi_score += 1

        # Component 3: Daily QQQ Index
        c3_positive, c3_details = self.calculate_daily_qqq_index(qqq_data)
        components['daily_qqq'] = {
            'positive': c3_positive,
            'details': c3_details
        }
        if c3_positive:
            gmi_score += 1

        # Component 4: Daily SPY Index
        c4_positive, c4_details = self.calculate_daily_spy_index(spy_data)
        components['daily_spy'] = {
            'positive': c4_positive,
            'details': c4_details
        }
        if c4_positive:
            gmi_score += 1

        # Component 5: Weekly QQQ Index
        c5_positive, c5_details = self.calculate_weekly_qqq_index(qqq_data)
        components['weekly_qqq'] = {
            'positive': c5_positive,
            'details': c5_details
        }
        if c5_positive:
            gmi_score += 1

        # Component 6: IBD Fund Index
        c6_positive, c6_details = self.calculate_ibd_fund_index(fund_data)
        components['ibd_fund_index'] = {
            'positive': c6_positive,
            'details': c6_details
        }
        if c6_positive:
            gmi_score += 1

        # Determine signal
        if gmi_score >= 5:
            signal = "GREEN"
            signal_action = "BUY"
        elif gmi_score >= 3:
            signal = "YELLOW"
            signal_action = "HOLD"
        else:
            signal = "RED"
            signal_action = "SELL"

        return {
            'gmi_score': gmi_score,
            'gmi_max': 6,
            'signal': signal,
            'signal_action': signal_action,
            'components': components,
            'timestamp': datetime.now().isoformat()
        }

    def _proxy_successful_new_high(self, qqq_data: pd.DataFrame) -> bool:
        """
        Proxy for Successful New High Index when universe data unavailable.
        Uses QQQ momentum and trend strength.
        """
        if len(qqq_data) < 20:
            return False

        # Check if QQQ has been making higher highs
        current_high = qqq_data['High'].iloc[-1]
        high_10d_ago = qqq_data['High'].iloc[-10]
        high_20d = qqq_data['High'].iloc[-20:].max()

        # Positive if current high is within 2% of 20-day high
        # and higher than 10 days ago
        return (current_high >= high_20d * 0.98) and (current_high > high_10d_ago)

    def _proxy_daily_new_highs(self, qqq_data: pd.DataFrame) -> bool:
        """
        Proxy for Daily New Highs when actual count unavailable.
        Uses QQQ relative strength and momentum.
        """
        if len(qqq_data) < 20:
            return False

        # Strong momentum suggests broad new highs
        returns_5d = (qqq_data['Close'].iloc[-1] / qqq_data['Close'].iloc[-5]) - 1
        returns_20d = (qqq_data['Close'].iloc[-1] / qqq_data['Close'].iloc[-20]) - 1

        # Positive momentum on both timeframes
        return returns_5d > 0 and returns_20d > 0


def get_gmi_interpretation(gmi_score: int) -> str:
    """Get detailed interpretation of GMI score."""
    interpretations = {
        0: "Extreme bearish - All indicators negative. Stay in cash or short.",
        1: "Very bearish - Market in decline. Avoid long positions.",
        2: "Bearish - Weak market conditions. Be defensive.",
        3: "Neutral/Cautious - Mixed signals. Reduce exposure, be nimble.",
        4: "Cautiously bullish - Improving conditions. Selective longs OK.",
        5: "Bullish - Strong market. Good conditions for growth stocks.",
        6: "Very bullish - All indicators positive. Aggressive long positioning."
    }
    return interpretations.get(gmi_score, "Unknown")
