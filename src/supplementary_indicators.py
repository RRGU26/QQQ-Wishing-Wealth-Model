"""
Supplementary Technical Indicators for Wishing Wealth Strategy

These indicators complement the GMI to provide:
- Oversold/overbought conditions (Stochastic, T2108)
- Trend confirmation (Moving averages, Bollinger Bands)
- Volatility context (VIX)
- Entry/exit timing signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StochasticSignal:
    """Stochastic oscillator signal details."""
    k_value: float
    d_value: float
    signal: str  # OVERSOLD, OVERBOUGHT, NEUTRAL
    buy_signal: bool  # True when crossing up from oversold


@dataclass
class T2108Signal:
    """T2108 breadth indicator signal."""
    value: float
    signal: str  # EXTREME_OVERSOLD, OVERSOLD, NEUTRAL, OVERBOUGHT
    expect_bounce: bool


@dataclass
class BollingerSignal:
    """Bollinger Bands position and signal."""
    upper: float
    middle: float
    lower: float
    position: float  # 0 = at lower, 1 = at upper
    bandwidth: float
    signal: str  # OVERSOLD, OVERBOUGHT, NEUTRAL


class SupplementaryIndicators:
    """
    Calculate supplementary indicators for the Wishing Wealth strategy.
    """

    def __init__(self):
        # Stochastic settings (10.4.4 as per Wishing Wealth)
        self.stoch_k_period = 10
        self.stoch_d_period = 4
        self.stoch_smooth = 4
        self.stoch_oversold = 20
        self.stoch_overbought = 80

        # T2108 thresholds
        self.t2108_extreme_oversold = 10
        self.t2108_oversold = 20
        self.t2108_overbought = 70

        # Bollinger settings
        self.bb_period = 20
        self.bb_std = 2

    def calculate_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int = None,
        d_period: int = None,
        smooth: int = None
    ) -> StochasticSignal:
        """
        Calculate Stochastic Oscillator (10.4.4 settings by default).

        The stochastic measures momentum by comparing closing price
        to the high-low range over a period.

        Args:
            data: DataFrame with High, Low, Close columns
            k_period: %K lookback period (default 10)
            d_period: %D smoothing period (default 4)
            smooth: %K smoothing (default 4)

        Returns:
            StochasticSignal with values and interpretation
        """
        k_period = k_period or self.stoch_k_period
        d_period = d_period or self.stoch_d_period
        smooth = smooth or self.stoch_smooth

        if len(data) < k_period + d_period + smooth:
            return StochasticSignal(
                k_value=50, d_value=50,
                signal="NEUTRAL", buy_signal=False
            )

        # Calculate raw %K
        low_min = data['Low'].rolling(k_period).min()
        high_max = data['High'].rolling(k_period).max()

        raw_k = 100 * (data['Close'] - low_min) / (high_max - low_min + 0.0001)

        # Smooth %K
        k = raw_k.rolling(smooth).mean()

        # Calculate %D (signal line)
        d = k.rolling(d_period).mean()

        k_current = k.iloc[-1]
        d_current = d.iloc[-1]
        k_prev = k.iloc[-2] if len(k) > 1 else k_current

        # Determine signal
        if k_current < self.stoch_oversold:
            signal = "OVERSOLD"
        elif k_current > self.stoch_overbought:
            signal = "OVERBOUGHT"
        else:
            signal = "NEUTRAL"

        # Buy signal: K crosses above D from oversold
        buy_signal = (
            k_prev < d.iloc[-2] and
            k_current > d_current and
            k_current < 50  # Still in lower half
        )

        return StochasticSignal(
            k_value=round(k_current, 2),
            d_value=round(d_current, 2),
            signal=signal,
            buy_signal=buy_signal
        )

    def calculate_t2108_proxy(
        self,
        market_data: pd.DataFrame
    ) -> T2108Signal:
        """
        Calculate T2108 proxy (% of stocks above 200-day MA).

        The actual T2108 from Worden Brothers tracks all NYSE stocks.
        This proxy uses QQQ constituent behavior estimated from QQQ itself.

        Args:
            market_data: DataFrame with Close prices

        Returns:
            T2108Signal with value and interpretation
        """
        if len(market_data) < 200:
            return T2108Signal(
                value=50, signal="NEUTRAL", expect_bounce=False
            )

        # Proxy: Use distance from 200-day MA as breadth estimate
        ma_200 = market_data['Close'].rolling(200).mean()
        current_close = market_data['Close'].iloc[-1]
        ma_200_current = ma_200.iloc[-1]

        # Calculate how many recent days closed above 200-day MA
        above_200 = (market_data['Close'].iloc[-20:] > ma_200.iloc[-20:]).mean() * 100

        # Blend with current position
        pct_vs_ma = ((current_close / ma_200_current) - 1) * 100

        # Estimate T2108 (simplified model)
        # When QQQ is 5% above 200-day MA, estimate ~60% of stocks above
        # When QQQ is at 200-day MA, estimate ~50% of stocks above
        # When QQQ is 5% below 200-day MA, estimate ~30% of stocks above
        t2108_estimate = 50 + (pct_vs_ma * 4)  # Scale factor
        t2108_estimate = max(5, min(95, t2108_estimate))

        # Blend with recent price action
        t2108_proxy = (t2108_estimate * 0.7) + (above_200 * 0.3)

        # Determine signal
        if t2108_proxy < self.t2108_extreme_oversold:
            signal = "EXTREME_OVERSOLD"
            expect_bounce = True
        elif t2108_proxy < self.t2108_oversold:
            signal = "OVERSOLD"
            expect_bounce = True
        elif t2108_proxy > self.t2108_overbought:
            signal = "OVERBOUGHT"
            expect_bounce = False
        else:
            signal = "NEUTRAL"
            expect_bounce = False

        return T2108Signal(
            value=round(t2108_proxy, 2),
            signal=signal,
            expect_bounce=expect_bounce
        )

    def calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = None,
        std_dev: float = None
    ) -> BollingerSignal:
        """
        Calculate Bollinger Bands and price position.

        Used for Blue Dot of Happiness setup - bounces from lower band.

        Args:
            data: DataFrame with Close prices
            period: Moving average period (default 20)
            std_dev: Standard deviations for bands (default 2)

        Returns:
            BollingerSignal with bands and position
        """
        period = period or self.bb_period
        std_dev = std_dev or self.bb_std

        if len(data) < period:
            return BollingerSignal(
                upper=0, middle=0, lower=0,
                position=0.5, bandwidth=0, signal="NEUTRAL"
            )

        close = data['Close']
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        current_close = close.iloc[-1]
        upper_current = upper.iloc[-1]
        middle_current = middle.iloc[-1]
        lower_current = lower.iloc[-1]

        # Position within bands (0 = lower, 1 = upper)
        band_range = upper_current - lower_current
        if band_range > 0:
            position = (current_close - lower_current) / band_range
        else:
            position = 0.5

        # Bandwidth (volatility measure)
        bandwidth = (band_range / middle_current) * 100

        # Determine signal
        if position < 0.1:
            signal = "OVERSOLD"  # At or below lower band
        elif position > 0.9:
            signal = "OVERBOUGHT"  # At or above upper band
        elif position < 0.3:
            signal = "NEAR_LOWER"
        elif position > 0.7:
            signal = "NEAR_UPPER"
        else:
            signal = "NEUTRAL"

        return BollingerSignal(
            upper=round(upper_current, 2),
            middle=round(middle_current, 2),
            lower=round(lower_current, 2),
            position=round(position, 3),
            bandwidth=round(bandwidth, 2),
            signal=signal
        )

    def calculate_ma_framework(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        Calculate the complete moving average framework.

        Key levels from Wishing Wealth:
        - 10-week (50-day) MA: Defensive trigger
        - 30-week (150-day) MA: Exit trigger
        - 10-week vs 30-week: Trend confirmation

        Args:
            data: DataFrame with Close prices

        Returns:
            Dictionary with all MA metrics
        """
        if len(data) < 150:
            return {"error": "Insufficient data for MA framework"}

        close = data['Close']
        current = close.iloc[-1]

        # Daily MAs
        ma_10 = close.rolling(10).mean().iloc[-1]
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1]   # ~10 weeks
        ma_150 = close.rolling(150).mean().iloc[-1]  # ~30 weeks
        ma_200 = close.rolling(200).mean().iloc[-1] if len(data) >= 200 else None

        # Trend signals
        above_10w = current > ma_50
        above_30w = current > ma_150
        trend_bullish = ma_50 > ma_150  # 10-week > 30-week

        # Calculate percentages
        pct_vs_10w = ((current / ma_50) - 1) * 100
        pct_vs_30w = ((current / ma_150) - 1) * 100

        # Determine overall MA signal
        if above_10w and above_30w and trend_bullish:
            ma_signal = "STRONG_BULLISH"
        elif above_10w and above_30w:
            ma_signal = "BULLISH"
        elif above_30w:
            ma_signal = "DEFENSIVE"  # Below 10-week but above 30-week
        else:
            ma_signal = "BEARISH"  # Below both

        return {
            "current_price": round(current, 2),
            "ma_10d": round(ma_10, 2),
            "ma_20d": round(ma_20, 2),
            "ma_50d_10w": round(ma_50, 2),
            "ma_150d_30w": round(ma_150, 2),
            "ma_200d": round(ma_200, 2) if ma_200 else None,
            "above_10w_ma": above_10w,
            "above_30w_ma": above_30w,
            "trend_bullish": trend_bullish,
            "pct_vs_10w": round(pct_vs_10w, 2),
            "pct_vs_30w": round(pct_vs_30w, 2),
            "ma_signal": ma_signal
        }

    def calculate_vix_context(
        self,
        vix_data: pd.DataFrame
    ) -> Dict:
        """
        Calculate VIX context for volatility regime.

        Args:
            vix_data: DataFrame with VIX Close prices

        Returns:
            Dictionary with VIX metrics and regime
        """
        if len(vix_data) < 20:
            return {"error": "Insufficient VIX data"}

        current_vix = vix_data['Close'].iloc[-1]
        ma_20 = vix_data['Close'].rolling(20).mean().iloc[-1]
        ma_50 = vix_data['Close'].rolling(50).mean().iloc[-1] if len(vix_data) >= 50 else ma_20

        # VIX percentile over last year
        if len(vix_data) >= 252:
            percentile = (vix_data['Close'].iloc[-252:] < current_vix).mean() * 100
        else:
            percentile = 50

        # Determine regime
        if current_vix < 15:
            regime = "LOW_VOL"
        elif current_vix < 20:
            regime = "NORMAL"
        elif current_vix < 30:
            regime = "ELEVATED"
        else:
            regime = "HIGH_VOL"

        # VIX trend
        vix_rising = current_vix > ma_20

        return {
            "current_vix": round(current_vix, 2),
            "vix_ma_20": round(ma_20, 2),
            "vix_ma_50": round(ma_50, 2),
            "vix_percentile": round(percentile, 1),
            "regime": regime,
            "vix_rising": vix_rising
        }

    def detect_blue_dot_setup(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        Detect Blue Dot of Happiness setup.

        Criteria:
        1. Stock recently at all-time high (within 20 days)
        2. Currently oversold (near lower Bollinger Band)
        3. Signs of bounce (higher close than recent low)

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with setup detection results
        """
        if len(data) < 50:
            return {"setup_detected": False, "reason": "Insufficient data"}

        # Check for recent ATH
        high_20d = data['High'].iloc[-20:].max()
        ath = data['High'].max()
        recently_at_ath = high_20d >= ath * 0.98

        # Check Bollinger position
        bb = self.calculate_bollinger_bands(data)
        near_lower_band = bb.position < 0.25

        # Check for bounce
        low_5d = data['Low'].iloc[-5:].min()
        current_close = data['Close'].iloc[-1]
        bouncing = current_close > low_5d

        # Volume confirmation
        avg_volume = data['Volume'].iloc[-20:].mean()
        recent_volume = data['Volume'].iloc[-1]
        volume_surge = recent_volume > avg_volume * 1.2

        setup_detected = recently_at_ath and near_lower_band and bouncing

        return {
            "setup_detected": setup_detected,
            "recently_at_ath": recently_at_ath,
            "near_lower_band": near_lower_band,
            "bouncing": bouncing,
            "volume_surge": volume_surge,
            "bb_position": bb.position,
            "stop_level": round(low_5d * 0.99, 2)  # Stop just below recent low
        }

    def calculate_all_supplementary(
        self,
        qqq_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calculate all supplementary indicators at once.

        Args:
            qqq_data: DataFrame with QQQ OHLCV data
            vix_data: Optional DataFrame with VIX data

        Returns:
            Dictionary with all supplementary indicators
        """
        result = {
            "stochastic": None,
            "t2108_proxy": None,
            "bollinger": None,
            "ma_framework": None,
            "vix_context": None,
            "blue_dot_setup": None
        }

        # Stochastic
        stoch = self.calculate_stochastic(qqq_data)
        result["stochastic"] = {
            "k": stoch.k_value,
            "d": stoch.d_value,
            "signal": stoch.signal,
            "buy_signal": stoch.buy_signal
        }

        # T2108 proxy
        t2108 = self.calculate_t2108_proxy(qqq_data)
        result["t2108_proxy"] = {
            "value": t2108.value,
            "signal": t2108.signal,
            "expect_bounce": t2108.expect_bounce
        }

        # Bollinger Bands
        bb = self.calculate_bollinger_bands(qqq_data)
        result["bollinger"] = {
            "upper": bb.upper,
            "middle": bb.middle,
            "lower": bb.lower,
            "position": bb.position,
            "bandwidth": bb.bandwidth,
            "signal": bb.signal
        }

        # MA Framework
        result["ma_framework"] = self.calculate_ma_framework(qqq_data)

        # VIX Context
        if vix_data is not None and len(vix_data) > 0:
            result["vix_context"] = self.calculate_vix_context(vix_data)

        # Blue Dot Setup
        result["blue_dot_setup"] = self.detect_blue_dot_setup(qqq_data)

        return result
