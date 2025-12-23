"""
QQQ Options Analyzer

Generates options trading recommendations based on GMI signals
for post-market trading (4:00 - 4:15 PM ET).

Integrates with the Wishing Wealth model to provide:
- Call/Put recommendations
- Strike selection
- Position sizing
- Risk management levels
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class OptionsRecommendation:
    """Options trade recommendation."""
    action: str           # BUY_CALLS, BUY_PUTS, NO_TRADE
    direction: str        # BULLISH, BEARISH, NEUTRAL
    strike_atm: float     # ATM strike price
    strike_recommended: float  # Recommended strike
    strike_type: str      # ATM, OTM, ITM
    expiration: str       # Expiration date
    confidence: float     # 0-1 confidence level
    position_size: str    # FULL, HALF, NONE
    max_risk_dollars: float
    profit_target_pct: float
    stop_loss_pct: float
    reasoning: str


class OptionsAnalyzer:
    """
    Analyze market conditions and generate options recommendations.
    """

    def __init__(self, options_capital: float = 5000):
        """
        Initialize options analyzer.

        Args:
            options_capital: Capital allocated for options trading
        """
        self.options_capital = options_capital
        self.risk_per_trade_pct = 0.05  # 5% max risk per trade
        self.confidence_threshold = 0.55  # Minimum confidence to trade

    def analyze(
        self,
        prediction: Dict,
        current_price: float = None
    ) -> OptionsRecommendation:
        """
        Generate options recommendation from model prediction.

        Args:
            prediction: Output from QQQWishingWealthModel.predict()
            current_price: Current QQQ price (optional, uses prediction if not provided)

        Returns:
            OptionsRecommendation with trade details
        """
        # Extract key data
        gmi_score = prediction.get('gmi', {}).get('score', 3)
        gmi_signal = prediction.get('gmi', {}).get('signal', 'YELLOW')
        final = prediction.get('final_prediction', {})
        direction = final.get('direction', 'NEUTRAL')
        confidence = final.get('confidence', 0.5)
        expected_move = final.get('expected_move_pct', 0)
        signals_aligned = final.get('signals_aligned', False)

        # Get supplementary data
        supplementary = prediction.get('supplementary', {})
        stochastic = supplementary.get('stochastic', {})
        stoch_signal = stochastic.get('signal', 'NEUTRAL')
        vix_context = supplementary.get('vix_context', {})
        vix_level = vix_context.get('current_vix', 20) if vix_context else 20

        # Current price
        price = current_price or prediction.get('current_price', 500)

        # Calculate ATM strike (round to nearest $1 for QQQ)
        strike_atm = round(price)

        # Determine expiration (next trading day)
        today = datetime.now()
        if today.weekday() == 4:  # Friday
            expiration = (today + timedelta(days=3)).strftime('%Y-%m-%d')
        elif today.weekday() == 5:  # Saturday
            expiration = (today + timedelta(days=2)).strftime('%Y-%m-%d')
        elif today.weekday() == 6:  # Sunday
            expiration = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            expiration = (today + timedelta(days=1)).strftime('%Y-%m-%d')

        # Determine action based on GMI
        action, strike_recommended, strike_type, reasoning = self._determine_action(
            gmi_score=gmi_score,
            gmi_signal=gmi_signal,
            direction=direction,
            confidence=confidence,
            stoch_signal=stoch_signal,
            vix_level=vix_level,
            signals_aligned=signals_aligned,
            strike_atm=strike_atm,
            expected_move=expected_move
        )

        # Determine position size
        position_size = self._determine_position_size(confidence, signals_aligned, vix_level)

        # Calculate risk parameters
        if position_size == 'NONE':
            max_risk = 0
        elif position_size == 'HALF':
            max_risk = self.options_capital * self.risk_per_trade_pct * 0.5
        else:
            max_risk = self.options_capital * self.risk_per_trade_pct

        return OptionsRecommendation(
            action=action,
            direction='BULLISH' if 'CALLS' in action else 'BEARISH' if 'PUTS' in action else 'NEUTRAL',
            strike_atm=strike_atm,
            strike_recommended=strike_recommended,
            strike_type=strike_type,
            expiration=expiration,
            confidence=confidence,
            position_size=position_size,
            max_risk_dollars=round(max_risk, 2),
            profit_target_pct=100,  # 100% gain target
            stop_loss_pct=50,       # 50% loss stop
            reasoning=reasoning
        )

    def _determine_action(
        self,
        gmi_score: int,
        gmi_signal: str,
        direction: str,
        confidence: float,
        stoch_signal: str,
        vix_level: float,
        signals_aligned: bool,
        strike_atm: float,
        expected_move: float
    ) -> tuple:
        """Determine options action based on signals."""

        reasoning_parts = []

        # GMI-based primary decision
        if gmi_score >= 5:
            action = 'BUY_CALLS'
            reasoning_parts.append(f"GMI {gmi_score}/6 (GREEN) favors calls")

            # Strike selection
            if confidence > 0.7 and signals_aligned:
                strike = strike_atm + 2  # Aggressive OTM
                strike_type = 'OTM'
                reasoning_parts.append("High confidence - aggressive OTM strike")
            else:
                strike = strike_atm
                strike_type = 'ATM'
                reasoning_parts.append("Standard ATM strike")

        elif gmi_score <= 1:
            action = 'BUY_PUTS'
            reasoning_parts.append(f"GMI {gmi_score}/6 (RED) favors puts")

            if confidence > 0.7 and signals_aligned:
                strike = strike_atm - 2  # Aggressive OTM
                strike_type = 'OTM'
                reasoning_parts.append("High confidence - aggressive OTM strike")
            else:
                strike = strike_atm
                strike_type = 'ATM'
                reasoning_parts.append("Standard ATM strike")

        elif gmi_score == 2:
            if stoch_signal == 'OVERBOUGHT':
                action = 'BUY_PUTS'
                strike = strike_atm
                strike_type = 'ATM'
                reasoning_parts.append(f"GMI {gmi_score}/6 + overbought stochastic = light puts")
            else:
                action = 'NO_TRADE'
                strike = strike_atm
                strike_type = 'ATM'
                reasoning_parts.append(f"GMI {gmi_score}/6 without confirmation - no trade")

        elif gmi_score == 4:
            if stoch_signal == 'OVERSOLD':
                action = 'BUY_CALLS'
                strike = strike_atm
                strike_type = 'ATM'
                reasoning_parts.append(f"GMI {gmi_score}/6 + oversold stochastic = light calls")
            else:
                action = 'NO_TRADE'
                strike = strike_atm
                strike_type = 'ATM'
                reasoning_parts.append(f"GMI {gmi_score}/6 without confirmation - no trade")

        else:  # GMI = 3
            action = 'NO_TRADE'
            strike = strike_atm
            strike_type = 'ATM'
            reasoning_parts.append("GMI 3/6 (NEUTRAL) - no edge, skip trade")

        # Confidence check
        if confidence < self.confidence_threshold and action != 'NO_TRADE':
            action = 'NO_TRADE'
            reasoning_parts.append(f"Confidence {confidence:.0%} below threshold - no trade")

        # VIX warning
        if vix_level > 30:
            reasoning_parts.append(f"WARNING: VIX elevated at {vix_level:.1f}")

        # Signals alignment note
        if signals_aligned and action != 'NO_TRADE':
            reasoning_parts.append("Signals aligned - increased conviction")

        return action, strike, strike_type, "; ".join(reasoning_parts)

    def _determine_position_size(
        self,
        confidence: float,
        signals_aligned: bool,
        vix_level: float
    ) -> str:
        """Determine position size based on confidence and conditions."""

        if confidence < self.confidence_threshold:
            return 'NONE'

        # Base size on confidence
        if confidence >= 0.7:
            base_size = 'FULL'
        elif confidence >= 0.55:
            base_size = 'HALF'
        else:
            return 'NONE'

        # Adjust for VIX
        if vix_level > 30:
            # High VIX = expensive premiums, reduce size
            if base_size == 'FULL':
                return 'HALF'
            else:
                return 'NONE'

        # Boost for alignment
        if signals_aligned and base_size == 'HALF':
            return 'FULL'

        return base_size

    def generate_report(
        self,
        prediction: Dict,
        recommendation: OptionsRecommendation
    ) -> str:
        """Generate formatted options analysis report."""

        gmi = prediction.get('gmi', {})
        supplementary = prediction.get('supplementary', {})
        stoch = supplementary.get('stochastic', {})
        ma = supplementary.get('ma_framework', {})
        vix = supplementary.get('vix_context', {})
        vol_regime = prediction.get('volatility_regime', {})

        # Format the report
        lines = []
        lines.append("=" * 65)
        lines.append(f"QQQ OPTIONS DAILY ANALYSIS - {prediction.get('as_of_date', 'N/A')}")
        lines.append("=" * 65)
        lines.append("")
        lines.append(f"MARKET CLOSE: ${prediction.get('current_price', 0):.2f}")
        lines.append(f"TARGET PRICE: ${prediction.get('target_price', 0):.2f} ({prediction.get('final_prediction', {}).get('expected_move_pct', 0):+.2f}%)")
        lines.append("")

        # GMI Status
        lines.append("GMI STATUS")
        lines.append("-" * 65)
        lines.append(f"Score: {gmi.get('score', 'N/A')}/6 ({gmi.get('signal', 'N/A')})")

        components = gmi.get('components', {})
        positive = sum(1 for c in components.values() if isinstance(c, dict) and c.get('positive', False))
        lines.append(f"Components: {positive} positive, {6 - positive} negative")
        lines.append("")

        # Supplementary Signals
        lines.append("SUPPLEMENTARY SIGNALS")
        lines.append("-" * 65)
        lines.append(f"Stochastic (10.4.4): {stoch.get('k', 'N/A'):.0f} - {stoch.get('signal', 'N/A')}")
        lines.append(f"MA Framework: {ma.get('ma_signal', 'N/A')}")

        vix_level = vix.get('current_vix', 'N/A') if vix else 'N/A'
        vix_regime = vix.get('regime', 'N/A') if vix else 'N/A'
        lines.append(f"VIX Level: {vix_level} ({vix_regime})")

        aligned = prediction.get('final_prediction', {}).get('signals_aligned', False)
        lines.append(f"Signals Aligned: {'YES' if aligned else 'NO'}")
        lines.append("")

        # Options Recommendation
        lines.append("OPTIONS RECOMMENDATION")
        lines.append("-" * 65)

        if recommendation.action == 'NO_TRADE':
            lines.append("ACTION: NO TRADE")
            lines.append(f"REASON: {recommendation.reasoning}")
        else:
            action_emoji = ""  # Removed emoji for Windows console compatibility
            lines.append(f"ACTION: {action_emoji}{recommendation.action}")
            lines.append(f"STRIKE: ${recommendation.strike_recommended:.0f} ({recommendation.strike_type})")
            lines.append(f"EXPIRATION: {recommendation.expiration}")
            lines.append(f"CONFIDENCE: {recommendation.confidence:.0%}")
            lines.append(f"POSITION SIZE: {recommendation.position_size}")
            lines.append("")
            lines.append(f"REASONING: {recommendation.reasoning}")

        lines.append("")

        # Risk Management
        lines.append("RISK MANAGEMENT")
        lines.append("-" * 65)
        if recommendation.action != 'NO_TRADE':
            lines.append(f"Max Risk: ${recommendation.max_risk_dollars:.2f}")
            lines.append(f"Stop Loss: -{recommendation.stop_loss_pct}% of premium")
            lines.append(f"Profit Target: +{recommendation.profit_target_pct}% of premium")
        else:
            lines.append("No trade - no risk parameters needed")

        lines.append("")
        lines.append("=" * 65)

        return "\n".join(lines)


def analyze_prediction(prediction: Dict, options_capital: float = 5000) -> tuple:
    """
    Convenience function to analyze a prediction and get options recommendation.

    Args:
        prediction: Output from QQQWishingWealthModel.predict()
        options_capital: Capital for options trading

    Returns:
        Tuple of (recommendation, report_string)
    """
    analyzer = OptionsAnalyzer(options_capital=options_capital)
    recommendation = analyzer.analyze(prediction)
    report = analyzer.generate_report(prediction, recommendation)
    return recommendation, report
