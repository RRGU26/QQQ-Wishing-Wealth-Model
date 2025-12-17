"""
QQQ Wishing Wealth Prediction Model

A unified model for predicting next-day QQQ price movements based on
Dr. Eric Wish's Wishing Wealth methodology enhanced with ML.

This model combines:
1. GMI (General Market Index) - 6-component rule-based scoring
2. Supplementary indicators (Stochastic, T2108, Bollinger, MA framework)
3. ML enhancement layer for probability and magnitude estimation

Usage:
    from qqq_wishing_wealth_model import QQQWishingWealthModel

    model = QQQWishingWealthModel()
    model.load_data()  # Fetches latest data
    model.train()      # Trains ML component
    prediction = model.predict()  # Get next-day prediction
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import json
import os

from gmi_calculator import GMICalculator, get_gmi_interpretation
from supplementary_indicators import SupplementaryIndicators
from ml_enhancement import MLEnhancement, SignalCombiner


class QQQWishingWealthModel:
    """
    Unified QQQ prediction model based on Wishing Wealth methodology.

    Attributes:
        gmi_calculator: GMI calculation engine
        supplementary: Supplementary indicator calculator
        ml_engine: ML enhancement layer
        signal_combiner: Signal combination engine
    """

    def __init__(self, data_dir: str = None):
        """
        Initialize the model components.

        Args:
            data_dir: Directory for caching data (optional)
        """
        self.gmi_calculator = GMICalculator()
        self.supplementary = SupplementaryIndicators()
        self.ml_engine = MLEnhancement()
        self.signal_combiner = SignalCombiner()

        # Data storage
        self.qqq_data = None
        self.spy_data = None
        self.xlk_data = None  # Growth fund proxy
        self.vix_data = None

        # Data directory
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # Model state
        self.is_trained = False
        self.last_prediction = None

    def load_data(
        self,
        start_date: str = None,
        end_date: str = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Load market data for all required symbols.

        Args:
            start_date: Start date for historical data (default: 2 years ago)
            end_date: End date (default: today)
            use_cache: Whether to use cached data if available

        Returns:
            Dictionary with data loading status
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        symbols = {
            'QQQ': 'qqq_data',
            'SPY': 'spy_data',
            'XLK': 'xlk_data',
            '^VIX': 'vix_data'
        }

        results = {}

        for symbol, attr_name in symbols.items():
            try:
                # Try cache first
                cache_file = os.path.join(self.data_dir, f'{symbol.replace("^", "")}_data.csv')

                if use_cache and os.path.exists(cache_file):
                    # Check if cache is fresh (less than 1 day old)
                    cache_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
                    if cache_age < 86400:  # 24 hours
                        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        setattr(self, attr_name, data)
                        results[symbol] = {"status": "loaded_from_cache", "rows": len(data)}
                        continue

                # Fetch fresh data
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                if len(data) == 0:
                    results[symbol] = {"status": "error", "message": "No data returned"}
                    continue

                # Ensure datetime index
                data.index = pd.to_datetime(data.index)

                # Save to cache
                data.to_csv(cache_file)

                setattr(self, attr_name, data)
                results[symbol] = {"status": "success", "rows": len(data)}

            except Exception as e:
                results[symbol] = {"status": "error", "message": str(e)}

        return results

    def train(self, min_samples: int = 252) -> Dict:
        """
        Train the ML enhancement layer.

        Args:
            min_samples: Minimum samples required for training

        Returns:
            Training results dictionary
        """
        if self.qqq_data is None:
            return {"error": "No data loaded. Call load_data() first."}

        if len(self.qqq_data) < min_samples:
            return {"error": f"Insufficient data. Need {min_samples}, have {len(self.qqq_data)}"}

        # Train ML model
        result = self.ml_engine.fit(self.qqq_data)

        if result.get('success'):
            self.is_trained = True

        return result

    def calculate_gmi(self, new_highs_count: int = None, successful_nh_count: int = None) -> Dict:
        """
        Calculate the current GMI score and signal.

        Args:
            new_highs_count: Actual count of daily new highs (Component 2)
            successful_nh_count: Actual count of successful new highs (Component 1)

        Returns:
            GMI calculation results
        """
        if self.qqq_data is None or self.spy_data is None or self.xlk_data is None:
            return {"error": "Missing required data. Call load_data() first."}

        return self.gmi_calculator.calculate_gmi(
            qqq_data=self.qqq_data,
            spy_data=self.spy_data,
            fund_data=self.xlk_data,
            new_highs_count=new_highs_count,
            successful_nh_count=successful_nh_count
        )

    def calculate_supplementary(self) -> Dict:
        """
        Calculate all supplementary indicators.

        Returns:
            Supplementary indicator results
        """
        if self.qqq_data is None:
            return {"error": "No QQQ data loaded."}

        return self.supplementary.calculate_all_supplementary(
            qqq_data=self.qqq_data,
            vix_data=self.vix_data
        )

    def predict(self, breadth_data: Dict = None) -> Dict:
        """
        Generate next-day QQQ prediction.

        Args:
            breadth_data: Real market breadth data (new_highs, new_lows, etc.)
                         If not provided, GMI will use proxy calculations.

        Returns:
            Complete prediction dictionary with:
            - date: Prediction date
            - gmi: GMI score and signal
            - supplementary: Supporting indicators
            - ml_prediction: ML model output
            - volatility_regime: Current volatility environment
            - final_prediction: Combined prediction
            - interpretation: Human-readable summary
        """
        if self.qqq_data is None:
            return {"error": "No data loaded. Call load_data() first."}

        # Current date
        current_date = self.qqq_data.index[-1].strftime('%Y-%m-%d')
        prediction_for = (self.qqq_data.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')

        # Extract breadth data if available
        new_highs_count = None
        successful_nh_count = None
        if breadth_data:
            new_highs_count = breadth_data.get('new_highs_lows', {}).get('new_highs')
            successful_nh_count = breadth_data.get('successful_new_high', {}).get('successful_count')

        # 1. Calculate GMI with real breadth data (NO PROXIES when data available)
        gmi_result = self.calculate_gmi(
            new_highs_count=new_highs_count,
            successful_nh_count=successful_nh_count
        )

        # 2. Calculate supplementary indicators
        supplementary_result = self.calculate_supplementary()

        # 3. Detect volatility regime
        volatility_regime = self.ml_engine.detect_volatility_regime(self.qqq_data)

        # 4. Get ML prediction (if trained)
        if self.is_trained:
            ml_prediction = self.ml_engine.predict(self.qqq_data)
        else:
            # Use rule-based estimate
            ml_prediction = self._rule_based_estimate(gmi_result, supplementary_result)

        # 5. Combine signals
        final_prediction = self.signal_combiner.combine_signals(
            gmi_signal=gmi_result.get('signal', 'YELLOW'),
            gmi_score=gmi_result.get('gmi_score', 3),
            ml_prediction=ml_prediction,
            volatility_regime=volatility_regime,
            supplementary=supplementary_result
        )

        # 6. Generate interpretation
        interpretation = self._generate_interpretation(
            gmi_result, supplementary_result, final_prediction, volatility_regime
        )

        # Calculate target price
        current_price = round(self.qqq_data['Close'].iloc[-1], 2)
        expected_move_pct = final_prediction['expected_move_pct']
        target_price = round(current_price * (1 + expected_move_pct / 100), 2)

        # Build complete prediction
        prediction = {
            "as_of_date": current_date,
            "prediction_for": prediction_for,
            "current_price": current_price,
            "target_price": target_price,
            "gmi": {
                "score": gmi_result.get('gmi_score'),
                "signal": gmi_result.get('signal'),
                "action": gmi_result.get('signal_action'),
                "interpretation": get_gmi_interpretation(gmi_result.get('gmi_score', 3)),
                "components": gmi_result.get('components')
            },
            "supplementary": supplementary_result,
            "volatility_regime": volatility_regime,
            "ml_prediction": ml_prediction if self.is_trained else None,
            "final_prediction": {
                "direction": final_prediction['final_direction'],
                "expected_move_pct": final_prediction['expected_move_pct'],
                "target_price": target_price,
                "confidence": final_prediction['confidence'],
                "signals_aligned": final_prediction['signals_aligned']
            },
            "interpretation": interpretation,
            "model_trained": self.is_trained,
            "timestamp": datetime.now().isoformat()
        }

        self.last_prediction = prediction
        return prediction

    def _rule_based_estimate(self, gmi_result: Dict, supplementary: Dict) -> Dict:
        """
        Generate rule-based prediction when ML model isn't trained.

        Uses GMI and supplementary indicators to estimate direction and magnitude.
        """
        gmi_score = gmi_result.get('gmi_score', 3)
        gmi_signal = gmi_result.get('signal', 'YELLOW')

        # Direction based on GMI
        if gmi_signal == 'GREEN':
            direction = 'UP'
            base_confidence = 0.65 + (gmi_score - 5) * 0.05
        elif gmi_signal == 'RED':
            direction = 'DOWN'
            base_confidence = 0.65 + (2 - gmi_score) * 0.05
        else:
            direction = 'NEUTRAL'
            base_confidence = 0.50

        # Adjust based on supplementary
        stoch = supplementary.get('stochastic', {})
        ma_framework = supplementary.get('ma_framework', {})

        if stoch.get('signal') == 'OVERSOLD' and direction != 'DOWN':
            base_confidence += 0.05  # Bounce expected

        if ma_framework.get('ma_signal') == 'STRONG_BULLISH' and direction == 'UP':
            base_confidence += 0.05

        # Expected move based on historical volatility
        expected_move = 0.5 if direction == 'UP' else -0.5 if direction == 'DOWN' else 0

        return {
            "direction": direction,
            "direction_probability": base_confidence if direction == 'UP' else 1 - base_confidence,
            "expected_return_pct": expected_move,
            "confidence": min(0.85, base_confidence),
            "note": "Rule-based estimate (ML not trained)"
        }

    def _generate_interpretation(
        self,
        gmi: Dict,
        supplementary: Dict,
        final: Dict,
        volatility: Dict
    ) -> str:
        """Generate human-readable interpretation."""
        lines = []

        # GMI summary
        gmi_score = gmi.get('gmi_score', 3)
        gmi_signal = gmi.get('signal', 'YELLOW')
        lines.append(f"GMI: {gmi_score}/6 ({gmi_signal})")

        # Components breakdown
        components = gmi.get('components', {})
        positive_count = sum(1 for c in components.values() if c.get('positive', False))
        lines.append(f"  - {positive_count} of 6 components positive")

        # MA Framework
        ma = supplementary.get('ma_framework', {})
        if ma.get('ma_signal'):
            lines.append(f"MA Framework: {ma.get('ma_signal')}")
            lines.append(f"  - Price vs 10-week MA: {ma.get('pct_vs_10w', 0):+.1f}%")

        # Stochastic
        stoch = supplementary.get('stochastic', {})
        if stoch.get('signal'):
            lines.append(f"Stochastic (10.4.4): {stoch.get('k', 50):.0f} - {stoch.get('signal')}")

        # Volatility
        lines.append(f"Volatility Regime: {volatility.get('regime', 'UNKNOWN')} ({volatility.get('volatility_20d', 0):.1f}% annualized)")

        # Final prediction
        lines.append("")
        lines.append(f"PREDICTION: {final['final_direction']}")
        lines.append(f"  Expected move: {final['expected_move_pct']:+.2f}%")
        lines.append(f"  Confidence: {final['confidence']:.0%}")

        if final.get('signals_aligned'):
            lines.append("  (Signals aligned - higher confidence)")

        return "\n".join(lines)

    def get_trading_recommendation(self) -> Dict:
        """
        Get actionable trading recommendation.

        Returns:
            Trading recommendation with position sizing guidance
        """
        if self.last_prediction is None:
            pred = self.predict()
        else:
            pred = self.last_prediction

        gmi_score = pred['gmi']['score']
        direction = pred['final_prediction']['direction']
        confidence = pred['final_prediction']['confidence']
        vol_regime = pred['volatility_regime']['regime']
        risk_adj = pred['volatility_regime'].get('risk_adjustment', 1.0)

        # Base position size (as % of portfolio to allocate)
        if gmi_score >= 5:
            base_position = 1.0  # Full position
        elif gmi_score >= 3:
            base_position = 0.5  # Half position
        else:
            base_position = 0.0  # No long positions

        # Adjust for volatility
        adjusted_position = base_position * risk_adj

        # Determine action
        if direction == 'UP' and gmi_score >= 5:
            action = "BUY"
            instrument = "QQQ or TQQQ (leveraged)"
        elif direction == 'UP' and gmi_score >= 3:
            action = "HOLD/SMALL_BUY"
            instrument = "QQQ"
        elif direction == 'DOWN' and gmi_score <= 2:
            action = "SELL/SHORT"
            instrument = "SQQQ or cash"
        elif direction == 'DOWN':
            action = "REDUCE"
            instrument = "Trim positions"
        else:
            action = "HOLD"
            instrument = "Maintain current"

        return {
            "action": action,
            "instrument": instrument,
            "position_size_pct": round(adjusted_position * 100, 0),
            "reasoning": {
                "gmi_score": gmi_score,
                "direction": direction,
                "confidence": confidence,
                "volatility_regime": vol_regime
            },
            "risk_management": {
                "stop_loss": "Below 10-week MA" if action == "BUY" else "N/A",
                "take_profit": "Trail with 30-week MA" if action == "BUY" else "N/A"
            }
        }

    def save_prediction(self, filepath: str = None) -> str:
        """Save the last prediction to a JSON file."""
        if self.last_prediction is None:
            self.predict()

        if filepath is None:
            date_str = datetime.now().strftime('%Y%m%d')
            filepath = os.path.join(
                os.path.dirname(__file__), '..', 'outputs',
                f'prediction_{date_str}.json'
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.last_prediction, f, indent=2, default=str)

        return filepath

    def run_backtest(
        self,
        start_date: str,
        end_date: str = None
    ) -> Dict:
        """
        Run a simple backtest of the model.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date (default: yesterday)

        Returns:
            Backtest results with accuracy metrics
        """
        if self.qqq_data is None:
            return {"error": "No data loaded."}

        # Filter data to backtest period
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else datetime.now() - timedelta(days=1)

        mask = (self.qqq_data.index >= start) & (self.qqq_data.index <= end)
        test_indices = self.qqq_data.index[mask]

        if len(test_indices) < 50:
            return {"error": "Insufficient data for backtest"}

        results = []
        correct_direction = 0
        total_predictions = 0

        for i, date in enumerate(test_indices[:-1]):  # Exclude last day
            idx = self.qqq_data.index.get_loc(date)

            # Use data up to this point
            historical = self.qqq_data.iloc[:idx+1]

            if len(historical) < 100:
                continue

            # Calculate GMI with historical data
            gmi_result = self.gmi_calculator.calculate_gmi(
                qqq_data=historical,
                spy_data=self.spy_data.iloc[:idx+1],
                fund_data=self.xlk_data.iloc[:idx+1]
            )

            # Simple prediction based on GMI
            gmi_score = gmi_result.get('gmi_score', 3)
            predicted_direction = 'UP' if gmi_score >= 4 else 'DOWN' if gmi_score <= 2 else 'NEUTRAL'

            # Actual next-day return
            actual_return = (
                self.qqq_data['Close'].iloc[idx+1] /
                self.qqq_data['Close'].iloc[idx]
            ) - 1

            actual_direction = 'UP' if actual_return > 0 else 'DOWN'

            # Score
            if predicted_direction != 'NEUTRAL':
                total_predictions += 1
                if predicted_direction == actual_direction:
                    correct_direction += 1

            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'gmi_score': gmi_score,
                'predicted': predicted_direction,
                'actual': actual_direction,
                'actual_return_pct': round(actual_return * 100, 2),
                'correct': predicted_direction == actual_direction or predicted_direction == 'NEUTRAL'
            })

        accuracy = correct_direction / total_predictions if total_predictions > 0 else 0

        return {
            "period": f"{start_date} to {end_date or 'present'}",
            "total_days": len(results),
            "predictions_made": total_predictions,
            "correct_predictions": correct_direction,
            "directional_accuracy": round(accuracy * 100, 2),
            "detailed_results": results[-20:]  # Last 20 for brevity
        }


def main():
    """Main entry point for running the model."""
    print("=" * 60)
    print("QQQ Wishing Wealth Prediction Model")
    print("Based on Dr. Eric Wish's methodology")
    print("=" * 60)
    print()

    # Initialize model
    model = QQQWishingWealthModel()

    # Load data
    print("Loading market data...")
    load_result = model.load_data()
    for symbol, status in load_result.items():
        print(f"  {symbol}: {status.get('status')} ({status.get('rows', 0)} rows)")
    print()

    # Train ML component
    print("Training ML enhancement layer...")
    train_result = model.train()
    if train_result.get('success'):
        print(f"  Training samples: {train_result.get('training_samples')}")
        print(f"  Direction accuracy: {train_result.get('direction_accuracy'):.1%}")
    else:
        print(f"  Warning: {train_result.get('error', 'Training failed')}")
    print()

    # Generate prediction
    print("Generating prediction...")
    prediction = model.predict()
    print()

    # Display results
    print("=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print()
    print(f"As of: {prediction['as_of_date']}")
    print(f"Prediction for: {prediction['prediction_for']}")
    print(f"Current QQQ Price: ${prediction['current_price']}")
    print()
    print(prediction['interpretation'])
    print()

    # Trading recommendation
    print("=" * 60)
    print("TRADING RECOMMENDATION")
    print("=" * 60)
    rec = model.get_trading_recommendation()
    print(f"Action: {rec['action']}")
    print(f"Instrument: {rec['instrument']}")
    print(f"Position Size: {rec['position_size_pct']:.0f}% of portfolio")
    print()

    # Save prediction
    filepath = model.save_prediction()
    print(f"Prediction saved to: {filepath}")

    return prediction


if __name__ == "__main__":
    main()
