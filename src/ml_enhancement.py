"""
ML Enhancement Layer for Wishing Wealth QQQ Model

This layer adds machine learning predictions on top of the rule-based
GMI and supplementary indicators to:
1. Estimate expected price movement percentage
2. Calculate prediction confidence
3. Detect volatility regimes
4. Combine signals with historical accuracy weighting
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class MLEnhancement:
    """
    Machine learning enhancement layer for QQQ predictions.

    Features engineered from Wishing Wealth indicators plus
    proven features from the Trading-Dashboard repo.
    """

    def __init__(self):
        self.direction_model = None
        self.magnitude_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML model.

        Combines Wishing Wealth indicators with proven features
        from the Trading-Dashboard repo.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        df = data.copy()

        # Price-based features
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)

        # Moving averages
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_10'] = df['Close'].rolling(10).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()

        # Price vs MAs (key Wishing Wealth indicators)
        df['price_vs_ma10'] = (df['Close'] / df['ma_10']) - 1
        df['price_vs_ma20'] = (df['Close'] / df['ma_20']) - 1
        df['price_vs_ma50'] = (df['Close'] / df['ma_50']) - 1

        # MA crossovers
        df['ma_5_10_cross'] = (df['ma_5'] > df['ma_10']).astype(int)
        df['ma_10_20_cross'] = (df['ma_10'] > df['ma_20']).astype(int)
        df['ma_20_50_cross'] = (df['ma_20'] > df['ma_50']).astype(int)

        # Volatility features
        df['volatility_5d'] = df['returns_1d'].rolling(5).std() * np.sqrt(252)
        df['volatility_20d'] = df['returns_1d'].rolling(20).std() * np.sqrt(252)
        df['vol_ratio'] = df['volatility_5d'] / df['volatility_20d']

        # RSI (14-period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Stochastic (10.4.4 - Wishing Wealth settings)
        low_10 = df['Low'].rolling(10).min()
        high_10 = df['High'].rolling(10).max()
        df['stoch_k'] = 100 * (df['Close'] - low_10) / (high_10 - low_10 + 0.0001)
        df['stoch_k'] = df['stoch_k'].rolling(4).mean()  # Smooth
        df['stoch_d'] = df['stoch_k'].rolling(4).mean()

        # Bollinger Bands position
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_mid + (2 * bb_std)
        bb_lower = bb_mid - (2 * bb_std)
        df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 0.0001)
        df['bb_width'] = (bb_upper - bb_lower) / bb_mid

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()

        # High-Low range
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        df['range_vs_avg'] = df['daily_range'] / df['daily_range'].rolling(20).mean()

        # Momentum
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5)
        df['momentum_10d'] = df['Close'] / df['Close'].shift(10)
        df['momentum_acceleration'] = df['momentum_5d'] - df['momentum_5d'].shift(5)

        # Distance from recent highs/lows
        df['high_20d'] = df['High'].rolling(20).max()
        df['low_20d'] = df['Low'].rolling(20).min()
        df['dist_from_high'] = (df['high_20d'] - df['Close']) / df['Close']
        df['dist_from_low'] = (df['Close'] - df['low_20d']) / df['Close']

        # Trend strength
        df['trend_strength'] = abs(df['ma_10'] - df['ma_50']) / df['ma_50']

        # Target: Next day's return
        df['target_return'] = df['Close'].shift(-1) / df['Close'] - 1
        df['target_direction'] = (df['target_return'] > 0).astype(int)

        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model."""
        return [
            'returns_1d', 'returns_3d', 'returns_5d', 'returns_10d', 'returns_20d',
            'price_vs_ma10', 'price_vs_ma20', 'price_vs_ma50',
            'ma_5_10_cross', 'ma_10_20_cross', 'ma_20_50_cross',
            'volatility_5d', 'volatility_20d', 'vol_ratio',
            'rsi_14', 'stoch_k', 'stoch_d',
            'bb_position', 'bb_width',
            'macd', 'macd_signal', 'macd_hist',
            'volume_ratio', 'volume_trend',
            'daily_range', 'range_vs_avg',
            'momentum_5d', 'momentum_10d', 'momentum_acceleration',
            'dist_from_high', 'dist_from_low',
            'trend_strength'
        ]

    def fit(self, data: pd.DataFrame) -> Dict:
        """
        Train the ML models on historical data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with training metrics
        """
        # Engineer features
        df = self.engineer_features(data)
        feature_cols = self.get_feature_columns()

        # Remove rows with NaN
        df_clean = df.dropna(subset=feature_cols + ['target_return', 'target_direction'])

        if len(df_clean) < 100:
            return {"error": "Insufficient data for training", "rows": len(df_clean)}

        X = df_clean[feature_cols]
        y_direction = df_clean['target_direction']
        y_magnitude = df_clean['target_return']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train direction classifier (ensemble)
        self.direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.direction_model.fit(X_scaled, y_direction)

        # Train magnitude regressor (voting ensemble)
        rf1 = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf2 = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=43)

        self.magnitude_model = VotingRegressor([
            ('rf1', rf1),
            ('rf2', rf2)
        ])
        self.magnitude_model.fit(X_scaled, y_magnitude)

        self.feature_names = feature_cols
        self.is_fitted = True

        # Calculate training metrics
        dir_accuracy = self.direction_model.score(X_scaled, y_direction)

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.direction_model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            "success": True,
            "training_samples": len(df_clean),
            "direction_accuracy": round(dir_accuracy, 4),
            "top_features": importance.head(10).to_dict('records')
        }

    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Generate ML predictions for the latest data point.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            return {"error": "Model not fitted. Call fit() first."}

        # Engineer features
        df = self.engineer_features(data)
        feature_cols = self.get_feature_columns()

        # Get latest complete row
        latest = df[feature_cols].iloc[-1:].copy()

        if latest.isna().any().any():
            return {"error": "Missing feature values for prediction"}

        # Scale features
        X_scaled = self.scaler.transform(latest)

        # Predict direction
        direction_proba = self.direction_model.predict_proba(X_scaled)[0]
        direction_pred = self.direction_model.predict(X_scaled)[0]

        # Predict magnitude
        magnitude_pred = self.magnitude_model.predict(X_scaled)[0]

        # Calculate confidence
        confidence = max(direction_proba)

        return {
            "direction": "UP" if direction_pred == 1 else "DOWN",
            "direction_probability": round(direction_proba[1], 4),  # Prob of UP
            "expected_return_pct": round(magnitude_pred * 100, 4),
            "confidence": round(confidence, 4)
        }

    def detect_volatility_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect current volatility regime.

        Regimes:
        - LOW: Volatility below 15% annualized
        - NORMAL: 15-25% annualized
        - ELEVATED: 25-35% annualized
        - HIGH: Above 35% annualized

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with regime classification
        """
        if len(data) < 30:
            return {"regime": "UNKNOWN", "error": "Insufficient data"}

        returns = data['Close'].pct_change()

        # Current volatility (20-day)
        vol_20d = returns.iloc[-20:].std() * np.sqrt(252) * 100

        # Historical context (252-day if available)
        if len(data) >= 252:
            vol_252d = returns.iloc[-252:].std() * np.sqrt(252) * 100
            vol_percentile = (returns.rolling(20).std() < returns.iloc[-20:].std()).iloc[-252:].mean() * 100
        else:
            vol_252d = vol_20d
            vol_percentile = 50

        # Determine regime
        if vol_20d < 15:
            regime = "LOW"
            risk_adjustment = 1.2  # Can take more risk
        elif vol_20d < 25:
            regime = "NORMAL"
            risk_adjustment = 1.0
        elif vol_20d < 35:
            regime = "ELEVATED"
            risk_adjustment = 0.7  # Reduce exposure
        else:
            regime = "HIGH"
            risk_adjustment = 0.5  # Significantly reduce

        # Volatility trend
        vol_5d = returns.iloc[-5:].std() * np.sqrt(252) * 100
        vol_expanding = vol_5d > vol_20d

        return {
            "regime": regime,
            "volatility_20d": round(vol_20d, 2),
            "volatility_252d": round(vol_252d, 2),
            "vol_percentile": round(vol_percentile, 1),
            "vol_expanding": vol_expanding,
            "risk_adjustment": risk_adjustment
        }


class SignalCombiner:
    """
    Combine GMI signals with ML predictions using confidence weighting.
    """

    def __init__(self):
        # Historical accuracy weights (can be updated with actual performance)
        self.gmi_weight = 0.6  # Rule-based GMI
        self.ml_weight = 0.4   # ML predictions

    def combine_signals(
        self,
        gmi_signal: str,
        gmi_score: int,
        ml_prediction: Dict,
        volatility_regime: Dict,
        supplementary: Dict
    ) -> Dict:
        """
        Combine all signals into final prediction.

        Args:
            gmi_signal: GMI signal (GREEN, YELLOW, RED)
            gmi_score: GMI score (0-6)
            ml_prediction: ML model predictions
            volatility_regime: Volatility regime info
            supplementary: Supplementary indicator signals

        Returns:
            Combined prediction with confidence
        """
        # Convert GMI signal to numeric
        gmi_numeric = {
            "GREEN": 1.0,
            "YELLOW": 0.0,
            "RED": -1.0
        }.get(gmi_signal, 0)

        # Scale GMI score to -1 to 1
        gmi_scaled = (gmi_score - 3) / 3  # 0->-1, 3->0, 6->1

        # ML direction as numeric
        ml_direction = 1 if ml_prediction.get('direction') == 'UP' else -1
        ml_confidence = ml_prediction.get('confidence', 0.5)

        # Combine signals
        combined_direction = (
            self.gmi_weight * gmi_scaled +
            self.ml_weight * ml_direction * ml_confidence
        )

        # Adjust for volatility regime
        risk_adj = volatility_regime.get('risk_adjustment', 1.0)
        combined_direction *= risk_adj

        # Boost confidence if signals align
        signals_aligned = (
            (gmi_signal == "GREEN" and ml_prediction.get('direction') == 'UP') or
            (gmi_signal == "RED" and ml_prediction.get('direction') == 'DOWN')
        )

        # Check supplementary alignment
        stoch = supplementary.get('stochastic', {})
        ma_framework = supplementary.get('ma_framework', {})

        supplementary_bullish = (
            stoch.get('signal') != 'OVERBOUGHT' and
            ma_framework.get('ma_signal') in ['STRONG_BULLISH', 'BULLISH']
        )
        supplementary_bearish = (
            stoch.get('signal') != 'OVERSOLD' and
            ma_framework.get('ma_signal') == 'BEARISH'
        )

        # Calculate final confidence
        base_confidence = (abs(gmi_scaled) * 0.5 + ml_confidence * 0.5)

        if signals_aligned:
            base_confidence *= 1.15  # 15% boost for alignment

        if (combined_direction > 0 and supplementary_bullish) or \
           (combined_direction < 0 and supplementary_bearish):
            base_confidence *= 1.1  # 10% boost for supplementary alignment

        final_confidence = min(0.95, base_confidence)  # Cap at 95%

        # Final direction
        if combined_direction > 0.2:
            final_direction = "UP"
        elif combined_direction < -0.2:
            final_direction = "DOWN"
        else:
            final_direction = "NEUTRAL"

        # Expected move - use GMI-based minimum when ML prediction is too small
        expected_move = ml_prediction.get('expected_return_pct', 0)

        # Set minimum expected move based on GMI score
        # Strong signals should have meaningful expected moves
        gmi_min_moves = {
            6: 0.75,  # GMI 6/6 = expect at least 0.75% move
            5: 0.50,  # GMI 5/6 = expect at least 0.50% move
            4: 0.30,  # GMI 4/6 = expect at least 0.30% move
            3: 0.15,  # GMI 3/6 = neutral
            2: 0.30,  # GMI 2/6 = expect down move
            1: 0.50,  # GMI 1/6 = expect down move
            0: 0.75,  # GMI 0/6 = expect at least 0.75% down
        }
        min_move = gmi_min_moves.get(gmi_score, 0.25)

        # Use the larger of ML prediction or GMI minimum
        if abs(expected_move) < min_move:
            expected_move = min_move if final_direction == "UP" else -min_move

        if final_direction == "DOWN":
            expected_move = -abs(expected_move)
        elif final_direction == "NEUTRAL":
            expected_move = 0

        return {
            "final_direction": final_direction,
            "expected_move_pct": round(expected_move, 3),
            "confidence": round(final_confidence, 3),
            "signals_aligned": signals_aligned,
            "combined_score": round(combined_direction, 3),
            "risk_adjusted": risk_adj != 1.0,
            "components": {
                "gmi_contribution": round(gmi_scaled * self.gmi_weight, 3),
                "ml_contribution": round(ml_direction * ml_confidence * self.ml_weight, 3)
            }
        }
