"""
QQQ Wishing Wealth Model

A unified model for predicting next-day QQQ price movements based on
Dr. Eric Wish's Wishing Wealth methodology (wishingwealthblog.com).

Components:
- GMI Calculator: 6-component General Market Index
- Supplementary Indicators: Stochastic, T2108, Bollinger, MA framework
- ML Enhancement: Direction/magnitude prediction with confidence
- Backtester: Historical performance analysis
- Daily Runner: Scheduled execution with email notifications

Usage:
    from src import QQQWishingWealthModel

    model = QQQWishingWealthModel()
    model.load_data()
    model.train()
    prediction = model.predict()
"""

from .gmi_calculator import GMICalculator, get_gmi_interpretation
from .supplementary_indicators import SupplementaryIndicators
from .ml_enhancement import MLEnhancement, SignalCombiner
from .qqq_wishing_wealth_model import QQQWishingWealthModel
from .backtester import Backtester, BacktestResults, WalkForwardAnalyzer
from .daily_runner import DailyRunner, PredictionDatabase, EmailNotifier

__all__ = [
    'QQQWishingWealthModel',
    'GMICalculator',
    'SupplementaryIndicators',
    'MLEnhancement',
    'SignalCombiner',
    'Backtester',
    'BacktestResults',
    'WalkForwardAnalyzer',
    'DailyRunner',
    'PredictionDatabase',
    'EmailNotifier',
    'get_gmi_interpretation'
]

__version__ = '1.0.0'
