# QQQ Wishing Wealth Prediction Model

A standalone model for predicting next-day QQQ price movements based on Dr. Eric Wish's [Wishing Wealth](https://wishingwealthblog.com/) methodology, enhanced with machine learning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run prediction
python run_prediction.py
```

## Features

### Core Strategy (Wishing Wealth GMI)
The model calculates a 6-component General Market Index (GMI) scoring 0-6:

| Component | What It Measures | Threshold |
|-----------|------------------|-----------|
| Successful New High Index | Stocks at 52-week high 10 days ago still higher | >100 stocks |
| Daily New Highs | Breadth of market strength | >100 new highs |
| Daily QQQ | QQQ above 10-week MA | Uptrend |
| Daily SPY | SPY above 10-week MA | Uptrend |
| Weekly QQQ | 10-week MA > 30-week MA | Bullish trend |
| IBD Fund Proxy | Growth fund (XLK) above 50-day MA | Above MA |

**Signal Interpretation:**
- **GMI 5-6 (GREEN):** Go long aggressively
- **GMI 3-4 (YELLOW):** Be defensive/nimble
- **GMI 0-2 (RED):** Go to cash or short

### Supplementary Indicators
- **Stochastic (10.4.4):** Oversold bounce detection (<20)
- **T2108 Proxy:** Market breadth (<20% = oversold)
- **Bollinger Bands:** Blue Dot setup detection
- **MA Framework:** 10-week vs 30-week trend confirmation

### ML Enhancement
- Random Forest direction classifier
- Voting ensemble for magnitude prediction
- Volatility regime detection
- Confidence-weighted signal combination

## Usage

### Quick Prediction
```bash
python run_prediction.py
```

### Run Backtest
```bash
python run_prediction.py --backtest
```

### Daily Run (with database logging)
```bash
python run_prediction.py --daily
```

### Configure Email Notifications
```bash
python run_prediction.py --setup-email
```

### 4:05 PM Options Trading Report
```bash
python run_405.py
```
Generates complete report with:
- GMI score with real breadth data from FINVIZ
- Options recommendation (calls/puts/no trade)
- Blog comparison with wishingwealthblog.com
- Database storage for tracking

### Schedule Daily Runs (Windows)
Run as Administrator:
```powershell
.\setup_scheduler.bat
```
This schedules the model to run at **4:05 PM ET** on weekdays for the 4:00-4:15 PM options trading window.

## Project Structure

```
QQQ-Wishing-Wealth-Model/
├── run_prediction.py          # Main entry point
├── requirements.txt           # Dependencies
├── setup_scheduler.bat        # Windows Task Scheduler setup
├── setup_scheduler.ps1        # PowerShell scheduler setup
├── config/
│   └── config.json            # Email and model settings
├── src/
│   ├── __init__.py
│   ├── gmi_calculator.py      # GMI (6 components)
│   ├── supplementary_indicators.py
│   ├── ml_enhancement.py      # ML layer
│   ├── qqq_wishing_wealth_model.py  # Main model
│   ├── backtester.py          # Historical testing
│   └── daily_runner.py        # Scheduled execution
├── data/                      # Cached market data
├── outputs/                   # Predictions and reports
│   └── backtests/             # Backtest results
└── logs/                      # Daily run logs
```

## Output Example

```
============================================================
RESULTS
============================================================

Date: 2024-12-16
Prediction For: 2024-12-17
Current QQQ: $527.45

GMI: 5/6 (GREEN)
  - 5 of 6 components positive
MA Framework: STRONG_BULLISH
  - Price vs 10-week MA: +3.2%
Stochastic (10.4.4): 65 - NEUTRAL
Volatility Regime: NORMAL (18.5% annualized)

PREDICTION: UP
  Expected move: +0.45%
  Confidence: 72%
  (Signals aligned - higher confidence)

============================================================
TRADING RECOMMENDATION
============================================================
Action: BUY
Instrument: QQQ or TQQQ (leveraged)
Position Size: 100%
```

## Configuration

Edit `config/config.json`:

```json
{
  "email": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your-email@gmail.com",
    "sender_password": "your-app-password",
    "recipients": ["recipient@email.com"]
  },
  "model": {
    "lookback_days": 730,
    "gmi_entry_threshold": 5,
    "gmi_exit_threshold": 2
  }
}
```

## Strategy Reference

Based on [Wishing Wealth Blog](https://wishingwealthblog.com/):
- [About the GMI](https://wishingwealthblog.com/2005/04/general-market-index-gmi/)
- [Green Line Breakout](https://wishingwealthblog.com/2018/05/green-line-breakout-glb-explained-gmi-remains-green/)
- [Stochastic Bounces](https://www.wishingwealthblog.com/2011/10/4th-day-of-qqq-short-term-down-trend-stochastic-signals-bounces/)

## Disclaimer

This model is for educational purposes only. Past performance does not guarantee future results. Always do your own research and consult a financial advisor before making investment decisions.
