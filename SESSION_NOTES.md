# Session Notes - December 23, 2025

## Current State
- Model is running daily at 4:10 PM via Windows Task Scheduler
- GMI aligned with blog (6/6 vs 5/6 - only 1 point difference)
- Stock screener finding 20+ setups when GMI is GREEN

## Today's Report (Dec 23)
- GMI: 6/6 GREEN
- Blog: 5/6 GREEN (minor difference)
- VIX: 14.1 (low volatility)

### Top Picks
| Pattern | Stock | Setup |
|---------|-------|-------|
| GLB | ADI | Breaking out +8.0% above ATH |
| Blue Dot | CRWD | Stoch 17, down 11% from high |
| RWB | GOOGL | 10/10 score, +1.8% above 10-day MA |

## User Considering
- GOOGL position - RWB setup, strong trend, close to 10-day MA support

## GMI Difference Analysis
1-point difference between our model (6/6) and blog (5/6) likely due to:
- **Successful New High Index**: We use FINVIZ proxy, blog uses proprietary calc
- **IBD Mutual Fund Index**: We use XLK as proxy, blog uses actual IBD index

Both agree on GREEN signal - difference is minor.

## Key Files
- `run_daily.py` - Main daily report runner
- `src/stock_screener.py` - GLB, Blue Dot, RWB pattern detection
- `src/gmi_calculator.py` - GMI calculation (6 components)
- `setup_scheduler.bat` - Windows Task Scheduler setup

## Potential Improvements
- Get actual IBD Mutual Fund Index data for more accurate GMI
- Add more stocks to screening universe
- Track trade outcomes for backtesting patterns
