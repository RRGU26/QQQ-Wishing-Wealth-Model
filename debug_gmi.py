"""Debug GMI components to identify divergence from blog."""

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from gmi_calculator import GMICalculator

def debug_gmi_components():
    print("=" * 70)
    print("GMI COMPONENT BREAKDOWN - Debugging Divergence")
    print("=" * 70)
    print()

    # Fetch fresh data (no cache)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print("Fetching FRESH market data (bypassing cache)...")
    qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False, auto_adjust=True)
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
    xlk = yf.download('XLK', start=start_date, end=end_date, progress=False, auto_adjust=True)

    # Handle multi-index columns from yfinance
    for df in [qqq, spy, xlk]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel('Ticker')

    print(f"Data through: {qqq.index[-1].strftime('%Y-%m-%d')}")
    print()

    # Initialize calculator
    calc = GMICalculator()

    # Component by component analysis
    print("-" * 70)
    print("COMPONENT 1: Successful New High Index")
    print("-" * 70)
    c1_result = calc._proxy_successful_new_high(qqq)
    print(f"  Result: {'POSITIVE' if c1_result else 'NEGATIVE'}")
    print(f"  (Using proxy - checks QQQ making higher highs)")
    current_high = qqq['High'].iloc[-1]
    high_10d_ago = qqq['High'].iloc[-10]
    high_20d = qqq['High'].iloc[-20:].max()
    print(f"  Current High: ${current_high:.2f}")
    print(f"  High 10 days ago: ${high_10d_ago:.2f}")
    print(f"  20-day High: ${high_20d:.2f}")
    print(f"  Current vs 20d high: {(current_high/high_20d - 1)*100:.2f}%")
    print()

    print("-" * 70)
    print("COMPONENT 2: Daily New Highs Count")
    print("-" * 70)
    c2_result = calc._proxy_daily_new_highs(qqq)
    print(f"  Result: {'POSITIVE' if c2_result else 'NEGATIVE'}")
    print(f"  (Using proxy - checks 5d and 20d momentum)")
    returns_5d = (qqq['Close'].iloc[-1] / qqq['Close'].iloc[-5]) - 1
    returns_20d = (qqq['Close'].iloc[-1] / qqq['Close'].iloc[-20]) - 1
    print(f"  5-day return: {returns_5d*100:.2f}%")
    print(f"  20-day return: {returns_20d*100:.2f}%")
    print()

    print("-" * 70)
    print("COMPONENT 3: Daily QQQ Index")
    print("-" * 70)
    c3_result, c3_details = calc.calculate_daily_qqq_index(qqq)
    print(f"  Result: {'POSITIVE' if c3_result else 'NEGATIVE'}")
    print(f"  QQQ Close: ${c3_details.get('close', 0):.2f}")
    print(f"  50-day MA: ${c3_details.get('ma_50', 0):.2f}")
    print(f"  10-day MA: ${c3_details.get('ma_10', 0):.2f}")
    print(f"  % above 50-day MA: {c3_details.get('pct_above_ma50', 0):.2f}%")
    print(f"  Condition: Close > 50MA AND 10MA > 50MA")
    close = c3_details.get('close', 0)
    ma50 = c3_details.get('ma_50', 0)
    ma10 = c3_details.get('ma_10', 0)
    print(f"    Close > 50MA: {close > ma50} (${close:.2f} > ${ma50:.2f})")
    print(f"    10MA > 50MA: {ma10 > ma50} (${ma10:.2f} > ${ma50:.2f})")
    print()

    print("-" * 70)
    print("COMPONENT 4: Daily SPY Index")
    print("-" * 70)
    c4_result, c4_details = calc.calculate_daily_spy_index(spy)
    print(f"  Result: {'POSITIVE' if c4_result else 'NEGATIVE'}")
    print(f"  SPY Close: ${c4_details.get('close', 0):.2f}")
    print(f"  50-day MA: ${c4_details.get('ma_50', 0):.2f}")
    print(f"  % above 50-day MA: {c4_details.get('pct_above_ma50', 0):.2f}%")
    print(f"  Condition: Close > 50MA")
    print()

    print("-" * 70)
    print("COMPONENT 5: Weekly QQQ Index")
    print("-" * 70)
    c5_result, c5_details = calc.calculate_weekly_qqq_index(qqq)
    print(f"  Result: {'POSITIVE' if c5_result else 'NEGATIVE'}")
    if 'error' in c5_details:
        print(f"  Error: {c5_details['error']}")
    else:
        print(f"  Weekly Close: ${c5_details.get('close_weekly', 0):.2f}")
        print(f"  10-week MA: ${c5_details.get('ma_10w', 0):.2f}")
        print(f"  30-week MA: ${c5_details.get('ma_30w', 0):.2f}")
        print(f"  MA Spread: {c5_details.get('ma_spread_pct', 0):.2f}%")
        print(f"  Condition: 10-week MA > 30-week MA")
    print()

    print("-" * 70)
    print("COMPONENT 6: IBD Fund Index (XLK proxy)")
    print("-" * 70)
    c6_result, c6_details = calc.calculate_ibd_fund_index(xlk)
    print(f"  Result: {'POSITIVE' if c6_result else 'NEGATIVE'}")
    print(f"  XLK Close: ${c6_details.get('close', 0):.2f}")
    print(f"  50-day MA: ${c6_details.get('ma_50', 0):.2f}")
    print(f"  % above 50-day MA: {c6_details.get('pct_above_ma50', 0):.2f}%")
    print(f"  Condition: Close > 50MA")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    components = [
        ("1. Successful New High", c1_result),
        ("2. Daily New Highs", c2_result),
        ("3. Daily QQQ Index", c3_result),
        ("4. Daily SPY Index", c4_result),
        ("5. Weekly QQQ Index", c5_result),
        ("6. IBD Fund Index", c6_result),
    ]

    score = sum(1 for _, result in components if result)

    for name, result in components:
        status = "[+]" if result else "[-]"
        print(f"  {status} {name}")

    print()
    print(f"  OUR GMI SCORE: {score}/6")
    print(f"  BLOG GMI SCORE: 4/6 (per latest fetch)")
    print(f"  DIFFERENCE: {abs(score - 4)} points")
    print()

    # Identify likely issues
    print("-" * 70)
    print("DIAGNOSIS")
    print("-" * 70)

    negative_components = [name for name, result in components if not result]
    if len(negative_components) > 2:
        print("  Components scoring negative that may differ from blog:")
        for comp in negative_components:
            print(f"    - {comp}")
        print()
        print("  LIKELY CAUSE OF DIVERGENCE:")
        print("    The blog uses ACTUAL market breadth data (new highs/lows counts)")
        print("    from sources like Barchart or IBD. Our model uses QQQ-based proxies")
        print("    for components 1 and 2, which may underestimate breadth in narrow markets.")
        print()
        print("  RECOMMENDATION:")
        print("    Components 1 & 2 proxy calculations are too conservative.")
        print("    The blog likely sees more stocks at new highs than our QQQ proxy suggests.")

if __name__ == "__main__":
    debug_gmi_components()
