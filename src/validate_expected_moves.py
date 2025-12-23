"""
Validate Expected Moves by GMI Score

Calculates actual historical next-day returns for each GMI score
to validate the model's expected move assumptions.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from gmi_calculator import GMICalculator

def main():
    print('Loading 3 years of data...')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)

    qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False)
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    xlk = yf.download('XLK', start=start_date, end=end_date, progress=False)

    # Fix multi-index columns
    for df in [qqq, spy, xlk]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel('Ticker')

    print(f'Data loaded: {len(qqq)} days')
    print()

    # Calculate GMI for each day and track next-day returns
    gmi_calc = GMICalculator()
    results = []

    print('Calculating GMI scores for each day...')
    for i in range(150, len(qqq) - 1):  # Need 150 days history, exclude last day
        try:
            hist_qqq = qqq.iloc[:i+1]
            hist_spy = spy.iloc[:i+1]
            hist_xlk = xlk.iloc[:i+1]

            gmi_result = gmi_calc.calculate_gmi(
                qqq_data=hist_qqq,
                spy_data=hist_spy,
                fund_data=hist_xlk
            )
            gmi_score = gmi_result.get('gmi_score', 3)

            # Next day return
            next_day_return = (qqq['Close'].iloc[i+1] / qqq['Close'].iloc[i] - 1) * 100

            results.append({
                'date': qqq.index[i],
                'gmi_score': gmi_score,
                'next_day_return_pct': next_day_return
            })
        except Exception as e:
            pass

    df = pd.DataFrame(results)
    print(f'Analyzed {len(df)} trading days')
    print()

    # Calculate statistics for each GMI score
    print('=' * 70)
    print('ACTUAL HISTORICAL EXPECTED MOVES BY GMI SCORE')
    print('=' * 70)
    print()
    print(f"{'GMI':>4} {'Days':>6} {'Avg Move':>10} {'Median':>10} {'Win Rate':>10} {'Std Dev':>10}")
    print('-' * 70)

    for gmi in range(7):
        subset = df[df['gmi_score'] == gmi]
        if len(subset) > 0:
            avg = subset['next_day_return_pct'].mean()
            median = subset['next_day_return_pct'].median()
            win_rate = (subset['next_day_return_pct'] > 0).mean() * 100
            std = subset['next_day_return_pct'].std()
            print(f"{gmi:>4} {len(subset):>6} {avg:>+10.3f}% {median:>+10.3f}% {win_rate:>9.1f}% {std:>10.3f}%")
        else:
            print(f"{gmi:>4} {0:>6} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print()
    print('=' * 70)
    print('GROUPED BY SIGNAL')
    print('=' * 70)
    print()

    # GREEN (5-6)
    green = df[df['gmi_score'] >= 5]
    yellow = df[(df['gmi_score'] >= 3) & (df['gmi_score'] <= 4)]
    red = df[df['gmi_score'] <= 2]

    print(f"{'Signal':>12} {'Days':>6} {'Avg Move':>10} {'Median':>10} {'Win Rate':>10}")
    print('-' * 70)

    if len(green) > 0:
        print(f"{'GREEN (5-6)':>12} {len(green):>6} {green['next_day_return_pct'].mean():>+10.3f}% {green['next_day_return_pct'].median():>+10.3f}% {(green['next_day_return_pct'] > 0).mean()*100:>9.1f}%")
    if len(yellow) > 0:
        print(f"{'YELLOW (3-4)':>12} {len(yellow):>6} {yellow['next_day_return_pct'].mean():>+10.3f}% {yellow['next_day_return_pct'].median():>+10.3f}% {(yellow['next_day_return_pct'] > 0).mean()*100:>9.1f}%")
    if len(red) > 0:
        print(f"{'RED (0-2)':>12} {len(red):>6} {red['next_day_return_pct'].mean():>+10.3f}% {red['next_day_return_pct'].median():>+10.3f}% {(red['next_day_return_pct'] > 0).mean()*100:>9.1f}%")

    print()
    print('=' * 70)
    print('COMPARISON: MODEL ASSUMPTIONS vs ACTUAL HISTORY')
    print('=' * 70)
    print()
    print(f"{'GMI':>4} {'Model Expects':>15} {'Actual Avg':>15} {'Difference':>15} {'Assessment':>15}")
    print('-' * 70)

    model_expects = {6: 0.75, 5: 0.50, 4: 0.30, 3: 0.15, 2: -0.30, 1: -0.50, 0: -0.75}

    for gmi in range(7):
        subset = df[df['gmi_score'] == gmi]
        if len(subset) > 5:  # Need enough samples
            actual = subset['next_day_return_pct'].mean()
            expected = model_expects[gmi]
            diff = actual - expected

            if abs(diff) < 0.1:
                assessment = "ACCURATE"
            elif (expected > 0 and actual > 0) or (expected < 0 and actual < 0):
                assessment = "DIRECTION OK"
            else:
                assessment = "WRONG"

            print(f"{gmi:>4} {expected:>+15.2f}% {actual:>+15.3f}% {diff:>+15.3f}% {assessment:>15}")
        else:
            print(f"{gmi:>4} {model_expects[gmi]:>+15.2f}% {'N/A':>15} {'N/A':>15} {'NO DATA':>15}")

    print()
    print('=' * 70)
    print('RECOMMENDED EXPECTED MOVES (Based on Actual Data)')
    print('=' * 70)
    print()

    for gmi in range(7):
        subset = df[df['gmi_score'] == gmi]
        if len(subset) > 5:
            avg = subset['next_day_return_pct'].mean()
            print(f"GMI {gmi}: {avg:+.2f}% (based on {len(subset)} observations)")
        else:
            print(f"GMI {gmi}: Insufficient data")

    return df


if __name__ == "__main__":
    main()
