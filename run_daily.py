"""
Wishing Wealth Daily Market Report

Generates daily market analysis using the GMI methodology:
1. GMI Score (0-6) - Market health indicator
2. Market conditions (MAs, Stochastic, VIX)
3. Stock setups when GMI is GREEN

Runs at 4:10 PM ET after market close.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qqq_wishing_wealth_model import QQQWishingWealthModel
from gmi_calculator import get_gmi_interpretation
from breadth_data import BreadthDataFetcher
from blog_comparison import BlogComparison
from stock_screener import StockScreener


def run_daily_report():
    """Generate the daily market report."""

    print("=" * 65)
    print("WISHING WEALTH DAILY MARKET REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # Step 1: Load market data
    print("[1/5] Loading market data...")
    model = QQQWishingWealthModel()
    load_result = model.load_data(use_cache=False)

    for symbol, status in load_result.items():
        print(f"      {symbol}: {status.get('status')}")
    print()

    # Step 2: Fetch breadth data
    print("[2/5] Fetching market breadth data...")
    breadth_fetcher = BreadthDataFetcher()
    breadth_data = breadth_fetcher.get_all_breadth_data()

    if 'error' not in breadth_data:
        nh = breadth_data.get('new_highs_lows', {})
        print(f"      New Highs: {nh.get('new_highs', 'N/A')}")
        print(f"      New Lows: {nh.get('new_lows', 'N/A')}")
    else:
        print(f"      Warning: {breadth_data.get('error')}")
    print()

    # Step 3: Calculate GMI
    print("[3/5] Calculating GMI...")
    prediction = model.predict(breadth_data=breadth_data if 'error' not in breadth_data else None)

    gmi = prediction.get('gmi', {})
    gmi_score = gmi.get('score', 3)
    gmi_signal = gmi.get('signal', 'YELLOW')

    print(f"      GMI: {gmi_score}/6 ({gmi_signal})")
    print()

    # Step 4: Compare to blog
    print("[4/5] Comparing to Wishing Wealth blog...")
    blog = BlogComparison()
    blog_data = blog.fetch_latest()

    if 'error' not in blog_data:
        blog_gmi = blog_data.get('gmi_score')
        blog_signal = blog_data.get('gmi_signal')
        print(f"      Blog GMI: {blog_gmi}/6 ({blog_signal})")
    else:
        print(f"      Could not fetch blog data")
    print()

    # Step 5: Scan for stocks (only if GMI is green-ish)
    print("[5/5] Scanning for stock setups...")
    if gmi_score >= 4:
        screener = StockScreener()
        setups = screener.scan_universe()
        total_setups = sum(len(s) for s in setups.values())
        print(f"      Found {total_setups} setups")
    else:
        setups = None
        print(f"      Skipped - GMI not favorable for new entries")
    print()

    # Generate Report
    print("=" * 65)
    print("MARKET STATUS")
    print("=" * 65)
    print()

    current_price = prediction.get('current_price', 0)
    as_of_date = prediction.get('as_of_date', 'N/A')

    print(f"QQQ CLOSE: ${current_price:.2f}")
    print(f"AS OF: {as_of_date}")
    print()

    # GMI Section
    print("-" * 65)
    print("GMI (GENERAL MARKET INDEX)")
    print("-" * 65)
    print()
    print(f"  Score: {gmi_score}/6")
    print(f"  Signal: {gmi_signal}")
    print(f"  Interpretation: {get_gmi_interpretation(gmi_score)}")
    print()

    # Show components
    components = gmi.get('components', {})
    print("  Components:")
    component_names = {
        'successful_new_high': 'Successful New High Index',
        'daily_new_highs': 'Daily New Highs (>100)',
        'daily_qqq': 'QQQ Above 50-day MA',
        'daily_spy': 'SPY Above 50-day MA',
        'weekly_qqq': '10-week MA > 30-week MA',
        'ibd_fund_index': 'Growth Fund Above 50-day MA'
    }

    for key, name in component_names.items():
        comp = components.get(key, {})
        status = "+" if comp.get('positive', False) else "-"
        print(f"    [{status}] {name}")
    print()

    # Supplementary Indicators
    print("-" * 65)
    print("SUPPLEMENTARY INDICATORS")
    print("-" * 65)
    print()

    supp = prediction.get('supplementary', {})
    stoch = supp.get('stochastic', {})
    ma = supp.get('ma_framework', {})
    vix = supp.get('vix_context', {})

    print(f"  Stochastic (10.4.4): {stoch.get('k', 'N/A'):.0f} - {stoch.get('signal', 'N/A')}")
    print(f"  MA Framework: {ma.get('ma_signal', 'N/A')}")

    if vix:
        print(f"  VIX: {vix.get('current_vix', 'N/A'):.1f} ({vix.get('regime', 'N/A')})")
    print()

    # Blog Comparison
    print("-" * 65)
    print("BLOG COMPARISON")
    print("-" * 65)
    print()

    if 'error' not in blog_data:
        print(f"  {'':15} {'Our Model':>15} {'Blog':>15}")
        print(f"  {'-'*45}")
        print(f"  {'GMI Score':15} {gmi_score:>15} {blog_data.get('gmi_score', 'N/A'):>15}")
        print(f"  {'Signal':15} {gmi_signal:>15} {blog_data.get('gmi_signal', 'N/A'):>15}")

        if blog_data.get('qqq_trend_day'):
            print(f"  {'Trend Day':15} {'-':>15} {'Day ' + str(blog_data.get('qqq_trend_day')):>15}")

        if blog_data.get('post_date'):
            print()
            print(f"  Blog Post Date: {blog_data.get('post_date')}")

        # Alignment check
        if blog_data.get('gmi_score') is not None:
            diff = abs(gmi_score - blog_data.get('gmi_score', gmi_score))
            if diff == 0:
                print("  Status: ALIGNED")
            elif diff == 1:
                print("  Status: Minor difference (1 point)")
            else:
                print(f"  Status: DIVERGENCE ({diff} points) - verify data")
    else:
        print("  Could not fetch blog data for comparison")
    print()

    # Action Guidance
    print("-" * 65)
    print("ACTION GUIDANCE")
    print("-" * 65)
    print()

    if gmi_score >= 5:
        print("  GMI GREEN - Favorable conditions for:")
        print("    - Full position sizing")
        print("    - New breakout entries (GLB, RWB patterns)")
        print("    - Holding existing winners")
    elif gmi_score >= 3:
        print("  GMI YELLOW - Neutral conditions:")
        print("    - Reduce position sizes to 50%")
        print("    - Be selective with new entries")
        print("    - Tighten stops on existing positions")
    else:
        print("  GMI RED - Unfavorable conditions:")
        print("    - Avoid new long entries")
        print("    - Consider raising cash")
        print("    - Wait for GMI to improve")
    print()

    # Stock Setups (if scanned)
    if setups is not None:
        print("=" * 65)
        print("STOCK SETUPS - ACTIONABLE TRADE IDEAS")
        print("=" * 65)
        print()

        # GLB Section
        print("-" * 65)
        print("GLB (GREEN LINE BREAKOUT)")
        print("-" * 65)
        print()
        print("  WHAT IT MEANS:")
        print("    Stock breaking to NEW ALL-TIME HIGH after 3+ months of")
        print("    consolidation. This is the classic breakout pattern.")
        print()
        print("  ACTION:")
        print("    - BUY on breakout with volume confirmation")
        print("    - Set stop-loss just below the breakout level (prior ATH)")
        print("    - Target: Let winners run, trail stop with 10-week MA")
        print()

        if setups.get('GLB'):
            print(f"  {'Symbol':<8} {'Price':>10} {'Score':>6} {'Above ATH':>12} {'Base Days':>12}")
            print(f"  {'-'*56}")
            for s in setups['GLB'][:5]:
                print(
                    f"  {s.symbol:<8} ${s.price:>9.2f} {s.score:>6}/10 "
                    f"{s.details['breakout_pct']:>+11.1f}% {s.details['consolidation_days']:>12}"
                )
            print()
            print(f"  >> TOP PICK: {setups['GLB'][0].symbol} - Breaking out {setups['GLB'][0].details['breakout_pct']:+.1f}% above prior high")
        else:
            print("  No GLB setups found today.")
        print()

        # Blue Dot Section
        print("-" * 65)
        print("BLUE DOT (OVERSOLD BOUNCE)")
        print("-" * 65)
        print()
        print("  WHAT IT MEANS:")
        print("    Stock pulled back to support (50 or 200-day MA) with")
        print("    Stochastic oversold (<25). Bounce expected in uptrend.")
        print()
        print("  ACTION:")
        print("    - BUY when Stochastic turns up from oversold")
        print("    - Set stop-loss below the supporting MA")
        print("    - Target: Prior swing high or +10-15%")
        print()

        if setups.get('BLUE_DOT'):
            print(f"  {'Symbol':<8} {'Price':>10} {'Score':>6} {'Stoch':>8} {'vs MA50':>10} {'Off High':>10}")
            print(f"  {'-'*60}")
            for s in setups['BLUE_DOT'][:5]:
                print(
                    f"  {s.symbol:<8} ${s.price:>9.2f} {s.score:>6}/10 "
                    f"{s.details['stochastic_k']:>7.0f} {s.details['pct_vs_ma50']:>+9.1f}% "
                    f"{s.details['pct_off_high']:>9.1f}%"
                )
            print()
            top_bd = setups['BLUE_DOT'][0]
            print(f"  >> TOP PICK: {top_bd.symbol} - Stoch at {top_bd.details['stochastic_k']:.0f}, {top_bd.details['pct_off_high']:.0f}% off high")
        else:
            print("  No Blue Dot setups found today.")
        print()

        # RWB Section
        print("-" * 65)
        print("RWB (RED WHITE BLUE - TREND ALIGNED)")
        print("-" * 65)
        print()
        print("  WHAT IT MEANS:")
        print("    Moving averages perfectly aligned (10 > 30 > 50 > 100 > 200).")
        print("    Stock is in a strong, steady uptrend.")
        print()
        print("  ACTION:")
        print("    - BUY on pullback to 10 or 21-day MA")
        print("    - Set stop-loss below 50-day MA")
        print("    - Target: Hold as long as MA alignment persists")
        print()

        if setups.get('RWB'):
            print(f"  {'Symbol':<8} {'Price':>10} {'Score':>6} {'Trend Str':>10} {'vs MA10':>10}")
            print(f"  {'-'*52}")
            for s in setups['RWB'][:5]:
                print(
                    f"  {s.symbol:<8} ${s.price:>9.2f} {s.score:>6}/10 "
                    f"{s.details['ma_spread_pct']:>+9.1f}% {s.details['pct_above_ma10']:>+9.1f}%"
                )
            print()
            top_rwb = setups['RWB'][0]
            print(f"  >> TOP PICK: {top_rwb.symbol} - Strong trend, {top_rwb.details['pct_above_ma10']:+.1f}% above 10-day MA")
        else:
            print("  No RWB setups found today.")
        print()

        # Overall Recommendations
        total_setups = sum(len(s) for s in setups.values())
        print("=" * 65)
        print("TODAY'S RECOMMENDATIONS")
        print("=" * 65)
        print()

        if gmi_score >= 5:
            print("  MARKET: GREEN LIGHT - Conditions favor new long entries")
            print()
            print("  PRIORITY ORDER:")
            print("    1. GLB breakouts (highest conviction when GMI is green)")
            print("    2. RWB pullbacks to 10/21-day MA")
            print("    3. Blue Dot bounces for quick trades")
            print()

            if setups.get('GLB'):
                print(f"  BEST OPPORTUNITY: {setups['GLB'][0].symbol} (GLB breakout)")
            elif setups.get('RWB'):
                # Find RWB with smallest pct_above_ma10 (best entry)
                best_rwb = min(setups['RWB'][:5], key=lambda x: x.details['pct_above_ma10'])
                print(f"  BEST OPPORTUNITY: {best_rwb.symbol} (RWB, closest to MA support)")
            elif setups.get('BLUE_DOT'):
                print(f"  BEST OPPORTUNITY: {setups['BLUE_DOT'][0].symbol} (Blue Dot bounce)")
        else:
            print("  MARKET: YELLOW/RED - Be cautious with new entries")
            print()
            print("  RECOMMENDATION:")
            print("    - Wait for GMI to turn GREEN before adding positions")
            print("    - If trading, use half position sizes")
            print("    - Focus on Blue Dot bounces (lower risk)")
        print()

    # Summary Box
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print()
    print(f"  GMI: {gmi_score}/6 ({gmi_signal})")

    if gmi_score >= 5:
        print("  Stance: AGGRESSIVE LONG")
    elif gmi_score >= 3:
        print("  Stance: CAUTIOUS / NIMBLE")
    else:
        print("  Stance: DEFENSIVE / CASH")

    if setups:
        total = sum(len(s) for s in setups.values())
        print(f"  Stock Setups: {total} found")

    print()
    print("=" * 65)

    # Save to database
    print()
    print("[DB] Saving to database...")
    try:
        from daily_runner import PredictionDatabase
        db = PredictionDatabase()
        db.save_prediction(prediction)
        print("      Saved successfully")
    except Exception as e:
        print(f"      Warning: {e}")

    # Save report to file
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = os.path.join(output_dir, f'daily_report_{timestamp}.txt')

    print(f"Report saved: {output_file}")

    return prediction


if __name__ == "__main__":
    run_daily_report()
    print()
    # Keep window open if run from scheduler
    try:
        input("Press Enter to close...")
    except EOFError:
        pass  # Running in non-interactive mode
