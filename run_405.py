#!/usr/bin/env python3
"""
QQQ Wishing Wealth Model - 4:05 PM Options Trading Report

Run this at 4:05 PM ET to get:
1. GMI Score with REAL breadth data
2. Options recommendation (calls/puts)
3. Blog comparison with wishingwealthblog.com
4. Complete trade setup for 4:00-4:15 PM execution

Usage:
    python run_405.py
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_405_report():
    """Generate the complete 4:05 PM trading report."""

    print()
    print("=" * 70)
    print("QQQ WISHING WEALTH - 4:05 PM OPTIONS TRADING REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Step 1: Load model and data
    print("[1/6] Loading market data...")
    from qqq_wishing_wealth_model import QQQWishingWealthModel

    model = QQQWishingWealthModel()
    load_result = model.load_data(use_cache=False)

    for symbol, status in load_result.items():
        status_str = status.get('status', 'unknown')
        print(f"      {symbol}: {status_str}")

    # Step 2: Get real breadth data
    print()
    print("[2/6] Fetching market breadth data...")
    from breadth_data import BreadthDataFetcher

    breadth = BreadthDataFetcher()
    breadth_data = breadth.get_all_breadth_data()

    print(f"      New Highs: {breadth_data['new_highs_lows'].get('new_highs', 'N/A')}")
    print(f"      New Lows: {breadth_data['new_highs_lows'].get('new_lows', 'N/A')}")
    print(f"      Successful NH Index: {breadth_data['successful_new_high'].get('successful_count', 'N/A')}")
    print(f"      T2108 Equivalent: {breadth_data['t2108'].get('pct_above_200ma', 'N/A')}%")

    # Step 3: Train ML model
    print()
    print("[3/6] Training ML model...")
    train_result = model.train()
    if train_result.get('success'):
        print(f"      Accuracy: {train_result.get('direction_accuracy', 0):.1%}")
    else:
        print(f"      Using rule-based predictions")

    # Step 4: Generate prediction WITH real breadth data
    print()
    print("[4/6] Generating prediction with real breadth data...")
    prediction = model.predict(breadth_data=breadth_data)

    # Store breadth data in prediction for report
    prediction['breadth_data'] = breadth_data

    # Step 5: Generate options recommendation
    print()
    print("[5/6] Analyzing options setup...")
    from options_analyzer import OptionsAnalyzer

    options = OptionsAnalyzer(options_capital=5000)
    recommendation = options.analyze(prediction)
    options_report = options.generate_report(prediction, recommendation)

    # Step 6: Compare to Wishing Wealth blog
    print()
    print("[6/6] Comparing to Wishing Wealth blog...")
    try:
        from blog_comparison import BlogComparison
        blog = BlogComparison()
        blog_data = blog.fetch_latest()
        blog_report = blog.generate_comparison_report(prediction)
        print(f"      Blog GMI: {blog_data.get('gmi_score', 'N/A')}/6 ({blog_data.get('gmi_signal', 'N/A')})")
    except Exception as e:
        blog_report = f"\n[Blog comparison unavailable: {e}]\n"
        print(f"      Blog fetch error: {e}")

    # === PRINT COMPLETE REPORT ===
    print()
    print(options_report)
    print(blog_report)

    # === SUMMARY BOX ===
    print()
    print("=" * 70)
    print("QUICK SUMMARY - TRADE DECISION")
    print("=" * 70)
    print()
    print(f"  Current QQQ:     ${prediction.get('current_price', 0):.2f}")
    print(f"  Target Price:    ${prediction.get('target_price', 0):.2f}")
    print(f"  GMI Score:       {prediction.get('gmi', {}).get('score', 'N/A')}/6 ({prediction.get('gmi', {}).get('signal', 'N/A')})")
    print()

    if recommendation.action == 'NO_TRADE':
        print("  >>> ACTION: NO TRADE <<<")
        print(f"  Reason: {recommendation.reasoning}")
    else:
        action_str = "BUY CALLS" if 'CALLS' in recommendation.action else "BUY PUTS"
        print(f"  >>> ACTION: {action_str} <<<")
        print(f"  Strike:      ${recommendation.strike_recommended:.0f} ({recommendation.strike_type})")
        print(f"  Expiration:  {recommendation.expiration}")
        print(f"  Confidence:  {recommendation.confidence:.0%}")
        print(f"  Position:    {recommendation.position_size}")
        print(f"  Max Risk:    ${recommendation.max_risk_dollars:.2f}")

    print()
    print("=" * 70)
    print("EXECUTE BETWEEN 4:00 - 4:15 PM ET")
    print("=" * 70)
    print()

    # Save to database
    print()
    print("[DB] Saving to database...")
    try:
        from daily_runner import PredictionDatabase
        db = PredictionDatabase()
        db.save_prediction(prediction)

        # Get performance stats
        perf = db.get_performance_stats(days=30)
        print(f"      Saved to database successfully")
        print(f"      30-day stats: {perf.get('total_predictions', 0)} predictions, {perf.get('accuracy_pct', 0):.1f}% accuracy")
    except Exception as e:
        print(f"      Database error: {e}")

    # Save JSON report
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    report_data = {
        'timestamp': datetime.now().isoformat(),
        'prediction': {
            'as_of_date': prediction.get('as_of_date'),
            'current_price': prediction.get('current_price'),
            'target_price': prediction.get('target_price'),
            'gmi_score': prediction.get('gmi', {}).get('score'),
            'gmi_signal': prediction.get('gmi', {}).get('signal'),
            'direction': prediction.get('final_prediction', {}).get('direction'),
            'confidence': prediction.get('final_prediction', {}).get('confidence')
        },
        'breadth_data': {
            'new_highs': breadth_data['new_highs_lows'].get('new_highs'),
            'new_lows': breadth_data['new_highs_lows'].get('new_lows'),
            'successful_nh': breadth_data['successful_new_high'].get('successful_count'),
            't2108': breadth_data['t2108'].get('pct_above_200ma')
        },
        'options_recommendation': {
            'action': recommendation.action,
            'strike': recommendation.strike_recommended,
            'strike_type': recommendation.strike_type,
            'expiration': recommendation.expiration,
            'confidence': recommendation.confidence,
            'position_size': recommendation.position_size,
            'max_risk': recommendation.max_risk_dollars,
            'reasoning': recommendation.reasoning
        }
    }

    report_file = os.path.join(output_dir, f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"Report saved: {report_file}")

    return report_data


if __name__ == "__main__":
    run_405_report()
