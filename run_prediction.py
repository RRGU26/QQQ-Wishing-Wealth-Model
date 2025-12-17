#!/usr/bin/env python3
"""
QQQ Wishing Wealth Model - Quick Runner

Run this script to get today's prediction:
    python run_prediction.py

Options:
    python run_prediction.py --backtest      Run 2-year backtest
    python run_prediction.py --daily         Run with database logging
    python run_prediction.py --setup-email   Configure email notifications
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_prediction():
    """Run a single prediction."""
    from qqq_wishing_wealth_model import QQQWishingWealthModel

    print("=" * 60)
    print("QQQ WISHING WEALTH PREDICTION MODEL")
    print("Based on Dr. Eric Wish's methodology")
    print("=" * 60)
    print()

    model = QQQWishingWealthModel()

    # Load data
    print("Loading market data...")
    load_result = model.load_data()
    for symbol, status in load_result.items():
        status_str = status.get('status', 'unknown')
        rows = status.get('rows', 0)
        print(f"  {symbol}: {status_str} ({rows} rows)")
    print()

    # Train
    print("Training ML model...")
    train_result = model.train()
    if train_result.get('success'):
        print(f"  Samples: {train_result.get('training_samples')}")
        print(f"  Accuracy: {train_result.get('direction_accuracy', 0):.1%}")
    else:
        print(f"  Note: {train_result.get('error', 'Using rule-based only')}")
    print()

    # Predict
    print("Generating prediction...")
    prediction = model.predict()
    print()

    # Results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(f"Date: {prediction['as_of_date']}")
    print(f"Prediction For: {prediction['prediction_for']}")
    print(f"Current QQQ: ${prediction['current_price']}")
    print(f"Target Price: ${prediction.get('target_price', 'N/A')}")
    print()
    print(prediction['interpretation'])
    print()

    # Recommendation
    print("=" * 60)
    print("TRADING RECOMMENDATION")
    print("=" * 60)
    rec = model.get_trading_recommendation()
    print(f"Action: {rec['action']}")
    print(f"Instrument: {rec['instrument']}")
    print(f"Position Size: {rec['position_size_pct']:.0f}%")
    print()

    # Save
    filepath = model.save_prediction()
    print(f"Saved to: {filepath}")

    return prediction


def run_backtest():
    """Run historical backtest."""
    from backtester import run_full_backtest
    return run_full_backtest()


def run_daily():
    """Run with database logging and optional email."""
    from daily_runner import DailyRunner

    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')
    runner = DailyRunner(config_path if os.path.exists(config_path) else None)
    results = runner.run()
    runner.print_summary(results)
    return results


def setup_email():
    """Interactive email setup."""
    import json

    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.json')

    print("=" * 60)
    print("EMAIL CONFIGURATION")
    print("=" * 60)
    print()
    print("For Gmail, you need an App Password:")
    print("1. Go to Google Account > Security > 2-Step Verification")
    print("2. At bottom, click 'App passwords'")
    print("3. Generate a new app password for 'Mail'")
    print()

    # Load existing config
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    email_config = config.get('email', {})

    # Get inputs
    email_config['smtp_server'] = input(f"SMTP Server [{email_config.get('smtp_server', 'smtp.gmail.com')}]: ").strip() or email_config.get('smtp_server', 'smtp.gmail.com')
    email_config['smtp_port'] = int(input(f"SMTP Port [{email_config.get('smtp_port', 587)}]: ").strip() or email_config.get('smtp_port', 587))
    email_config['sender_email'] = input(f"Your Email [{email_config.get('sender_email', '')}]: ").strip() or email_config.get('sender_email', '')
    email_config['sender_password'] = input("App Password (hidden): ").strip() or email_config.get('sender_password', '')

    recipients_str = input(f"Recipients (comma-separated) [{','.join(email_config.get('recipients', []))}]: ").strip()
    if recipients_str:
        email_config['recipients'] = [r.strip() for r in recipients_str.split(',')]
    elif not email_config.get('recipients'):
        email_config['recipients'] = [email_config['sender_email']]

    email_config['enabled'] = True
    config['email'] = email_config

    # Save
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print(f"Config saved to: {config_path}")
    print("Email notifications enabled!")


def main():
    parser = argparse.ArgumentParser(
        description='QQQ Wishing Wealth Prediction Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_prediction.py              # Quick prediction
  python run_prediction.py --backtest   # Run 2-year backtest
  python run_prediction.py --daily      # Daily run with logging
  python run_prediction.py --setup-email # Configure email
        """
    )
    parser.add_argument('--backtest', action='store_true', help='Run historical backtest')
    parser.add_argument('--daily', action='store_true', help='Run daily with database logging')
    parser.add_argument('--setup-email', action='store_true', help='Configure email notifications')

    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    elif args.daily:
        run_daily()
    elif args.setup_email:
        setup_email()
    else:
        run_prediction()


if __name__ == "__main__":
    main()
