"""
Comprehensive Backtesting Module for QQQ Wishing Wealth Model

Provides:
- Historical simulation of GMI signals
- Performance metrics (Sharpe, drawdown, win rate)
- Comparison vs buy-and-hold
- Trade-by-trade analysis
- Monthly/yearly breakdowns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os

from gmi_calculator import GMICalculator
from supplementary_indicators import SupplementaryIndicators


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    direction: str  # LONG or SHORT
    gmi_score_entry: int
    gmi_score_exit: int
    return_pct: float
    holding_days: int


@dataclass
class BacktestResults:
    """Complete backtest results."""
    start_date: str
    end_date: str
    total_return_pct: float
    buy_hold_return_pct: float
    excess_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    avg_trade_return_pct: float
    avg_holding_days: float
    profit_factor: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame


class Backtester:
    """
    Backtest the Wishing Wealth strategy on historical data.

    Strategy Rules:
    - Enter LONG when GMI >= 5 (GREEN)
    - Exit when GMI <= 2 (RED) or close below 10-week MA
    - Optional: SHORT when GMI <= 1
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.gmi_calculator = GMICalculator()
        self.supplementary = SupplementaryIndicators()

        # Strategy parameters
        self.entry_threshold = 5      # GMI >= 5 to enter long
        self.exit_threshold = 2       # GMI <= 2 to exit
        self.allow_short = False      # Whether to short on RED signals
        self.use_stop_loss = True     # Exit if below 10-week MA

    def run_backtest(
        self,
        qqq_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        xlk_data: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> BacktestResults:
        """
        Run full backtest simulation.

        Args:
            qqq_data: QQQ OHLCV data
            spy_data: SPY OHLCV data
            xlk_data: XLK OHLCV data
            start_date: Backtest start (default: 1 year ago)
            end_date: Backtest end (default: yesterday)

        Returns:
            BacktestResults with all metrics
        """
        # Ensure datetime index
        qqq_data.index = pd.to_datetime(qqq_data.index)
        spy_data.index = pd.to_datetime(spy_data.index)
        xlk_data.index = pd.to_datetime(xlk_data.index)

        # Set date range
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Filter to backtest period
        mask = (qqq_data.index >= start_dt) & (qqq_data.index <= end_dt)
        test_dates = qqq_data.index[mask]

        if len(test_dates) < 50:
            raise ValueError(f"Insufficient data for backtest: {len(test_dates)} days")

        # Initialize tracking
        trades = []
        daily_returns = []
        positions = []  # 1 = long, -1 = short, 0 = cash
        gmi_scores = []

        current_position = 0
        entry_price = 0
        entry_date = None
        entry_gmi = 0

        capital = self.initial_capital
        equity_history = []

        # Calculate GMI for each day
        print(f"Running backtest from {start_date} to {end_date}...")
        print(f"Total trading days: {len(test_dates)}")

        for i, date in enumerate(test_dates):
            idx = qqq_data.index.get_loc(date)

            # Need enough history for indicators
            if idx < 150:
                continue

            # Get historical data up to this point
            hist_qqq = qqq_data.iloc[:idx+1]
            hist_spy = spy_data.iloc[:idx+1] if idx < len(spy_data) else spy_data
            hist_xlk = xlk_data.iloc[:idx+1] if idx < len(xlk_data) else xlk_data

            current_price = qqq_data['Close'].iloc[idx]

            # Calculate GMI
            try:
                gmi_result = self.gmi_calculator.calculate_gmi(
                    qqq_data=hist_qqq,
                    spy_data=hist_spy,
                    fund_data=hist_xlk
                )
                gmi_score = gmi_result.get('gmi_score', 3)
            except:
                gmi_score = 3  # Neutral on error

            gmi_scores.append({'date': date, 'gmi_score': gmi_score})

            # Calculate 10-week (50-day) MA for stop loss
            ma_50 = hist_qqq['Close'].rolling(50).mean().iloc[-1]
            below_ma = current_price < ma_50

            # Trading logic
            prev_position = current_position

            # Entry logic
            if current_position == 0:
                if gmi_score >= self.entry_threshold:
                    # Enter long
                    current_position = 1
                    entry_price = current_price
                    entry_date = date
                    entry_gmi = gmi_score
                elif self.allow_short and gmi_score <= 1:
                    # Enter short
                    current_position = -1
                    entry_price = current_price
                    entry_date = date
                    entry_gmi = gmi_score

            # Exit logic
            elif current_position == 1:  # Long position
                exit_signal = False

                if gmi_score <= self.exit_threshold:
                    exit_signal = True
                elif self.use_stop_loss and below_ma:
                    exit_signal = True

                if exit_signal:
                    # Close long position
                    return_pct = (current_price / entry_price - 1) * 100
                    holding_days = (date - entry_date).days

                    trades.append(Trade(
                        entry_date=entry_date.strftime('%Y-%m-%d'),
                        exit_date=date.strftime('%Y-%m-%d'),
                        entry_price=round(entry_price, 2),
                        exit_price=round(current_price, 2),
                        direction='LONG',
                        gmi_score_entry=entry_gmi,
                        gmi_score_exit=gmi_score,
                        return_pct=round(return_pct, 2),
                        holding_days=holding_days
                    ))

                    capital *= (1 + return_pct / 100)
                    current_position = 0

            elif current_position == -1:  # Short position
                if gmi_score >= 4:  # Exit short when market improves
                    return_pct = (entry_price / current_price - 1) * 100
                    holding_days = (date - entry_date).days

                    trades.append(Trade(
                        entry_date=entry_date.strftime('%Y-%m-%d'),
                        exit_date=date.strftime('%Y-%m-%d'),
                        entry_price=round(entry_price, 2),
                        exit_price=round(current_price, 2),
                        direction='SHORT',
                        gmi_score_entry=entry_gmi,
                        gmi_score_exit=gmi_score,
                        return_pct=round(return_pct, 2),
                        holding_days=holding_days
                    ))

                    capital *= (1 + return_pct / 100)
                    current_position = 0

            positions.append(current_position)

            # Track daily return
            if idx > 0:
                daily_return = (current_price / qqq_data['Close'].iloc[idx-1] - 1)
                strategy_return = daily_return * prev_position  # 0 if in cash
                daily_returns.append({
                    'date': date,
                    'qqq_return': daily_return,
                    'strategy_return': strategy_return,
                    'position': prev_position,
                    'gmi_score': gmi_score
                })

            equity_history.append({
                'date': date,
                'equity': capital,
                'position': current_position,
                'gmi_score': gmi_score
            })

        # Close any open position at end
        if current_position != 0:
            final_price = qqq_data['Close'].iloc[-1]
            if current_position == 1:
                return_pct = (final_price / entry_price - 1) * 100
            else:
                return_pct = (entry_price / final_price - 1) * 100

            trades.append(Trade(
                entry_date=entry_date.strftime('%Y-%m-%d'),
                exit_date=test_dates[-1].strftime('%Y-%m-%d'),
                entry_price=round(entry_price, 2),
                exit_price=round(final_price, 2),
                direction='LONG' if current_position == 1 else 'SHORT',
                gmi_score_entry=entry_gmi,
                gmi_score_exit=gmi_scores[-1]['gmi_score'],
                return_pct=round(return_pct, 2),
                holding_days=(test_dates[-1] - entry_date).days
            ))
            capital *= (1 + return_pct / 100)

        # Calculate metrics
        equity_df = pd.DataFrame(equity_history)
        equity_df.set_index('date', inplace=True)

        returns_df = pd.DataFrame(daily_returns)
        if len(returns_df) > 0:
            returns_df.set_index('date', inplace=True)

        # Total returns
        total_return = (capital / self.initial_capital - 1) * 100

        # Buy and hold return
        start_price = qqq_data.loc[test_dates[0], 'Close']
        end_price = qqq_data.loc[test_dates[-1], 'Close']
        buy_hold_return = (end_price / start_price - 1) * 100

        # Sharpe ratio (annualized)
        if len(returns_df) > 0 and returns_df['strategy_return'].std() > 0:
            sharpe = (returns_df['strategy_return'].mean() / returns_df['strategy_return'].std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak'] - 1) * 100
        max_drawdown = equity_df['drawdown'].min()

        # Win rate
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.return_pct > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            avg_return = np.mean([t.return_pct for t in trades])
            avg_holding = np.mean([t.holding_days for t in trades])

            # Profit factor
            gross_profit = sum(t.return_pct for t in trades if t.return_pct > 0)
            gross_loss = abs(sum(t.return_pct for t in trades if t.return_pct < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_return = 0
            avg_holding = 0
            profit_factor = 0

        # Monthly returns
        if len(returns_df) > 0:
            monthly = returns_df['strategy_return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
            monthly_df = monthly.to_frame('return_pct')
        else:
            monthly_df = pd.DataFrame()

        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            total_return_pct=round(total_return, 2),
            buy_hold_return_pct=round(buy_hold_return, 2),
            excess_return_pct=round(total_return - buy_hold_return, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            win_rate_pct=round(win_rate, 1),
            total_trades=len(trades),
            avg_trade_return_pct=round(avg_return, 2),
            avg_holding_days=round(avg_holding, 1),
            profit_factor=round(profit_factor, 2),
            trades=trades,
            equity_curve=equity_df,
            monthly_returns=monthly_df
        )

    def generate_report(self, results: BacktestResults, output_dir: str = None) -> str:
        """
        Generate a comprehensive backtest report.

        Args:
            results: BacktestResults from run_backtest
            output_dir: Directory to save report

        Returns:
            Report as string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("QQQ WISHING WEALTH MODEL - BACKTEST REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Period: {results.start_date} to {results.end_date}")
        lines.append("")

        # Performance Summary
        lines.append("-" * 70)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 70)
        lines.append(f"{'Strategy Return:':<30} {results.total_return_pct:>10.2f}%")
        lines.append(f"{'Buy & Hold Return:':<30} {results.buy_hold_return_pct:>10.2f}%")
        lines.append(f"{'Excess Return (Alpha):':<30} {results.excess_return_pct:>10.2f}%")
        lines.append(f"{'Sharpe Ratio:':<30} {results.sharpe_ratio:>10.2f}")
        lines.append(f"{'Max Drawdown:':<30} {results.max_drawdown_pct:>10.2f}%")
        lines.append("")

        # Trade Statistics
        lines.append("-" * 70)
        lines.append("TRADE STATISTICS")
        lines.append("-" * 70)
        lines.append(f"{'Total Trades:':<30} {results.total_trades:>10}")
        lines.append(f"{'Win Rate:':<30} {results.win_rate_pct:>10.1f}%")
        lines.append(f"{'Avg Trade Return:':<30} {results.avg_trade_return_pct:>10.2f}%")
        lines.append(f"{'Avg Holding Period:':<30} {results.avg_holding_days:>10.1f} days")
        lines.append(f"{'Profit Factor:':<30} {results.profit_factor:>10.2f}")
        lines.append("")

        # Monthly Returns
        if len(results.monthly_returns) > 0:
            lines.append("-" * 70)
            lines.append("MONTHLY RETURNS")
            lines.append("-" * 70)
            for date, row in results.monthly_returns.iterrows():
                month_str = date.strftime('%Y-%m')
                ret = row['return_pct']
                bar = '+' * int(max(0, ret)) + '-' * int(max(0, -ret))
                lines.append(f"{month_str}: {ret:>7.2f}% {bar}")
            lines.append("")

        # Recent Trades
        lines.append("-" * 70)
        lines.append("RECENT TRADES (Last 10)")
        lines.append("-" * 70)
        lines.append(f"{'Entry':<12} {'Exit':<12} {'Dir':<6} {'Entry$':<10} {'Exit$':<10} {'Return':<10} {'Days':<6}")
        lines.append("-" * 70)

        for trade in results.trades[-10:]:
            lines.append(
                f"{trade.entry_date:<12} {trade.exit_date:<12} {trade.direction:<6} "
                f"${trade.entry_price:<9.2f} ${trade.exit_price:<9.2f} "
                f"{trade.return_pct:>8.2f}% {trade.holding_days:>5}"
            )

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        report = "\n".join(lines)

        # Save to file if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"backtest_report_{datetime.now().strftime('%Y%m%d')}.txt")
            with open(filepath, 'w') as f:
                f.write(report)

            # Also save JSON with full data
            json_path = os.path.join(output_dir, f"backtest_data_{datetime.now().strftime('%Y%m%d')}.json")
            json_data = {
                'summary': {
                    'start_date': results.start_date,
                    'end_date': results.end_date,
                    'total_return_pct': results.total_return_pct,
                    'buy_hold_return_pct': results.buy_hold_return_pct,
                    'excess_return_pct': results.excess_return_pct,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown_pct': results.max_drawdown_pct,
                    'win_rate_pct': results.win_rate_pct,
                    'total_trades': results.total_trades,
                    'profit_factor': results.profit_factor
                },
                'trades': [
                    {
                        'entry_date': t.entry_date,
                        'exit_date': t.exit_date,
                        'direction': t.direction,
                        'entry_price': t.entry_price,
                        'exit_price': t.exit_price,
                        'return_pct': t.return_pct,
                        'holding_days': t.holding_days,
                        'gmi_entry': t.gmi_score_entry,
                        'gmi_exit': t.gmi_score_exit
                    }
                    for t in results.trades
                ]
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"Report saved to: {filepath}")
            print(f"Data saved to: {json_path}")

        return report


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for more robust backtesting.

    Splits data into multiple train/test periods to simulate
    real-world performance more accurately.
    """

    def __init__(self, train_months: int = 6, test_months: int = 1):
        self.train_months = train_months
        self.test_months = test_months

    def run_analysis(
        self,
        qqq_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        xlk_data: pd.DataFrame
    ) -> Dict:
        """
        Run walk-forward analysis.

        Args:
            qqq_data, spy_data, xlk_data: Market data

        Returns:
            Dictionary with walk-forward results
        """
        results = []
        backtester = Backtester()

        # Get date range
        start = qqq_data.index.min()
        end = qqq_data.index.max()

        current = start + pd.DateOffset(months=self.train_months)

        while current + pd.DateOffset(months=self.test_months) <= end:
            test_start = current
            test_end = current + pd.DateOffset(months=self.test_months)

            try:
                period_results = backtester.run_backtest(
                    qqq_data=qqq_data,
                    spy_data=spy_data,
                    xlk_data=xlk_data,
                    start_date=test_start.strftime('%Y-%m-%d'),
                    end_date=test_end.strftime('%Y-%m-%d')
                )

                results.append({
                    'period': f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
                    'return_pct': period_results.total_return_pct,
                    'buy_hold_pct': period_results.buy_hold_return_pct,
                    'trades': period_results.total_trades,
                    'win_rate': period_results.win_rate_pct
                })
            except Exception as e:
                print(f"Error in period {test_start} to {test_end}: {e}")

            current += pd.DateOffset(months=self.test_months)

        # Aggregate results
        if results:
            avg_return = np.mean([r['return_pct'] for r in results])
            avg_excess = np.mean([r['return_pct'] - r['buy_hold_pct'] for r in results])
            win_periods = sum(1 for r in results if r['return_pct'] > r['buy_hold_pct'])

            return {
                'periods_tested': len(results),
                'avg_period_return': round(avg_return, 2),
                'avg_excess_return': round(avg_excess, 2),
                'periods_beat_benchmark': win_periods,
                'consistency_pct': round(win_periods / len(results) * 100, 1),
                'detailed_results': results
            }

        return {'error': 'No valid periods tested'}


def run_full_backtest():
    """Run a complete backtest with all features."""
    import yfinance as yf

    print("Loading data...")

    # Load 3 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1095)  # 3 years

    qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False)
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    xlk = yf.download('XLK', start=start_date, end=end_date, progress=False)

    # Fix multi-index columns from new yfinance
    for df in [qqq, spy, xlk]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel('Ticker')

    print(f"QQQ: {len(qqq)} days")
    print(f"SPY: {len(spy)} days")
    print(f"XLK: {len(xlk)} days")
    print()

    # Run backtest
    backtester = Backtester(initial_capital=100000)

    # 2-year backtest
    results = backtester.run_backtest(
        qqq_data=qqq,
        spy_data=spy,
        xlk_data=xlk,
        start_date=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    )

    # Generate report
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'backtests')
    report = backtester.generate_report(results, output_dir)
    print(report)

    # Walk-forward analysis
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 70)

    wfa = WalkForwardAnalyzer(train_months=6, test_months=1)
    wf_results = wfa.run_analysis(qqq, spy, xlk)

    print(f"Periods Tested: {wf_results.get('periods_tested', 0)}")
    print(f"Avg Period Return: {wf_results.get('avg_period_return', 0):.2f}%")
    print(f"Avg Excess Return: {wf_results.get('avg_excess_return', 0):.2f}%")
    print(f"Beat Benchmark: {wf_results.get('periods_beat_benchmark', 0)}/{wf_results.get('periods_tested', 0)}")
    print(f"Consistency: {wf_results.get('consistency_pct', 0):.1f}%")

    return results, wf_results


if __name__ == "__main__":
    run_full_backtest()
