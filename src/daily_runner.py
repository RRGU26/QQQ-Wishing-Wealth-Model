"""
Daily Scheduled Runner for QQQ Wishing Wealth Model

Provides:
- Automated daily predictions at market close
- Email notifications with predictions
- Performance tracking over time
- Historical prediction accuracy logging
- Windows Task Scheduler integration
"""

import os
import sys
import json
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from qqq_wishing_wealth_model import QQQWishingWealthModel


class PredictionDatabase:
    """SQLite database for tracking predictions and performance."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'predictions.db'
            )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    current_price REAL,
                    predicted_direction TEXT,
                    predicted_move_pct REAL,
                    confidence REAL,
                    gmi_score INTEGER,
                    gmi_signal TEXT,
                    actual_price REAL,
                    actual_direction TEXT,
                    actual_move_pct REAL,
                    correct INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE,
                    predictions_total INTEGER,
                    predictions_correct INTEGER,
                    accuracy_pct REAL,
                    avg_confidence REAL,
                    cumulative_return_pct REAL
                )
            """)
            conn.commit()

    def save_prediction(self, prediction: Dict):
        """Save a new prediction to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO predictions (
                    prediction_date, target_date, current_price,
                    predicted_direction, predicted_move_pct, confidence,
                    gmi_score, gmi_signal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction['as_of_date'],
                prediction['prediction_for'],
                prediction['current_price'],
                prediction['final_prediction']['direction'],
                prediction['final_prediction']['expected_move_pct'],
                prediction['final_prediction']['confidence'],
                prediction['gmi']['score'],
                prediction['gmi']['signal']
            ))
            conn.commit()

    def update_actual(self, target_date: str, actual_price: float):
        """Update prediction with actual results."""
        with sqlite3.connect(self.db_path) as conn:
            # Get the prediction for this target date
            cursor = conn.execute("""
                SELECT id, current_price, predicted_direction
                FROM predictions
                WHERE target_date = ? AND actual_price IS NULL
            """, (target_date,))

            row = cursor.fetchone()
            if row:
                pred_id, prev_price, pred_direction = row
                actual_move = (actual_price / prev_price - 1) * 100
                actual_direction = 'UP' if actual_move > 0 else 'DOWN' if actual_move < 0 else 'NEUTRAL'

                correct = 1 if (
                    (pred_direction == 'UP' and actual_move > 0) or
                    (pred_direction == 'DOWN' and actual_move < 0) or
                    (pred_direction == 'NEUTRAL' and abs(actual_move) < 0.5)
                ) else 0

                conn.execute("""
                    UPDATE predictions
                    SET actual_price = ?, actual_direction = ?,
                        actual_move_pct = ?, correct = ?
                    WHERE id = ?
                """, (actual_price, actual_direction, round(actual_move, 4), correct, pred_id))
                conn.commit()

                return {'updated': True, 'correct': correct}

        return {'updated': False}

    def get_performance_stats(self, days: int = 30) -> Dict:
        """Get performance statistics for recent predictions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(correct) as correct,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN predicted_direction = actual_direction THEN actual_move_pct ELSE 0 END) as aligned_moves
                FROM predictions
                WHERE actual_price IS NOT NULL
                AND prediction_date >= date('now', ?)
            """, (f'-{days} days',))

            row = cursor.fetchone()
            total, correct, avg_conf, aligned_moves = row

            if total and total > 0:
                return {
                    'total_predictions': total,
                    'correct_predictions': correct or 0,
                    'accuracy_pct': round((correct or 0) / total * 100, 1),
                    'avg_confidence': round(avg_conf or 0, 3),
                    'period_days': days
                }

        return {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy_pct': 0,
            'avg_confidence': 0,
            'period_days': days
        }

    def get_recent_predictions(self, limit: int = 10) -> List[Dict]:
        """Get recent predictions with their outcomes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    prediction_date, target_date, current_price,
                    predicted_direction, predicted_move_pct, confidence,
                    gmi_score, gmi_signal,
                    actual_price, actual_direction, actual_move_pct, correct
                FROM predictions
                ORDER BY prediction_date DESC
                LIMIT ?
            """, (limit,))

            columns = [
                'prediction_date', 'target_date', 'current_price',
                'predicted_direction', 'predicted_move_pct', 'confidence',
                'gmi_score', 'gmi_signal',
                'actual_price', 'actual_direction', 'actual_move_pct', 'correct'
            ]

            return [dict(zip(columns, row)) for row in cursor.fetchall()]


class EmailNotifier:
    """Send email notifications with predictions."""

    def __init__(self, config: Dict = None):
        """
        Initialize email notifier.

        Config should contain:
        - smtp_server: SMTP server address
        - smtp_port: SMTP port (587 for TLS)
        - sender_email: Your email address
        - sender_password: App password (not regular password)
        - recipients: List of recipient emails
        """
        self.config = config or {}
        self.enabled = all(k in self.config for k in ['smtp_server', 'sender_email', 'sender_password'])

    def send_prediction_email(
        self,
        prediction: Dict,
        recommendation: Dict,
        performance: Dict
    ) -> bool:
        """Send prediction email to all recipients."""
        if not self.enabled:
            print("Email not configured. Skipping notification.")
            return False

        subject = self._generate_subject(prediction)
        body = self._generate_body(prediction, recommendation, performance)

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config['sender_email']
            msg['To'] = ', '.join(self.config.get('recipients', [self.config['sender_email']]))

            # Plain text version
            text_part = MIMEText(body, 'plain')
            msg.attach(text_part)

            # HTML version
            html_body = self._generate_html_body(prediction, recommendation, performance)
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)

            with smtplib.SMTP(self.config['smtp_server'], self.config.get('smtp_port', 587)) as server:
                server.starttls()
                server.login(self.config['sender_email'], self.config['sender_password'])
                server.send_message(msg)

            print(f"Email sent successfully to {msg['To']}")
            return True

        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    def _generate_subject(self, prediction: Dict) -> str:
        """Generate email subject line."""
        gmi = prediction['gmi']
        final = prediction['final_prediction']

        signal_emoji = {
            'GREEN': '🟢',
            'YELLOW': '🟡',
            'RED': '🔴'
        }.get(gmi['signal'], '⚪')

        direction_emoji = '📈' if final['direction'] == 'UP' else '📉' if final['direction'] == 'DOWN' else '➡️'

        return f"{signal_emoji} QQQ Wishing Wealth: GMI {gmi['score']}/6 {gmi['signal']} | {direction_emoji} {final['direction']} ({final['confidence']:.0%})"

    def _generate_body(self, prediction: Dict, recommendation: Dict, performance: Dict) -> str:
        """Generate plain text email body."""
        lines = []
        lines.append("=" * 60)
        lines.append("QQQ WISHING WEALTH DAILY PREDICTION")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Date: {prediction['as_of_date']}")
        lines.append(f"Prediction For: {prediction['prediction_for']}")
        lines.append(f"Current QQQ Price: ${prediction['current_price']}")
        lines.append("")

        # GMI Status
        lines.append("-" * 60)
        lines.append("GMI STATUS")
        lines.append("-" * 60)
        lines.append(f"Score: {prediction['gmi']['score']}/6")
        lines.append(f"Signal: {prediction['gmi']['signal']}")
        lines.append(f"Action: {prediction['gmi']['action']}")
        lines.append("")

        # Prediction
        lines.append("-" * 60)
        lines.append("PREDICTION")
        lines.append("-" * 60)
        final = prediction['final_prediction']
        lines.append(f"Direction: {final['direction']}")
        lines.append(f"Expected Move: {final['expected_move_pct']:+.2f}%")
        lines.append(f"Confidence: {final['confidence']:.0%}")
        lines.append("")

        # Recommendation
        lines.append("-" * 60)
        lines.append("TRADING RECOMMENDATION")
        lines.append("-" * 60)
        lines.append(f"Action: {recommendation['action']}")
        lines.append(f"Instrument: {recommendation['instrument']}")
        lines.append(f"Position Size: {recommendation['position_size_pct']:.0f}%")
        lines.append("")

        # Performance
        if performance.get('total_predictions', 0) > 0:
            lines.append("-" * 60)
            lines.append(f"RECENT PERFORMANCE ({performance['period_days']} days)")
            lines.append("-" * 60)
            lines.append(f"Predictions: {performance['total_predictions']}")
            lines.append(f"Correct: {performance['correct_predictions']}")
            lines.append(f"Accuracy: {performance['accuracy_pct']:.1f}%")
            lines.append("")

        # Interpretation
        lines.append("-" * 60)
        lines.append("ANALYSIS")
        lines.append("-" * 60)
        lines.append(prediction.get('interpretation', 'No interpretation available'))
        lines.append("")

        lines.append("=" * 60)
        lines.append("Generated by QQQ Wishing Wealth Model")
        lines.append("Based on Dr. Eric Wish's methodology")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_html_body(self, prediction: Dict, recommendation: Dict, performance: Dict) -> str:
        """Generate HTML email body."""
        gmi = prediction['gmi']
        final = prediction['final_prediction']

        signal_color = {
            'GREEN': '#28a745',
            'YELLOW': '#ffc107',
            'RED': '#dc3545'
        }.get(gmi['signal'], '#6c757d')

        direction_color = '#28a745' if final['direction'] == 'UP' else '#dc3545' if final['direction'] == 'DOWN' else '#6c757d'

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #333; border-bottom: 2px solid {signal_color}; padding-bottom: 10px;">
                QQQ Wishing Wealth Prediction
            </h1>

            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <p style="margin: 5px 0;"><strong>Date:</strong> {prediction['as_of_date']}</p>
                <p style="margin: 5px 0;"><strong>Prediction For:</strong> {prediction['prediction_for']}</p>
                <p style="margin: 5px 0;"><strong>Current Price:</strong> ${prediction['current_price']}</p>
            </div>

            <h2 style="color: {signal_color};">GMI: {gmi['score']}/6 - {gmi['signal']}</h2>
            <p>{gmi['interpretation']}</p>

            <div style="background: {direction_color}; color: white; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                <h2 style="margin: 0;">PREDICTION: {final['direction']}</h2>
                <p style="margin: 10px 0 0 0; font-size: 18px;">
                    Expected Move: {final['expected_move_pct']:+.2f}% | Confidence: {final['confidence']:.0%}
                </p>
            </div>

            <h3>Trading Recommendation</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Action</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{recommendation['action']}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Instrument</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{recommendation['instrument']}</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>Position Size</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{recommendation['position_size_pct']:.0f}%</td>
                </tr>
            </table>

            <h3>Analysis</h3>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; white-space: pre-wrap;">
{prediction.get('interpretation', 'No interpretation available')}
            </pre>

            <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
            <p style="color: #666; font-size: 12px;">
                Generated by QQQ Wishing Wealth Model<br>
                Based on Dr. Eric Wish's methodology from wishingwealthblog.com
            </p>
        </body>
        </html>
        """
        return html


class DailyRunner:
    """Main daily runner that coordinates all components."""

    def __init__(self, config_path: str = None):
        """
        Initialize the daily runner.

        Args:
            config_path: Path to config JSON file with email settings
        """
        self.model = QQQWishingWealthModel()
        self.db = PredictionDatabase()

        # Load config
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                self.config = json.load(f)

        self.email = EmailNotifier(self.config.get('email', {}))

        # Setup logging
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'daily_runner.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run(self) -> Dict:
        """
        Execute the daily prediction workflow.

        Returns:
            Dictionary with run results
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting QQQ Wishing Wealth Daily Run")
        self.logger.info("=" * 60)

        results = {'success': False, 'steps': []}

        try:
            # Step 1: Load fresh data
            self.logger.info("Step 1: Loading market data...")
            load_result = self.model.load_data(use_cache=False)
            results['steps'].append({'step': 'load_data', 'result': load_result})
            self.logger.info(f"Data loaded: {load_result}")

            # Step 2: Update previous predictions with actuals
            self.logger.info("Step 2: Updating previous predictions...")
            if self.model.qqq_data is not None:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                try:
                    actual_price = self.model.qqq_data['Close'].iloc[-1]
                    update_result = self.db.update_actual(yesterday, actual_price)
                    results['steps'].append({'step': 'update_actuals', 'result': update_result})
                    self.logger.info(f"Updated actuals: {update_result}")
                except Exception as e:
                    self.logger.warning(f"Could not update actuals: {e}")

            # Step 3: Train ML model
            self.logger.info("Step 3: Training ML model...")
            train_result = self.model.train()
            results['steps'].append({'step': 'train', 'result': train_result})
            self.logger.info(f"Training result: {train_result.get('success', False)}")

            # Step 4: Generate prediction
            self.logger.info("Step 4: Generating prediction...")
            prediction = self.model.predict()
            results['prediction'] = prediction
            self.logger.info(f"Prediction: {prediction['final_prediction']}")

            # Step 5: Get trading recommendation
            recommendation = self.model.get_trading_recommendation()
            results['recommendation'] = recommendation

            # Step 6: Save prediction to database
            self.logger.info("Step 5: Saving prediction to database...")
            self.db.save_prediction(prediction)
            results['steps'].append({'step': 'save_prediction', 'result': 'success'})

            # Step 7: Save prediction to JSON file
            filepath = self.model.save_prediction()
            results['prediction_file'] = filepath
            self.logger.info(f"Prediction saved to: {filepath}")

            # Step 8: Get performance stats
            performance = self.db.get_performance_stats(days=30)
            results['performance'] = performance
            self.logger.info(f"30-day performance: {performance}")

            # Step 9: Send email notification
            self.logger.info("Step 6: Sending email notification...")
            email_sent = self.email.send_prediction_email(prediction, recommendation, performance)
            results['email_sent'] = email_sent

            results['success'] = True
            self.logger.info("Daily run completed successfully!")

        except Exception as e:
            self.logger.error(f"Daily run failed: {e}", exc_info=True)
            results['error'] = str(e)

        return results

    def print_summary(self, results: Dict):
        """Print a summary of the run results."""
        print("\n" + "=" * 60)
        print("DAILY RUN SUMMARY")
        print("=" * 60)

        if results.get('success'):
            pred = results.get('prediction', {})
            rec = results.get('recommendation', {})
            perf = results.get('performance', {})

            print(f"\nPrediction Date: {pred.get('as_of_date', 'N/A')}")
            print(f"Target Date: {pred.get('prediction_for', 'N/A')}")
            print(f"Current Price: ${pred.get('current_price', 'N/A')}")

            print(f"\nGMI: {pred.get('gmi', {}).get('score', 'N/A')}/6 ({pred.get('gmi', {}).get('signal', 'N/A')})")

            final = pred.get('final_prediction', {})
            print(f"\nPrediction: {final.get('direction', 'N/A')}")
            print(f"Expected Move: {final.get('expected_move_pct', 0):+.2f}%")
            print(f"Confidence: {final.get('confidence', 0):.0%}")

            print(f"\nRecommendation: {rec.get('action', 'N/A')}")
            print(f"Position Size: {rec.get('position_size_pct', 0):.0f}%")

            if perf.get('total_predictions', 0) > 0:
                print(f"\n30-Day Accuracy: {perf.get('accuracy_pct', 0):.1f}% ({perf.get('correct_predictions', 0)}/{perf.get('total_predictions', 0)})")

            print(f"\nEmail Sent: {'Yes' if results.get('email_sent') else 'No'}")
            print(f"Prediction Saved: {results.get('prediction_file', 'N/A')}")

        else:
            print(f"\nRun Failed: {results.get('error', 'Unknown error')}")

        print("\n" + "=" * 60)


def create_config_template():
    """Create a template config file."""
    config = {
        "email": {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your-email@gmail.com",
            "sender_password": "your-app-password",
            "recipients": [
                "recipient1@email.com",
                "recipient2@email.com"
            ]
        }
    }

    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config template created at: {config_path}")
    print("Edit this file with your email settings to enable notifications.")
    return config_path


def main():
    """Main entry point for daily runner."""
    import argparse

    parser = argparse.ArgumentParser(description='QQQ Wishing Wealth Daily Runner')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--create-config', action='store_true', help='Create config template')
    args = parser.parse_args()

    if args.create_config:
        create_config_template()
        return

    config_path = args.config
    if config_path is None:
        default_config = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
        if os.path.exists(default_config):
            config_path = default_config

    runner = DailyRunner(config_path)
    results = runner.run()
    runner.print_summary(results)

    return results


if __name__ == "__main__":
    main()
