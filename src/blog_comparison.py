"""
Wishing Wealth Blog Comparison Module

Fetches and parses the latest GMI reading from wishingwealthblog.com
to compare with our model's prediction.
"""

import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional
import json


class BlogComparison:
    """
    Fetch and compare our model to the Wishing Wealth blog.
    """

    BLOG_URL = "https://wishingwealthblog.com/"

    def __init__(self):
        self.last_fetch = None
        self.blog_data = None

    def fetch_latest(self) -> Dict:
        """
        Fetch the latest post from the Wishing Wealth blog.

        Returns:
            Dictionary with blog GMI data and observations
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.BLOG_URL, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content
            text = soup.get_text()

            # Parse GMI information
            blog_data = {
                'fetched_at': datetime.now().isoformat(),
                'post_date': self._extract_post_date(text),
                'gmi_score': self._extract_gmi_score(text),
                'gmi_signal': self._extract_gmi_signal(text),
                'qqq_trend_day': self._extract_trend_day(text),
                'trend_direction': self._extract_trend_direction(text),
                'key_observations': self._extract_observations(text),
                'raw_snippet': text[:2000]  # First 2000 chars for context
            }

            self.blog_data = blog_data
            self.last_fetch = datetime.now()

            return blog_data

        except Exception as e:
            return {
                'error': str(e),
                'fetched_at': datetime.now().isoformat()
            }

    def _extract_gmi_score(self, text: str) -> Optional[int]:
        """Extract GMI score from blog text."""
        # Look for patterns like "GMI=5" or "GMI: 5" or "GMI is 5"
        patterns = [
            r'GMI[=:\s]+(\d)[/\s]?(?:of\s)?6',
            r'GMI.*?(\d)\s*(?:of|/)\s*6',
            r'GMI.*?back to (\d)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_post_date(self, text: str) -> Optional[str]:
        """Extract the blog post date - only match dates within last 7 days."""
        from datetime import timedelta

        text_start = text[:1500]
        today = datetime.now()

        # Month name to number mapping
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }

        # Look for "Month Day, Year" pattern
        pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})'
        matches = re.findall(pattern, text_start, re.IGNORECASE)

        for month_name, day, year in matches:
            try:
                month_num = month_map[month_name.lower()]
                parsed_date = datetime(int(year), month_num, int(day))
                # Only accept dates within the last 7 days
                if (today - parsed_date).days <= 7 and (today - parsed_date).days >= 0:
                    return f"{month_name} {day}, {year}"
            except (ValueError, KeyError):
                continue

        # Fallback: return None rather than a stale date
        return None

    def _extract_gmi_signal(self, text: str) -> Optional[str]:
        """Extract GMI signal color from blog text."""
        if re.search(r'GMI.*(?:is|remains?|on)\s*(?:a\s)?Green', text, re.IGNORECASE):
            return 'GREEN'
        elif re.search(r'GMI.*(?:is|remains?|on)\s*(?:a\s)?Red', text, re.IGNORECASE):
            return 'RED'
        elif re.search(r'Green\s*signal', text, re.IGNORECASE):
            return 'GREEN'
        elif re.search(r'Red\s*signal', text, re.IGNORECASE):
            return 'RED'
        return None

    def _extract_trend_day(self, text: str) -> Optional[int]:
        """Extract QQQ trend day count."""
        # Look for "Day X of QQQ" pattern
        match = re.search(r'Day\s*(\d+)\s*of\s*\$?QQQ', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_trend_direction(self, text: str) -> Optional[str]:
        """Extract trend direction (up-trend or down-trend)."""
        if re.search(r'short.?term\s+up.?trend', text, re.IGNORECASE):
            return 'UP'
        elif re.search(r'short.?term\s+down.?trend', text, re.IGNORECASE):
            return 'DOWN'
        return None

    def _extract_observations(self, text: str) -> list:
        """Extract key observations from blog."""
        observations = []

        # Check for common themes
        if re.search(r'growth.*faltering|growth.*weak', text, re.IGNORECASE):
            observations.append("Growth stocks/funds weakening")

        if re.search(r'mainly.*cash|defensive', text, re.IGNORECASE):
            observations.append("Author in defensive/cash position")

        if re.search(r'oversold|bounce', text, re.IGNORECASE):
            observations.append("Oversold bounce potential noted")

        if re.search(r'BWR|decline', text, re.IGNORECASE):
            observations.append("BWR (bearish) pattern mentioned")

        if re.search(r'RWB|uptrend', text, re.IGNORECASE):
            observations.append("RWB (bullish) pattern mentioned")

        return observations

    def compare_to_model(self, model_prediction: Dict) -> Dict:
        """
        Compare our model prediction to the blog.

        Args:
            model_prediction: Output from QQQWishingWealthModel.predict()

        Returns:
            Comparison dictionary
        """
        if self.blog_data is None:
            self.fetch_latest()

        if self.blog_data is None or 'error' in self.blog_data:
            return {'error': 'Could not fetch blog data'}

        model_gmi = model_prediction.get('gmi', {}).get('score')
        model_signal = model_prediction.get('gmi', {}).get('signal')
        blog_gmi = self.blog_data.get('gmi_score')
        blog_signal = self.blog_data.get('gmi_signal')

        # Calculate alignment
        gmi_match = model_gmi == blog_gmi if (model_gmi and blog_gmi) else None
        signal_match = model_signal == blog_signal if (model_signal and blog_signal) else None

        comparison = {
            'model': {
                'gmi_score': model_gmi,
                'gmi_signal': model_signal,
                'direction': model_prediction.get('final_prediction', {}).get('direction'),
                'confidence': model_prediction.get('final_prediction', {}).get('confidence')
            },
            'blog': {
                'gmi_score': blog_gmi,
                'gmi_signal': blog_signal,
                'post_date': self.blog_data.get('post_date'),
                'trend_day': self.blog_data.get('qqq_trend_day'),
                'trend_direction': self.blog_data.get('trend_direction'),
                'observations': self.blog_data.get('key_observations', [])
            },
            'alignment': {
                'gmi_score_match': gmi_match,
                'signal_match': signal_match,
                'gmi_difference': abs(model_gmi - blog_gmi) if (model_gmi and blog_gmi) else None
            },
            'recommendation': self._generate_recommendation(model_gmi, blog_gmi, model_signal, blog_signal)
        }

        return comparison

    def _generate_recommendation(
        self,
        model_gmi: int,
        blog_gmi: int,
        model_signal: str,
        blog_signal: str
    ) -> str:
        """Generate recommendation based on model vs blog comparison."""

        if model_gmi is None or blog_gmi is None:
            return "Unable to compare - missing data"

        diff = abs(model_gmi - blog_gmi)

        if diff == 0:
            return "Model and blog ALIGNED - high confidence in signal"
        elif diff == 1:
            return "Minor difference - proceed with caution, use model signal"
        elif diff == 2:
            return "Moderate difference - consider reducing position size"
        else:
            return f"SIGNIFICANT DIVERGENCE ({diff} points) - recommend NO TRADE until signals align"

    def generate_comparison_report(self, model_prediction: Dict) -> str:
        """Generate formatted comparison report."""

        comparison = self.compare_to_model(model_prediction)

        if 'error' in comparison:
            return f"Blog Comparison Error: {comparison['error']}"

        model = comparison['model']
        blog = comparison['blog']
        alignment = comparison['alignment']

        lines = []
        lines.append("")
        lines.append("=" * 65)
        lines.append("MODEL vs WISHING WEALTH BLOG COMPARISON")
        lines.append("=" * 65)
        lines.append("")

        # Side by side comparison
        lines.append(f"{'Metric':<25} {'Our Model':<20} {'Blog':<20}")
        lines.append("-" * 65)
        lines.append(f"{'GMI Score':<25} {model['gmi_score'] or 'N/A':<20} {blog['gmi_score'] or 'N/A':<20}")
        lines.append(f"{'GMI Signal':<25} {model['gmi_signal'] or 'N/A':<20} {blog['gmi_signal'] or 'N/A':<20}")
        lines.append(f"{'Direction':<25} {model['direction'] or 'N/A':<20} {blog['trend_direction'] or 'N/A':<20}")

        if blog['trend_day']:
            lines.append(f"{'Trend Day':<25} {'-':<20} Day {blog['trend_day']:<17}")

        lines.append("")

        # Show blog post date for context
        if blog.get('post_date'):
            lines.append(f"Blog Post Date: {blog['post_date']}")
            lines.append("NOTE: Blog data may be from prior trading day")
            lines.append("")

        # Alignment status
        if alignment['gmi_score_match']:
            lines.append("[OK] GMI SCORES MATCH")
        elif alignment['gmi_difference']:
            lines.append(f"[!!] GMI DIFFERENCE: {alignment['gmi_difference']} points")
            if alignment['gmi_difference'] >= 2:
                lines.append("     (Large divergence - check if blog is stale)")

        if alignment['signal_match']:
            lines.append("[OK] SIGNALS MATCH")
        elif alignment['signal_match'] is False:
            lines.append("[XX] SIGNALS DIVERGE")

        lines.append("")

        # Blog observations
        if blog['observations']:
            lines.append("Blog Observations:")
            for obs in blog['observations']:
                lines.append(f"  - {obs}")
            lines.append("")

        # Recommendation
        lines.append("-" * 65)
        lines.append(f"RECOMMENDATION: {comparison['recommendation']}")
        lines.append("=" * 65)

        return "\n".join(lines)


def fetch_and_compare(model_prediction: Dict) -> str:
    """
    Convenience function to fetch blog and generate comparison.

    Args:
        model_prediction: Output from QQQWishingWealthModel.predict()

    Returns:
        Formatted comparison report string
    """
    comparator = BlogComparison()
    return comparator.generate_comparison_report(model_prediction)
