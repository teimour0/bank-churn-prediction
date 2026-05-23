"""
Translate model output into dollars.

Banks do not budget based on ROC-AUC. They budget based on revenue. This
module takes the per-customer churn probability and a rough customer lifetime
value assumption and produces an estimate of revenue at risk, both in total
and broken out by segment.

The numbers are illustrative. The framing is the point: a 5% lift in
recall on the high-balance segment is worth more than a 5% lift on the
zero-balance segment, and a model report that does not say so is leaving
useful information on the table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Rough assumption for annual revenue per retained customer. The README says
# this should be tuned by product mix and tenure in a real engagement.
DEFAULT_CLV = 2_000.0


@dataclass
class RiskBreakdown:
    total_expected_loss: float
    high_risk_count: int
    high_risk_loss: float
    by_segment: pd.DataFrame


def expected_loss_per_customer(
    proba: np.ndarray, clv: float = DEFAULT_CLV
) -> np.ndarray:
    """Expected loss = P(churn) * CLV. Straightforward, but worth naming."""
    return proba * clv


def summarise_risk(
    proba: np.ndarray,
    segments: np.ndarray | None = None,
    clv: float = DEFAULT_CLV,
    high_risk_threshold: float = 0.5,
) -> RiskBreakdown:
    """
    Build a small risk report. If segments is provided, also break it down
    by segment so the dashboard can show which group is bleeding the most.
    """
    expected = expected_loss_per_customer(proba, clv=clv)
    total = float(expected.sum())

    high_mask = proba >= high_risk_threshold
    high_count = int(high_mask.sum())
    high_loss = float(expected[high_mask].sum())

    if segments is not None:
        df = pd.DataFrame(
            {
                "segment": segments,
                "expected_loss": expected,
                "probability": proba,
                "is_high_risk": high_mask.astype(int),
            }
        )
        by_segment = (
            df.groupby("segment")
            .agg(
                customers=("expected_loss", "size"),
                avg_probability=("probability", "mean"),
                expected_loss=("expected_loss", "sum"),
                high_risk_count=("is_high_risk", "sum"),
            )
            .round(2)
            .sort_values("expected_loss", ascending=False)
        )
    else:
        by_segment = pd.DataFrame()

    return RiskBreakdown(
        total_expected_loss=total,
        high_risk_count=high_count,
        high_risk_loss=high_loss,
        by_segment=by_segment,
    )


def format_currency(value: float) -> str:
    """Cheap helper. The dashboard uses it for headline figures."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.0f}"
