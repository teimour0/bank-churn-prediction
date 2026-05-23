"""
Customer segmentation with KMeans.

The point of doing segmentation alongside a churn classifier is that a single
probability per customer is a bad answer to "who is leaving". Segments give
the business a vocabulary: "the wealthy-but-dormant cluster", "the young
single-product cluster", and so on. A relationship manager can act on that.

I tried k=3 through k=7 and landed on k=4 because the silhouette curve has a
soft knee there and, more importantly, the clusters are interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENTATION_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "IsActiveMember",
]


@dataclass
class SegmentationResult:
    labels: np.ndarray
    centroids: pd.DataFrame  # one row per cluster, columns = features
    profile: pd.DataFrame    # human-readable per-segment summary


def _name_segment(row: pd.Series, medians: pd.Series) -> str:
    """
    Generate a short, opinionated label for a cluster centroid.

    Rules are heuristic on purpose: a segmentation that needs a key-value
    legend before anyone can use it is a segmentation nobody will use.
    The output is a two-part label: an age descriptor and a behaviour
    descriptor, joined with a separator.
    """
    # Age descriptor
    if row["Age"] >= 50:
        age_part = "Senior"
    elif row["Age"] >= medians["Age"] + 5:
        age_part = "Mid-life"
    elif row["Age"] <= medians["Age"] - 5:
        age_part = "Young"
    else:
        age_part = "Mid-life"

    # Behaviour descriptor combines balance level and activity
    active = row["IsActiveMember"] >= 0.5
    multi_product = row.get("NumOfProducts", 1) >= 1.8

    # Treat anything below 30K as a transactor pattern: the customer uses the
    # bank for flow rather than as a savings vehicle.
    if row["Balance"] < 30_000:
        behaviour = "transactor" if active or multi_product else "lapsed"
    elif row["Balance"] >= medians["Balance"] + 30_000:
        behaviour = "high-value engaged" if active else "high-value dormant"
    else:
        behaviour = "engaged saver" if active else "passive saver"

    return f"{age_part} {behaviour}"


def segment_customers(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 7,
) -> SegmentationResult:
    """Fit KMeans on a fixed feature set and return labels plus a profile."""
    work = df[SEGMENTATION_FEATURES].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(work)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)

    # Bring centroids back to the original scale for interpretation.
    centroids_unscaled = scaler.inverse_transform(km.cluster_centers_)
    centroids = pd.DataFrame(centroids_unscaled, columns=SEGMENTATION_FEATURES)

    # Build the profile table that downstream code (and the dashboard) reads.
    work = work.assign(segment=labels)
    profile = (
        work.groupby("segment")
        .agg(
            customers=("Age", "size"),
            avg_age=("Age", "mean"),
            avg_balance=("Balance", "mean"),
            avg_salary=("EstimatedSalary", "mean"),
            avg_products=("NumOfProducts", "mean"),
            active_share=("IsActiveMember", "mean"),
        )
        .round(1)
    )

    # Attach churn rate per segment if we have it. Not required, but useful.
    if "Exited" in df.columns:
        churn_rate = df.assign(segment=labels).groupby("segment")["Exited"].mean()
        profile["churn_rate"] = (churn_rate * 100).round(1)

    medians = work.median(numeric_only=True)
    profile["label"] = [
        _name_segment(centroids.iloc[i], medians) for i in profile.index
    ]
    profile = profile.sort_values("customers", ascending=False)

    return SegmentationResult(
        labels=labels, centroids=centroids, profile=profile
    )
