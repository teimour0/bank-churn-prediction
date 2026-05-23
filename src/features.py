"""
Engineered features that came out of the EDA pass.

A couple of these are obvious (age buckets, balance-to-salary ratio), and a
couple were genuine surprises from staring at the data. The "balance but
inactive" flag was the one that made me reconsider how the bank's activity
metric is defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append engineered features. Returns a new DataFrame, leaves input alone."""
    out = df.copy()

    # Balance relative to salary. Customers with five-figure salaries and
    # six-figure balances behave very differently from the other way around.
    salary_floor = out["EstimatedSalary"].replace(0, np.nan)
    out["BalanceSalaryRatio"] = (out["Balance"] / salary_floor).fillna(0.0)

    # Age bands. KMeans and tree models can find these on their own, but
    # having an explicit bucket makes the EDA narrative much easier to write.
    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[17, 30, 45, 60, 100],
        labels=["under_30", "30_44", "45_59", "60_plus"],
    )

    # Tenure relative to age. A customer who has been with the bank for
    # eight years matters more if they are 28 than if they are 68.
    out["TenureByAge"] = out["Tenure"] / out["Age"].clip(lower=1)

    # "Sticky" customers: active, multi-product, has a card. This combo is
    # almost never associated with churn in the data, which is reassuring.
    out["IsSticky"] = (
        (out["IsActiveMember"] == 1)
        & (out["NumOfProducts"] >= 2)
        & (out["HasCrCard"] == 1)
    ).astype("int8")

    # The one that surprised me: customers with a meaningful balance who the
    # bank does not consider "active". These tend to churn at a higher rate.
    out["DormantWithMoney"] = (
        (out["IsActiveMember"] == 0) & (out["Balance"] > 50_000)
    ).astype("int8")

    # Cross-sell saturation flag. NumOfProducts >= 3 is correlated with churn
    # in this dataset, which goes against the usual cross-sell story.
    out["HighProductCount"] = (out["NumOfProducts"] >= 3).astype("int8")

    return out


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the columns we want to feed into a numeric model (post-encoding)."""
    # AgeGroup is categorical-by-construction; the rest are numeric or 0/1.
    drop = {"AgeGroup"}
    return [c for c in df.columns if c not in drop and df[c].dtype != "object"]
