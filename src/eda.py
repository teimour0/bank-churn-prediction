"""
Exploratory data analysis.

The plots here are the ones I actually found useful while working through the
dataset. They are not exhaustive. For example I left out the obvious credit
score histogram because it just looks like a normal distribution and tells
you nothing about churn.

Every plot is written to reports/figures/ as a PNG. Nothing pops up on
screen, which keeps the pipeline scriptable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DIR = Path("reports") / "figures"


def _setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk", palette="muted")


def _save(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def churn_overview(df: pd.DataFrame) -> Path:
    """Simple bar chart of the overall churn rate. Anchors the whole report."""
    counts = df["Exited"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Stayed", "Churned"], counts.values, color=["#3a86ff", "#e63946"])
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{value:,} ({value / len(df):.0%})",
            ha="center",
        )
    ax.set_title("Overall churn outcome")
    ax.set_ylabel("Customers")
    return _save(fig, "01_churn_overview")


def churn_by_geography(df: pd.DataFrame) -> Path:
    rates = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        x=rates.index, y=rates.values, hue=rates.index,
        palette="coolwarm", legend=False, ax=ax,
    )
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    ax.set_title("Churn rate by country")
    ax.set_ylabel("Share of customers who churned")
    ax.set_xlabel("")
    return _save(fig, "02_churn_by_geography")


def churn_by_age_group(df: pd.DataFrame) -> Path:
    """Needs the engineered AgeGroup column."""
    if "AgeGroup" not in df.columns:
        raise KeyError("AgeGroup column missing. Run features.add_features first.")
    rates = df.groupby("AgeGroup", observed=True)["Exited"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = rates.index.astype(str)
    sns.barplot(x=labels, y=rates.values, hue=labels, palette="YlOrRd",
                legend=False, ax=ax)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    ax.set_title("Churn rate by age group")
    ax.set_ylabel("Share of customers who churned")
    ax.set_xlabel("")
    return _save(fig, "03_churn_by_age_group")


def churn_by_products(df: pd.DataFrame) -> Path:
    rates = df.groupby("NumOfProducts")["Exited"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=rates.index, y=rates.values, hue=rates.index,
                palette="viridis", legend=False, ax=ax)
    for i, v in enumerate(rates.values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center")
    ax.set_title("Churn rate by product count (cross-sell saturation check)")
    ax.set_ylabel("Share of customers who churned")
    ax.set_xlabel("Number of products held")
    return _save(fig, "04_churn_by_products")


def balance_distribution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.kdeplot(
        data=df, x="Balance", hue="Exited", common_norm=False,
        fill=True, ax=ax, palette={0: "#3a86ff", 1: "#e63946"},
    )
    ax.set_title("Account balance distribution, by churn outcome")
    ax.set_xlabel("Balance (EUR)")
    return _save(fig, "05_balance_distribution")


def correlation_heatmap(df: pd.DataFrame) -> Path:
    """Numeric-only correlation. Useful as a quick sanity check."""
    numeric = df.select_dtypes(include=["number"]).copy()
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0,
        ax=ax, cbar_kws={"shrink": 0.7}, annot_kws={"size": 8},
    )
    ax.set_title("Correlation matrix (numeric features)")
    return _save(fig, "06_correlation_heatmap")


def run_all(df: pd.DataFrame) -> dict[str, Path]:
    """Generate every chart and return a dict of name -> file path."""
    _setup()
    paths: dict[str, Path] = {}
    paths["overview"] = churn_overview(df)
    paths["geography"] = churn_by_geography(df)
    if "AgeGroup" in df.columns:
        paths["age_group"] = churn_by_age_group(df)
    paths["products"] = churn_by_products(df)
    paths["balance"] = balance_distribution(df)
    paths["correlation"] = correlation_heatmap(df)
    print(f"[eda] wrote {len(paths)} figures to {FIG_DIR}")
    return paths
