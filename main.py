"""
End-to-end pipeline.

Run this from the project root:

    python main.py

It downloads the dataset, runs the EDA, fits segmentation, trains and
compares three models, prints a short scorecard, and writes the artifacts
the dashboard later reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import data_loader, economic_impact, eda, features, modeling, preprocessing, segmentation


ARTIFACT_DIR = Path("reports")
MODEL_PATH = "best_model.joblib"


def _print_header(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def run() -> None:
    _print_header("1. Loading data")
    df_raw = data_loader.load_data()

    _print_header("2. Cleaning")
    df = preprocessing.clean(df_raw)
    print(df.head(3))

    _print_header("3. Feature engineering")
    df = features.add_features(df)
    print(f"Columns now: {list(df.columns)}")

    _print_header("4. EDA - writing figures to reports/figures/")
    eda.run_all(df)

    _print_header("5. Customer segmentation")
    seg = segmentation.segment_customers(df, n_clusters=4)
    print(seg.profile.to_string())

    _print_header("6. Training and comparing models")
    df_model = preprocessing.encode_for_model(df.drop(columns=["AgeGroup"]))
    X, y = preprocessing.split_features_target(df_model)
    best, all_results = modeling.train_and_compare(X, y)
    print(modeling.summarise(all_results).to_string(index=False))
    print()
    print(f"Best model on this run: {best.name}")
    print(modeling.detailed_report(best))

    saved = modeling.save_best(best, MODEL_PATH)
    print(f"Saved best model to {saved}")

    _print_header("7. Economic impact - revenue at risk")
    # Score every customer with the best model.
    all_probas = best.model.predict_proba(X)[:, 1]
    risk = economic_impact.summarise_risk(
        proba=all_probas, segments=seg.labels, high_risk_threshold=0.5
    )
    print(
        f"Total expected loss across portfolio: "
        f"{economic_impact.format_currency(risk.total_expected_loss)}"
    )
    print(
        f"High-risk customers (P >= 0.5): {risk.high_risk_count:,} "
        f"with expected loss "
        f"{economic_impact.format_currency(risk.high_risk_loss)}"
    )
    print()
    print("Risk by segment:")
    print(risk.by_segment.to_string())

    # Persist a small table the dashboard can re-read.
    ARTIFACT_DIR.mkdir(exist_ok=True)
    seg.profile.to_csv(ARTIFACT_DIR / "segment_profile.csv")
    risk.by_segment.to_csv(ARTIFACT_DIR / "risk_by_segment.csv")
    print(f"\nWrote segment_profile.csv and risk_by_segment.csv into {ARTIFACT_DIR}/")

    _print_header("Done")
    print("Launch the dashboard with: streamlit run dashboard.py")


if __name__ == "__main__":
    run()
