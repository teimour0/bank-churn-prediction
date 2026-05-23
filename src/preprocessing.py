"""
Cleaning and basic encoding for the churn dataset.

There is no real "missing data" in this file. The Kaggle version is almost
suspiciously tidy. The preprocessing here is mostly about dropping useless
columns, normalising types, and producing a frame that downstream code can
trust without re-checking the schema on every call.
"""

from __future__ import annotations

import pandas as pd

# Columns that carry no signal for modeling. Surname especially: keeping it
# would basically let the model memorise individuals if it had enough capacity.
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy copy of the raw dataframe."""
    out = df.copy()

    # Drop any of the noise columns that happen to be present. Doing it this
    # way means the function still works if the upstream file shape changes.
    present_drops = [c for c in DROP_COLUMNS if c in out.columns]
    if present_drops:
        out = out.drop(columns=present_drops)

    # A few rows in some Kaggle copies of this dataset have stray whitespace
    # in the Geography column. Catch it here, once.
    for col in ["Geography", "Gender"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    # Boolean-ish columns are stored as 0/1 ints in the file. Leave them as
    # int8 so memory stays modest.
    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        if col in out.columns:
            out[col] = out[col].astype("int8")

    return out


def encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical columns. Returns a numeric-only frame."""
    cat_cols = [c for c in ["Geography", "Gender"] if c in df.columns]
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def split_features_target(df: pd.DataFrame, target: str = "Exited"):
    """Split into X and y. Kept as a tiny helper so call sites stay readable."""
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not present in dataframe")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y
