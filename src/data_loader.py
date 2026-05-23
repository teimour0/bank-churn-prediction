"""
Pulls the Kaggle churn dataset onto disk and reads it into a DataFrame.

The Kaggle API tends to be a bit fragile (auth issues on Windows, rate limits,
the occasional 503), so the loader is deliberately forgiving: if the API call
fails for any reason, we fall back to a local copy under data/raw/.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

DATASET_SLUG = "shubh0799/churn-modelling"
RAW_DIR = Path("data") / "raw"
CSV_NAME = "Churn_Modelling.csv"


def _try_kaggle_download(target_dir: Path) -> bool:
    """Attempt the Kaggle download. Returns True on success, False otherwise."""
    # Importing kaggle at module level eagerly looks for kaggle.json on some
    # versions, so we wrap the import itself in a try/except. The error
    # surface here is fiddly, so we report it honestly instead of pretending
    # the package is missing.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except OSError as e:
        # Common case on a fresh machine: package installs fine, but importing
        # it triggers the credentials check and raises OSError.
        print(f"[data_loader] Kaggle credentials missing ({e}); skipping API.")
        return False
    except ImportError as e:
        print(f"[data_loader] kaggle package not installed ({e}); skipping API.")
        return False
    except Exception as e:
        print(f"[data_loader] Kaggle import failed ({e}); skipping API.")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"[data_loader] Kaggle auth failed ({e}); will look for a local file.")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.dataset_download_files(DATASET_SLUG, path=str(target_dir), quiet=False)
    except Exception as e:
        print(f"[data_loader] Kaggle download failed ({e}); falling back to local.")
        return False

    # Kaggle ships a zip; unpack any zip we find in the target directory.
    for zpath in target_dir.glob("*.zip"):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(target_dir)
        zpath.unlink()

    return (target_dir / CSV_NAME).exists()


def load_data(force_download: bool = False) -> pd.DataFrame:
    """
    Return the churn dataset as a DataFrame.

    If a local CSV already exists we use it as-is. Pass force_download=True to
    re-pull from Kaggle even when the file is already on disk.
    """
    csv_path = RAW_DIR / CSV_NAME

    if force_download or not csv_path.exists():
        ok = _try_kaggle_download(RAW_DIR)
        if not ok and not csv_path.exists():
            abs_target = csv_path.resolve()
            abs_target.parent.mkdir(parents=True, exist_ok=True)
            msg = (
                "\n"
                "Could not load the dataset automatically.\n"
                "\n"
                "Fastest fix (no Kaggle API setup required):\n"
                "  1. Open this page in a browser and click the Download button:\n"
                "     https://www.kaggle.com/datasets/shubh0799/churn-modelling\n"
                "  2. Unzip the archive if needed.\n"
                f"  3. Move Churn_Modelling.csv to:\n"
                f"     {abs_target}\n"
                "  4. Re-run the pipeline.\n"
                "\n"
                "Alternative (Kaggle API):\n"
                "  - Create an API token at https://www.kaggle.com/settings\n"
                f"  - Save kaggle.json at: {Path.home() / '.kaggle' / 'kaggle.json'}\n"
                "  - Re-run the pipeline.\n"
            )
            raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    print(f"[data_loader] loaded {len(df):,} rows from {csv_path}")
    return df


if __name__ == "__main__":
    # Handy for ad-hoc runs: python -m src.data_loader
    df = load_data()
    print(df.head())
    print(df.dtypes)
