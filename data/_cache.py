"""
Cache versioning utility for IaR data fetchers.

Each fetcher module defines a CACHE_VERSION string.  When data is saved,
a sidecar <filename>.ver file is written alongside the parquet.  On load,
the sidecar is checked against the module's CACHE_VERSION; if it is missing
or outdated the cache is considered stale and the caller re-fetches.

Usage in a fetcher module
--------------------------
    from data._cache import load_cache, save_cache

    CACHE_VERSION = "v1"          # bump this whenever the schema or source changes

    def fetch_foo(save=True):
        ...
        if save:
            save_cache(df, DATA_DIR / "foo_raw.parquet", CACHE_VERSION)
        return df

    def load_foo() -> pd.DataFrame:
        df = load_cache(DATA_DIR / "foo_raw.parquet", CACHE_VERSION)
        if df is not None:
            return df
        return fetch_foo(save=True)
"""

from pathlib import Path
import pandas as pd


def load_cache(path: Path, version: str) -> pd.DataFrame | None:
    """
    Return the cached DataFrame if the parquet exists and its version matches.
    Returns None if the cache is absent, version-mismatched, or unreadable.
    """
    ver_path = path.with_suffix(".ver")
    if (
        path.exists()
        and ver_path.exists()
        and ver_path.read_text().strip() == version
    ):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def save_cache(df: pd.DataFrame, path: Path, version: str) -> None:
    """Save DataFrame to parquet and write the version sidecar file."""
    df.to_parquet(path, index=False)
    path.with_suffix(".ver").write_text(version)
    print(f"  Saved → {path}")
