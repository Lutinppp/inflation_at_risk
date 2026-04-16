from pathlib import Path
import pandas as pd


def load_cache(path: Path, version: str) -> pd.DataFrame | None:
    """
    Return cached DataFrame if parquet exists and version sidecar matches.
    Returns None when cache is absent, mismatched, or unreadable.
    """
    ver_path = path.with_suffix('.ver')
    if path.exists() and ver_path.exists() and ver_path.read_text().strip() == version:
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def save_cache(df: pd.DataFrame, path: Path, version: str) -> None:
    """Save parquet and version sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    path.with_suffix('.ver').write_text(version)
    print(f"  Saved -> {path}")
