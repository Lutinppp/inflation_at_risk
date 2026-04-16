"""
EC competitiveness indicators — NEER and REER, monthly.

Source: European Commission, DG ECFIN — m42neer.xlsx / m42reer.xlsx.
  NEER: Nominal Effective Exchange Rate (index)
    REER: Real Effective Exchange Rate, HICP-deflated

Raw workbooks use a decimal-year date format (e.g. 1994.01 = Jan 1994) and
country names as column headers.  This module tidies them into long-format
parquets with [iso3, year, month, <indicator>] for panel-builder consumption.

Saves:
  data/neer_raw.parquet  — [iso3, year, month, neer]
    data/reer_raw.parquet  — [iso3, year, month, reer_hicp]
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
from data._cache import load_cache, save_cache

CACHE_VERSION = "v2"

DATA_DIR = Path(__file__).resolve().parent.parent / "files"

NEER_URL = "https://ec.europa.eu/economy_finance/db_indicators/competitiveness/documents/m42neer.xlsx"
REER_URL = "https://ec.europa.eu/economy_finance/db_indicators/competitiveness/documents/m42reer.xlsx"

NEER_FILE = DATA_DIR / "m42neer.xlsx"
REER_FILE = DATA_DIR / "m42reer.xlsx"

# Workbook country-name columns → ISO3
# Only panel-relevant countries; others are silently dropped
COL_TO_ISO3 = {
    "Belgium": "BEL", "Bulgaria": "BGR", "Czech_Rep": "CZE",
    "Denmark": "DNK", "Germany": "DEU", "Estonia": "EST",
    "Ireland": "IRL", "Greece": "GRC", "Spain": "ESP",
    "France": "FRA", "Croatia": "HRV", "Italy": "ITA",
    "Cyprus": "CYP", "Latvia": "LVA", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Hungary": "HUN", "Malta": "MLT",
    "Netherlands": "NLD", "Austria": "AUT", "Poland": "POL",
    "Portugal": "PRT", "Romania": "ROU", "Slovenia": "SVN",
    "Slovakia": "SVK", "Finland": "FIN", "Sweden": "SWE",
    "United_Kingdom": "GBR", "Norway": "NOR",
}


def _decimal_year_to_year_month(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert decimal-year values (e.g. 1994.01 → year=1994, month=1;
    1994.12 → year=1994, month=12) to integer year and month series.
    """
    year  = s.astype(int)
    # fractional part encodes the month (0.01 = Jan, 0.12 = Dec)
    month = (s.round(2) - year).mul(100).round().astype(int)
    # guard against floating-point artefacts giving 0
    month = month.clip(lower=1, upper=12)
    return year, month


def _tidy_sheet(df_raw: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Melt a wide EER sheet (date column '0', country columns) into long format.
    Returns [iso3, year, month, <value_col>].
    """
    date_col = "0"
    country_cols = [c for c in df_raw.columns if c in COL_TO_ISO3]

    long = df_raw[[date_col] + country_cols].copy()
    long = long.melt(id_vars=date_col, var_name="country", value_name=value_col)
    long["iso3"] = long["country"].map(COL_TO_ISO3)
    long[value_col] = pd.to_numeric(long[value_col], errors="coerce")

    year, month = _decimal_year_to_year_month(pd.to_numeric(long[date_col], errors="coerce"))
    long["year"]  = year
    long["month"] = month

    return (
        long[["iso3", "year", "month", value_col]]
        .dropna(subset=["iso3", "year", "month", value_col])
        .astype({"year": int, "month": int})
        .sort_values(["iso3", "year", "month"])
        .reset_index(drop=True)
    )


def _download_xlsx(url: str, timeout: int = 60) -> bytes:
    """Download an XLSX file and return its raw bytes."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _workbook_sheets(content: bytes) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(io.BytesIO(content))
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}


# ── NEER ──────────────────────────────────────────────────────────────────────

def fetch_neer(save: bool = True) -> pd.DataFrame:
    """
    Download the NEER workbook and return a tidy monthly DataFrame.
    If save=True, cache the parquet locally.
    """
    print("  Fetching NEER workbook …")
    content = _download_xlsx(NEER_URL)
    if save:
        NEER_FILE.write_bytes(content)

    sheets = _workbook_sheets(content)
    # Workbook has a single sheet; take it regardless of name
    raw = next(iter(sheets.values()))
    df = _tidy_sheet(raw, "neer")

    n = df["iso3"].nunique()
    yr0, yr1 = int(df["year"].min()), int(df["year"].max())
    m1 = int(df.loc[df["year"] == yr1, "month"].max())
    print(f"  NEER: {n} countries, {yr0}-01 – {yr1}-{m1:02d}")

    if save:
        save_cache(df, DATA_DIR / "neer_raw.parquet", CACHE_VERSION)
    return df


def load_neer() -> pd.DataFrame:
    """Load cached NEER parquet; re-fetch if absent or stale."""
    df = load_cache(DATA_DIR / "neer_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_neer(save=True)


# ── REER ──────────────────────────────────────────────────────────────────────

def fetch_reer(save: bool = True) -> pd.DataFrame:
    """
    Download the REER workbook and return a tidy monthly DataFrame for the
    HICP-deflated REER series.
    If save=True, cache the parquet locally.
    """
    print("  Fetching REER workbook …")
    content = _download_xlsx(REER_URL)
    if save:
        REER_FILE.write_bytes(content)

    sheets = _workbook_sheets(content)
    # Sheet names observed: 'M42RHICP', 'M42RCOREHICP'
    # Keep only headline-HICP REER.
    sheet_names = list(sheets.keys())
    hicp_sheet = next(
        (s for s in sheet_names if "CORE" not in s.upper()), sheet_names[0]
    )
    df = _tidy_sheet(sheets[hicp_sheet], "reer_hicp")

    n = df["iso3"].nunique()
    yr0, yr1 = int(df["year"].min()), int(df["year"].max())
    m1 = int(df.loc[df["year"] == yr1, "month"].max())
    print(f"  REER: {n} countries, {yr0}-01 – {yr1}-{m1:02d}")

    if save:
        save_cache(df, DATA_DIR / "reer_raw.parquet", CACHE_VERSION)
    return df


def load_reer() -> pd.DataFrame:
    """Load cached REER parquet; re-fetch if absent or stale."""
    df = load_cache(DATA_DIR / "reer_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_reer(save=True)


# ── Convenience wrapper ────────────────────────────────────────────────────────

def fetch_competitiveness(save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch both NEER and REER and return as (neer_df, reer_df)."""
    return fetch_neer(save=save), fetch_reer(save=save)


if __name__ == "__main__":
    neer = load_neer()
    reer = load_reer()

    print("\n=== NEER ===")
    print(neer.dtypes)
    print(f"Shape: {neer.shape}")
    print(neer[neer["iso3"] == "DEU"].tail(6))

    print("\n=== REER ===")
    print(reer.dtypes)
    print(f"Shape: {reer.shape}")
    print(reer[reer["iso3"] == "DEU"].tail(6))
