"""
Eurostat Labour Cost Index by NACE Rev. 2 — quarterly, unadjusted.

Source: Eurostat SDMX 3.0 API, dataset lc_lci_r2_q.
  https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/
  lc_lci_r2_q/1.0

Dimensions kept for maximum granularity (aggregation deferred to panel builder):
  - freq    : Q  (quarterly)
  - s_adj   : NSA (unadjusted)
  - unit    : I20 (index, 2020=100)
  - nace_r2 : all available sectors (B, B-E, B-F, B-N, B-S, C, D, E, F, G,
                G-J, G-N, H, I, J, K, K-N, L, M, N, O, O-S, P, Q, R, S)
  - lcstruct: D11 (wages & salaries), D12_D4_MD5 (other labour costs),
              D1_D4_MD5 (total labour costs)

Output columns: iso3, year, quarter, nace_r2, lcstruct, labour_cost_idx
Saves to data/labour_costs_raw.parquet.
"""

import gzip
import requests
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from data._cache import load_cache, save_cache

CACHE_VERSION = "v1"

DATA_DIR = Path(__file__).resolve().parent.parent / "files"

EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/"
    "lc_lci_r2_q/1.0"
)

# Eurostat 2-letter geo codes → ISO3
# Note: Eurostat uses EL for Greece and UK for United Kingdom
GEO_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP", "FI": "FIN",
    "FR": "FRA", "EL": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT",
    "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK",
    "UK": "GBR", "NO": "NOR",
}

EU_PANEL_COUNTRIES = [
    "FRA", "DEU", "ITA", "ESP", "GRC", "PRT", "NLD", "BEL", "AUT",
    "FIN", "IRL", "LUX", "SVN", "SVK", "EST", "LVA", "LTU", "CZE",
    "HUN", "POL", "ROU", "BGR", "HRV", "CYP", "MLT", "DNK", "SWE",
    "GBR", "NOR",
]

# All NACE Rev.2 aggregates available in the dataset
NACE_CODES = [
    "B", "B-E", "B-F", "B-N", "B-S", "C", "D", "E", "F",
    "G", "G-J", "G-N", "H", "I", "J", "K", "K-N", "L",
    "M", "N", "O", "O-S", "P", "Q", "R", "S",
]

# Labour cost structure components
LCSTRUCT_CODES = ["D11", "D12_D4_MD5", "D1_D4_MD5"]


def _fetch_eurostat_labour_costs() -> pd.DataFrame | None:
    """
    Download quarterly labour cost indices (unadjusted, I20) for all panel
    countries with full NACE and lcstruct breakdown.

    Returns DataFrame with [iso3, year, quarter, nace_r2, lcstruct,
    labour_cost_idx] or None on failure.
    """
    geo_codes = sorted(GEO_TO_ISO3.keys())
    params = {
        "c[freq]": "Q",
        "c[s_adj]": "NSA",
        "c[unit]": "I20",
        "c[nace_r2]": ",".join(NACE_CODES),
        "c[lcstruct]": ",".join(LCSTRUCT_CODES),
        "c[geo]": ",".join(geo_codes),
        "startPeriod": "1996-Q1",
        "compress": "true",
        "format": "csvdata",
        "formatVersion": "2.0",
        "lang": "en",
    }
    try:
        resp = requests.get(EUROSTAT_URL, params=params, timeout=300)
        if resp.status_code != 200:
            print(f"  Eurostat labour costs returned HTTP {resp.status_code}")
            return None

        content = resp.content
        try:
            content = gzip.decompress(content)
        except Exception:
            pass  # already decompressed or not gzip

        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as exc:
        print(f"  Eurostat labour costs fetch error: {exc}")
        return None

    if df.empty:
        return None

    # Normalise column names
    df.columns = [c.strip().upper().split("\\")[0] for c in df.columns]

    time_col     = next((c for c in df.columns if c == "TIME_PERIOD"), None)
    geo_col      = next((c for c in df.columns if c == "GEO"), None)
    val_col      = next((c for c in df.columns if c == "OBS_VALUE"), None)
    nace_col     = next((c for c in df.columns if c == "NACE_R2"), None)
    lcstruct_col = next((c for c in df.columns if c == "LCSTRUCT"), None)

    if not all([time_col, geo_col, val_col, nace_col, lcstruct_col]):
        print(f"  Unexpected CSV columns: {df.columns.tolist()}")
        return None

    df["iso3"]           = df[geo_col].map(GEO_TO_ISO3)
    df["value"]          = pd.to_numeric(df[val_col], errors="coerce")
    df["nace_r2"]        = df[nace_col].str.strip()
    df["lcstruct"]       = df[lcstruct_col].str.strip()

    # Parse TIME_PERIOD: "2015-Q3" → year=2015, quarter=3
    df["year"]    = pd.to_numeric(df[time_col].str[:4], errors="coerce")
    df["quarter"] = pd.to_numeric(
        df[time_col].str.extract(r"Q(\d)")[0], errors="coerce"
    )

    df = df.dropna(subset=["iso3", "year", "quarter", "nace_r2", "lcstruct", "value"])
    df["year"]    = df["year"].astype(int)
    df["quarter"] = df["quarter"].astype(int)

    result = (
        df[["iso3", "year", "quarter", "nace_r2", "lcstruct", "value"]]
        .rename(columns={"value": "labour_cost_idx"})
        .sort_values(["iso3", "nace_r2", "lcstruct", "year", "quarter"])
        .reset_index(drop=True)
    )

    n_countries = result["iso3"].nunique()
    yr_min = int(result["year"].min())
    yr_max = int(result["year"].max())
    print(
        f"  Eurostat labour costs: {n_countries} countries, "
        f"{result['nace_r2'].nunique()} NACE sectors, "
        f"{yr_min}Q1–{yr_max}Q{result.loc[result['year']==yr_max,'quarter'].max()}"
    )
    return result


def fetch_labour_costs(
    countries: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch quarterly labour cost indices (I20, NSA) by NACE sector.

    Parameters
    ----------
    countries : optional list of ISO3 codes to filter (defaults to all panel countries)
    save      : persist raw data to data/labour_costs_raw.parquet

    Returns
    -------
    DataFrame with columns [iso3, year, quarter, nace_r2, lcstruct, labour_cost_idx].
    """
    if countries is None:
        countries = EU_PANEL_COUNTRIES

    print("Fetching labour cost data (Eurostat lc_lci_r2_q) …")
    df = _fetch_eurostat_labour_costs()

    if df is None or df.empty:
        print("  No labour cost data retrieved; returning empty frame.")
        return pd.DataFrame(
            columns=["iso3", "year", "quarter", "nace_r2", "lcstruct", "labour_cost_idx"]
        )

    df = df[df["iso3"].isin(countries)].copy()

    if save:
        save_cache(df, DATA_DIR / "labour_costs_raw.parquet", CACHE_VERSION)

    return df


def load_labour_costs() -> pd.DataFrame:
    """Load cached labour cost data; re-fetch if cache is absent or stale."""
    df = load_cache(DATA_DIR / "labour_costs_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_labour_costs(save=True)


if __name__ == "__main__":
    df = load_labour_costs()
    print(df.dtypes)
    print(f"\nShape: {df.shape}")
    print(f"NACE sectors: {sorted(df['nace_r2'].unique())}")
    print(f"lcstruct:     {sorted(df['lcstruct'].unique())}")
    print(df[df["iso3"] == "DEU"].tail(12))
