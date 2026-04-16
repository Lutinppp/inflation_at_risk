"""
Eurostat Food Price Monitoring Tool — monthly food price index.

Source: Eurostat SDMX 3.0 API, dataset prc_fsc_idx (monthly food price index,
unit=I15 [2015=100], indx=HICP, coicop=CP011 [Food]).

Raw monthly index values are stored as-is; aggregation to annual % change is
deferred to the panel builder.

Output columns: iso3, year, month, food_price_idx
Saves to data/food_prices_raw.parquet.
"""

import gzip
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from data._cache import load_cache, save_cache

CACHE_VERSION = "v3"

DATA_DIR = Path(__file__).resolve().parent.parent / "files"

EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/"
    "prc_fsc_idx/1.0"
)

# Eurostat 2-letter geo codes → ISO3
# Note: Eurostat uses EL for Greece and UK for the United Kingdom
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


def _fetch_eurostat_food_prices() -> pd.DataFrame | None:
    """
    Download monthly food price indices (HICP, CP011, I15) from Eurostat
    SDMX 3.0. Returns one row per (iso3, year, month) with the raw index value.

    Returns DataFrame with [iso3, year, month, food_price_idx] or None on failure.
    """
    geo_codes = sorted(GEO_TO_ISO3.keys())
    params = {
        "c[freq]": "M",
        "c[unit]": "I15",
        "c[indx]": "HICP",
        "c[coicop]": "CP011",
        "c[geo]": ",".join(geo_codes),
        "startPeriod": "2000-01",
        "compress": "true",
        "format": "csvdata",
        "formatVersion": "2.0",
        "lang": "en",
    }
    try:
        resp = requests.get(EUROSTAT_URL, params=params, timeout=180)
        if resp.status_code != 200:
            print(f"  Eurostat food prices returned HTTP {resp.status_code}")
            return None

        content = resp.content
        try:
            content = gzip.decompress(content)
        except Exception:
            pass  # already decompressed or not gzip

        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as exc:
        print(f"  Eurostat food prices fetch error: {exc}")
        return None

    if df.empty:
        return None

    # Normalise column names
    df.columns = [c.strip().upper().split("\\")[0] for c in df.columns]

    time_col = next((c for c in df.columns if c == "TIME_PERIOD"), None)
    geo_col  = next((c for c in df.columns if c == "GEO"), None)
    val_col  = next((c for c in df.columns if c == "OBS_VALUE"), None)

    if not all([time_col, geo_col, val_col]):
        print(f"  Unexpected CSV columns: {df.columns.tolist()}")
        return None

    df["iso3"]  = df[geo_col].map(GEO_TO_ISO3)
    # TIME_PERIOD format: "2005-01"
    df["year"]  = pd.to_numeric(df[time_col].str[:4], errors="coerce")
    df["month"] = pd.to_numeric(df[time_col].str[5:7], errors="coerce")
    df["food_price_idx"] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=["iso3", "year", "month", "food_price_idx"])
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    result = (
        df[["iso3", "year", "month", "food_price_idx"]]
        .sort_values(["iso3", "year", "month"])
        .reset_index(drop=True)
    )
    print(
        f"  Eurostat food prices: {result['iso3'].nunique()} countries, "
        f"{result['year'].min()}-{result['month'].min():02d} – "
        f"{result['year'].max()}-{result.loc[result['year']==result['year'].max(), 'month'].max():02d}"
    )
    return result


def fetch_food_prices(
    countries: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch monthly food price indices for EU panel countries.

    Parameters
    ----------
    countries : optional list of ISO3 codes to filter (defaults to all panel countries)
    save      : persist raw data to data/food_prices_raw.parquet

    Returns
    -------
    DataFrame with columns [iso3, year, month, food_price_idx].
    """
    if countries is None:
        countries = EU_PANEL_COUNTRIES

    print("Fetching food price data (Eurostat prc_fsc_idx) …")
    df = _fetch_eurostat_food_prices()

    if df is None or df.empty:
        print("  No food price data retrieved; returning empty frame.")
        return pd.DataFrame(columns=["iso3", "year", "month", "food_price_idx"])

    df = df[df["iso3"].isin(countries)].copy()

    if save:
        save_cache(df, DATA_DIR / "food_prices_raw.parquet", CACHE_VERSION)

    return df


def load_food_prices() -> pd.DataFrame:
    """Load cached food price data; re-fetch if cache is absent or stale."""
    df = load_cache(DATA_DIR / "food_prices_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_food_prices(save=True)


if __name__ == "__main__":
    df = load_food_prices()
    print(df.dtypes)
    print(f"\nShape: {df.shape}")
    print(df[df["iso3"] == "DEU"].tail(15))
    print(df.describe())
