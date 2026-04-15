"""
Brent crude oil price annual % change.

Primary source: FRED DCOILBRENTEU (St. Louis Fed, daily ICE Brent USD/bbl).
  https://fred.stlouisfed.org/series/DCOILBRENTEU
  No API key required.

Annual averages are computed from the daily series, then % change is taken.
The compiled fallback table is used when the FRED endpoint is unreachable or
returns insufficient data.

This is a global variable (single world price) broadcast to all EU panel
countries.  Returns [iso3, year, energy_price_chg] for all EU panel countries.
"""

import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from data._cache import load_cache, save_cache

CACHE_VERSION = "v1"

DATA_DIR = Path(__file__).parent

# FRED series for daily ICE Brent crude in USD/bbl
_FRED_BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

# All countries that will appear in the inflation panel
# (energy price is broadcast to all, so same value for every country in a year)
EU_PANEL_COUNTRIES = [
    "FRA", "DEU", "ITA", "ESP", "GRC", "PRT", "NLD", "BEL", "AUT",
    "FIN", "IRL", "LUX", "SVN", "SVK", "EST", "LVA", "LTU", "CZE",
    "HUN", "POL", "ROU", "BGR", "HRV", "CYP", "MLT", "DNK", "SWE",
    "GBR", "NOR",
]


def _fetch_fred_brent() -> pd.Series | None:
    """
    Fetch daily ICE Brent crude prices from FRED (DCOILBRENTEU) and
    compute annual averages.  No API key required.

    Returns a pd.Series indexed by year (USD/bbl annual average) or None.
    """
    try:
        resp = requests.get(_FRED_BRENT_URL, timeout=30)
        if resp.status_code != 200:
            print(f"  FRED Brent: HTTP {resp.status_code}")
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", "price"]
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna()
        df["year"] = df["date"].dt.year
        # Require at least 200 trading days for a full-year average;
        # partial current year is included as-is for the latest observation.
        counts  = df.groupby("year")["price"].count()
        full_years = counts[counts >= 200].index
        annual = df[df["year"].isin(full_years)].groupby("year")["price"].mean()
        # Append latest partial year so we always have the most recent reading
        latest_year = df["year"].max()
        if latest_year not in annual.index:
            annual.loc[latest_year] = df[df["year"] == latest_year]["price"].mean()
        annual = annual.sort_index()
        print(f"  FRED Brent: {len(annual)} annual obs ({annual.index[0]}–{annual.index[-1]})")
        return annual
    except Exception as exc:
        print(f"  FRED Brent fetch error: {exc}")
        return None


# Compiled Brent crude oil prices (USD/barrel, annual average)
# Source: IMF DataMapper / World Bank Commodity Markets
_BRENT_PRICES = {
    1990: 23.7, 1991: 19.9, 1992: 19.3, 1993: 17.0, 1994: 15.8,
    1995: 17.0, 1996: 20.7, 1997: 19.1, 1998: 13.1, 1999: 17.9,
    2000: 28.5, 2001: 24.4, 2002: 25.0, 2003: 28.8, 2004: 38.3,
    2005: 54.4, 2006: 65.1, 2007: 72.7, 2008: 97.7, 2009: 61.7,
    2010: 79.5, 2011: 111.0, 2012: 111.7, 2013: 108.7, 2014: 98.9,
    2015: 52.4, 2016: 43.6, 2017: 54.7, 2018: 71.3, 2019: 64.4,
    2020: 41.8, 2021: 70.9, 2022: 99.8, 2023: 82.6, 2024: 80.0,
    2025: 75.0, 2026: 70.0,
}


def _compiled_fallback() -> pd.Series:
    """Return compiled Brent price series."""
    return pd.Series(_BRENT_PRICES).sort_index()


def fetch_energy_prices(
    countries: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Fetch Brent crude annual % change and broadcast to all panel countries.

    Returns DataFrame: iso3, year, energy_price_chg (%).
    """
    print("  Fetching FRED Brent crude (DCOILBRENTEU) …")
    brent_series = _fetch_fred_brent()

    if brent_series is None or len(brent_series) < 10:
        print("  FRED unavailable — using compiled Brent price table.")
        brent_series = _compiled_fallback()
    else:
        # Fill any gaps with compiled fallback (pre-1987 years)
        fallback = _compiled_fallback()
        combined = fallback.copy()
        combined.update(brent_series)
        brent_series = combined.sort_index()

    # Compute annual % change
    brent_pct = brent_series.pct_change() * 100.0
    brent_pct = brent_pct.dropna()

    if countries is None:
        countries = EU_PANEL_COUNTRIES

    records = []
    for iso3 in countries:
        for year, pct in brent_pct.items():
            if pd.notna(pct):
                records.append({
                    "iso3": iso3,
                    "year": int(year),
                    "energy_price_chg": round(float(pct), 2),
                })

    df = pd.DataFrame(records).sort_values(["iso3", "year"]).reset_index(drop=True)

    if save:
        save_cache(df, DATA_DIR / "energy_prices_raw.parquet", CACHE_VERSION)
        print(f"  ({len(df)} rows)")

    return df


def load_energy_prices() -> pd.DataFrame:
    """
    Load energy price data.  Re-fetches from FRED if the cache is absent
    or version-mismatched.
    """
    df = load_cache(DATA_DIR / "energy_prices_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_energy_prices(save=True)


if __name__ == "__main__":
    df = fetch_energy_prices()
    print(df[df["iso3"] == "FRA"].tail(10))
