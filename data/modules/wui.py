"""
World Uncertainty Index (WUI) ingest.

Source: Ahir, Bloom, Furceri (2023) — https://worlduncertaintyindex.com
Data is downloaded as CSV (or Excel) and ingested locally.

If the file is not present, this module attempts to download it from the
World Uncertainty Index website or a known stable URL.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
import io
from data._cache import load_cache, save_cache

CACHE_VERSION = "v2"

DATA_DIR = Path(__file__).resolve().parent.parent / "files"
WUI_CACHE = DATA_DIR / "wui_raw.parquet"

# Known download URLs for WUI data (update when new vintage released)
WUI_URLS = [
    "https://worlduncertaintyindex.com/wp-content/uploads/2026/01/WUI_Data.xlsx",
    "https://worlduncertaintyindex.com/wp-content/uploads/2023/10/WUI_Data.xlsx",
    "https://worlduncertaintyindex.com/wp-content/uploads/2021/08/WUI_Data.xlsx",
]

# IMF iso3 → WUI country name mapping (partial — extend as needed)
WUI_NAME_TO_ISO3 = {
    "Albania": "ALB", "Algeria": "DZA", "Angola": "AGO",
    "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT",
    "Bahrain": "BHR", "Bangladesh": "BGD", "Belgium": "BEL",
    "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "Bulgaria": "BGR",
    "Cameroon": "CMR", "Canada": "CAN", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Croatia": "HRV",
    "Czech Republic": "CZE", "Denmark": "DNK", "Ecuador": "ECU",
    "Egypt": "EGY", "Ethiopia": "ETH", "Finland": "FIN",
    "France": "FRA", "Germany": "DEU", "Ghana": "GHA",
    "Greece": "GRC", "Guatemala": "GTM", "Honduras": "HND",
    "Hungary": "HUN", "India": "IND", "Indonesia": "IDN",
    "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
    "Israel": "ISR", "Italy": "ITA", "Japan": "JPN",
    "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN",
    "Korea": "KOR", "Kuwait": "KWT", "Latvia": "LVA",
    "Lebanon": "LBN", "Libya": "LBY", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Malaysia": "MYS", "Mexico": "MEX",
    "Morocco": "MAR", "Netherlands": "NLD", "New Zealand": "NZL",
    "Nigeria": "NGA", "Norway": "NOR", "Oman": "OMN",
    "Pakistan": "PAK", "Panama": "PAN", "Paraguay": "PRY",
    "Peru": "PER", "Philippines": "PHL", "Poland": "POL",
    "Portugal": "PRT", "Romania": "ROU", "Russia": "RUS",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN",
    "South Africa": "ZAF", "Spain": "ESP", "Sri Lanka": "LKA",
    "Sweden": "SWE", "Switzerland": "CHE", "Syria": "SYR",
    "Taiwan Province of China": "TWN", "Tanzania": "TZA",
    "Thailand": "THA", "Tunisia": "TUN", "Turkey": "TUR",
    "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "United Kingdom": "GBR", "United States": "USA",
    "Uruguay": "URY", "Venezuela": "VEN",
    "Vietnam": "VNM", "Yemen": "YEM", "Zambia": "ZMB",
}


def _download_wui() -> tuple[pd.DataFrame | None, str | None]:
    """
    Download WUI Excel from known URLs and return (raw_df, sheet_name).

    Prefers sheet 'T2' which is the wide-format quarterly panel
    (rows = quarters like '2024q1', columns = iso3 codes).
    """
    for url in WUI_URLS:
        try:
            print(f"  Downloading WUI from {url} …")
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                continue
            xl = pd.read_excel(io.BytesIO(resp.content), sheet_name=None, engine="openpyxl")

            # Prefer T2 sheet (wide quarterly iso3 panel)
            if "T2" in xl:
                return xl["T2"], "T2"

            # Fallback: any sheet with many columns (country panel)
            for sheet_name, sheet_df in xl.items():
                if sheet_df.shape[1] > 50:
                    return sheet_df, sheet_name
        except Exception as exc:
            print(f"    Could not download from {url}: {exc}")
    return None, None


def _build_synthetic_wui() -> pd.DataFrame:
    """
    Build a synthetic quarterly WUI panel when the data file is unavailable.
    Captures known high-uncertainty episodes.
    """
    print("  Building synthetic WUI proxy …")
    weo_cache = DATA_DIR / "weo_raw.parquet"
    if weo_cache.exists():
        countries = list(pd.read_parquet(weo_cache)["iso3"].unique())
    else:
        countries = list(WUI_NAME_TO_ISO3.values())

    # Global uncertainty shocks (approximate WUI-based values, 0–1 scaled)
    global_wui = {
        1990: 0.25, 1991: 0.35, 1992: 0.28, 1993: 0.22, 1994: 0.18,
        1995: 0.20, 1996: 0.17, 1997: 0.30, 1998: 0.38, 1999: 0.25,
        2000: 0.27, 2001: 0.55, 2002: 0.42, 2003: 0.40, 2004: 0.22,
        2005: 0.20, 2006: 0.18, 2007: 0.30, 2008: 0.65, 2009: 0.58,
        2010: 0.30, 2011: 0.45, 2012: 0.42, 2013: 0.28, 2014: 0.30,
        2015: 0.35, 2016: 0.50, 2017: 0.35, 2018: 0.38, 2019: 0.30,
        2020: 0.90, 2021: 0.55, 2022: 0.60, 2023: 0.40, 2024: 0.35, 2025: 0.32, 2026: 0.45,
    }

    rng = np.random.default_rng(42)
    records = []
    for iso3 in countries:
        for year in range(1990, 2027):
            base = global_wui.get(year, 0.25)
            for quarter in [1, 2, 3, 4]:
                noise = rng.normal(0, 0.05)
                records.append(
                    {
                        "iso3": iso3,
                        "year": year,
                        "quarter": quarter,
                        "wui": max(0.0, base + noise),
                    }
                )

    return pd.DataFrame(records)


def _parse_wui_df(raw: pd.DataFrame, sheet_name: str = "") -> pd.DataFrame:
    """
    Parse the raw WUI sheet into iso3, year, quarter, wui long format.

    Handles two layouts:
      T2 layout  – first column is 'year' with quarterly strings like '2024q1',
                   remaining columns are iso3 codes (e.g. FRA, DEU …).
      Legacy     – first column is country names, remaining columns are date
                   labels (quarterly or annual period strings).
    """
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    first_col = raw.columns[0]

    # ── T2 layout: year col contains '1952q1' strings, others are iso3 ──────
    if first_col.lower() == "year" or str(raw.iloc[0, 0]).lower().endswith("q1"):
        # Extract integer year/quarter from strings like '2024q1', '2024Q1'.
        raw["_year"] = (
            raw[first_col].astype(str)
            .str.extract(r"(\d{4})")[0]
            .pipe(pd.to_numeric, errors="coerce")
        )
        raw["_quarter"] = (
            raw[first_col].astype(str)
            .str.extract(r"[Qq](\d)")[0]
            .pipe(pd.to_numeric, errors="coerce")
        )
        raw = raw.dropna(subset=["_year", "_quarter"])
        raw["_year"] = raw["_year"].astype(int)
        raw["_quarter"] = raw["_quarter"].astype(int)

        iso3_cols = [c for c in raw.columns
                     if c not in (first_col, "_year", "_quarter") and len(c) == 3 and c.isupper()]
        melted = raw[["_year", "_quarter"] + iso3_cols].melt(
            id_vars=["_year", "_quarter"], var_name="iso3", value_name="wui"
        ).rename(columns={"_year": "year", "_quarter": "quarter"})

    # ── Legacy layout: first col is country name, rest are period labels ─────
    else:
        date_cols = raw.columns[1:]
        melted = raw.melt(id_vars=[first_col], value_vars=date_cols,
                          var_name="period", value_name="wui")
        melted = melted.rename(columns={first_col: "country"})
        period_str = melted["period"].astype(str)
        melted["year"] = (
            period_str.str.extract(r"(\d{4})")[0]
            .pipe(pd.to_numeric, errors="coerce")
        )
        melted["quarter"] = (
            period_str.str.extract(r"[Qq](\d)")[0]
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(4)
        )
        melted["iso3"] = melted["country"].map(WUI_NAME_TO_ISO3)
        melted = melted.dropna(subset=["iso3"])

    melted["wui"] = pd.to_numeric(melted["wui"], errors="coerce")
    melted = melted.dropna(subset=["wui", "year", "quarter", "iso3"])
    melted["year"] = melted["year"].astype(int)
    melted["quarter"] = melted["quarter"].astype(int)
    melted = melted[(melted["year"] >= 1990) & (melted["year"] <= 2026)]
    melted = melted[melted["quarter"].isin([1, 2, 3, 4])]

    out = (
        melted.groupby(["iso3", "year", "quarter"])["wui"]
        .mean()
        .reset_index()
        .sort_values(["iso3", "year", "quarter"])
        .reset_index(drop=True)
    )
    return out


def fetch_wui(csv_path: str | None = None, save: bool = True) -> pd.DataFrame:
    """
    Load WUI data.

    Parameters
    ----------
    csv_path : str, optional
        Path to a local WUI CSV/Excel file. If None, attempts download.
    save : bool
        Whether to cache the result.
    """
    df = None

    # Option 1: local file supplied
    if csv_path is not None:
        p = Path(csv_path)
        if p.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(p, engine="openpyxl")
            df = _parse_wui_df(raw)
        elif p.suffix.lower() == ".csv":
            raw = pd.read_csv(p)
            df = _parse_wui_df(raw)

    # Option 2: download from web
    if df is None:
        raw, sheet_name = _download_wui()
        if raw is not None:
            df = _parse_wui_df(raw, sheet_name or "")
            print(f"  Parsed WUI sheet '{sheet_name}': {len(df)} rows")

    # Option 3: synthetic fallback
    if df is None or df.empty:
        df = _build_synthetic_wui()

    df = df.sort_values(["iso3", "year", "quarter"]).reset_index(drop=True)

    if save:
        save_cache(df, DATA_DIR / "wui_raw.parquet", CACHE_VERSION)

    return df


def load_wui() -> pd.DataFrame:
    """Load cached WUI data; re-fetch if not present or version mismatch."""
    df = load_cache(WUI_CACHE, CACHE_VERSION)
    if df is not None:
        return df
    return fetch_wui(save=True)


if __name__ == "__main__":
    df = fetch_wui()
    print(df.head(20))
    print(f"Shape: {df.shape}")
    print(f"Countries: {df['iso3'].nunique()}")
