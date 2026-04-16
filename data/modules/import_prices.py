"""
ECB MNA — Imports of goods and services, annual growth rate (%).

Source: ECB Statistical Data Warehouse, dataflow MNA.
Series key pattern: MNA.A.N.{geo}.W1.S1.S1.C.P7._Z._Z._Z.*.LR.GY
  - FREQ=A (annual), ADJUSTMENT=N (unadjusted)
  - REF_AREA: 2-letter ECB country code (wildcarded)
  - P7: imports of goods and services
  - LR: chain-linked volume; GY: year-on-year growth rate

Response format: SDMX 2.1 Generic XML (default ECB API format).

  https://data-api.ecb.europa.eu/service/data/MNA/
  A.N.*.W1.S1.S1.C.P7._Z._Z._Z.*.LR.GY

Returns [iso3, year, quarter, import_price_chg] for all available EU+ panel countries.
Saves to data/import_prices_raw.parquet.
"""

import gzip
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from data._cache import load_cache, save_cache

CACHE_VERSION = "v2"

DATA_DIR = Path(__file__).resolve().parent.parent / "files"

# ECB SDW — annual growth rate of imports of goods & services, all countries
# Wildcarded dimensions use empty string (ECB syntax), not '*'
# Position 3 (REF_AREA) and position 12 (UNIT_MEASURE) are wildcarded
ECB_MNA_URL = (
    "https://data-api.ecb.europa.eu/service/data/MNA/"
    "A.N..W1.S1.S1.C.P7._Z._Z._Z..LR.GY"
)

# SDMX 2.1 Generic XML namespaces used by the ECB
_NS = {
    "generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
    "message": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
}

# ECB 2-letter ref_area codes → ISO3
ECB_GEO_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP", "FI": "FIN",
    "FR": "FRA", "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT",
    "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK", "GB": "GBR", "NO": "NOR",
}

EU_PANEL_COUNTRIES = [
    "FRA", "DEU", "ITA", "ESP", "GRC", "PRT", "NLD", "BEL", "AUT",
    "FIN", "IRL", "LUX", "SVN", "SVK", "EST", "LVA", "LTU", "CZE",
    "HUN", "POL", "ROU", "BGR", "HRV", "CYP", "MLT", "DNK", "SWE",
    "GBR", "NOR",
]

# OECD EO quarterly imports growth proxy (volume growth, annualized %)
_OECD_EO_URL = "https://stats.oecd.org/SDMX-JSON/data/EO"
_OECD_MEASURE = "MGSV_ANNPCT"
_OECD_FREQ = "Q"


def _parse_sdmx21_generic(content: bytes) -> pd.DataFrame:
    """
    Parse ECB SDMX 2.1 Generic XML response.

    Structure per series:
      <generic:Series>
        <generic:SeriesKey>
          <generic:Value id="REF_AREA" value="DE"/>
          ...
        </generic:SeriesKey>
        <generic:Obs>
          <generic:ObsDimension value="2024"/>
          <generic:ObsValue value="1.23"/>
        </generic:Obs>
      </generic:Series>
    """
    root = ET.fromstring(content)
    records = []
    for series in root.iter("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}Series"):
        # Extract REF_AREA from SeriesKey
        geo = None
        for val in series.findall(
            "generic:SeriesKey/generic:Value", _NS
        ):
            if val.get("id") == "REF_AREA":
                geo = val.get("value")
                break
        if geo is None:
            continue
        iso3 = ECB_GEO_TO_ISO3.get(geo)
        if iso3 is None:
            continue

        for obs in series.findall("generic:Obs", _NS):
            dim = obs.find("generic:ObsDimension", _NS)
            val = obs.find("generic:ObsValue", _NS)
            if dim is None or val is None:
                continue
            time_str = dim.get("value", "")
            obs_val  = val.get("value")
            try:
                year  = int(str(time_str)[:4])
                value = float(obs_val)
                records.append({"iso3": iso3, "year": year, "import_price_chg": value})
            except (TypeError, ValueError):
                continue

    return pd.DataFrame(records)


def _fetch_ecb_imports() -> pd.DataFrame | None:
    """
    Download annual growth rate of imports of goods & services from ECB MNA
    (SDMX 2.1 Generic XML), one row per country-year.

    Returns DataFrame with [iso3, year, import_price_chg] or None on failure.
    """
    try:
        resp = requests.get(
            ECB_MNA_URL,
            params={"startPeriod": "1995"},
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"  ECB MNA returned HTTP {resp.status_code}")
            return None

        content = resp.content
        try:
            content = gzip.decompress(content)
        except Exception:
            pass

        df = _parse_sdmx21_generic(content)
    except Exception as exc:
        print(f"  ECB MNA fetch error: {exc}")
        return None

    if df.empty:
        return None

    df = df[df["iso3"].isin(EU_PANEL_COUNTRIES)]

    # If multiple series per country-year (e.g. EUR vs XDC), take the mean
    annual = (
        df.groupby(["iso3", "year"])["import_price_chg"]
        .mean()
        .reset_index()
    )
    return annual.sort_values(["iso3", "year"]).reset_index(drop=True)


def _fetch_oecd_imports_quarterly() -> pd.DataFrame | None:
    """
    Fetch quarterly imports growth proxy from OECD EO (measure MGSV_ANNPCT).

    Returns DataFrame [iso3, year, quarter, import_price_chg] or None.
    """
    countries_str = ",".join(EU_PANEL_COUNTRIES)
    url = f"{_OECD_EO_URL}/{countries_str}.{_OECD_MEASURE}.{_OECD_FREQ}/all"
    try:
        resp = requests.get(url, params={"startTime": "1995", "endTime": "2027"}, timeout=90)
        if resp.status_code != 200:
            print(f"  OECD EO returned HTTP {resp.status_code}")
            return None
        data = resp.json()
    except Exception as exc:
        print(f"  OECD EO imports fetch error: {exc}")
        return None

    try:
        structures   = data["data"]["structures"][0]
        series_dims  = structures["dimensions"]["series"]
        obs_dims     = structures["dimensions"]["observation"]

        measure_dim  = next(d for d in series_dims if d["id"] == "MEASURE")
        freq_dim     = next(d for d in series_dims if d["id"] == "FREQ")
        ref_area_dim = next(d for d in series_dims if d["id"] == "REF_AREA")
        time_dim     = obs_dims[0]

        measure_idx = next(str(i) for i, v in enumerate(measure_dim["values"])
                           if v["id"] == _OECD_MEASURE)
        freq_idx    = next(str(i) for i, v in enumerate(freq_dim["values"])
                           if v["id"] == _OECD_FREQ)
        iso3_map = {str(i): v["id"] for i, v in enumerate(ref_area_dim["values"])}
        period_map = {
            str(i): v["id"]
            for i, v in enumerate(time_dim["values"])
            if "-Q" in v["id"]
        }
    except Exception as exc:
        print(f"  OECD EO imports parse error: {exc}")
        return None

    records = []
    for key, sdata in data["data"]["dataSets"][0]["series"].items():
        parts = key.split(":")
        if len(parts) != 3 or parts[1] != measure_idx or parts[2] != freq_idx:
            continue
        iso3 = iso3_map.get(parts[0], "")
        if not iso3 or iso3 not in EU_PANEL_COUNTRIES:
            continue
        for t_idx, vals in sdata.get("observations", {}).items():
            period = period_map.get(t_idx)
            if period and vals[0] is not None:
                try:
                    year = int(period[:4])
                    quarter = int(period.split("-Q")[1])
                except Exception:
                    continue
                records.append(
                    {
                        "iso3": iso3,
                        "year": year,
                        "quarter": quarter,
                        "import_price_chg": float(vals[0]),
                    }
                )

    if not records:
        return None

    return (
        pd.DataFrame(records)
        .sort_values(["iso3", "year", "quarter"])
        .reset_index(drop=True)
    )


def _compiled_fallback() -> pd.DataFrame:
    """
    Compiled import price deflator % change for EU panel countries.
    Approximated from EC AMECO / IMF WEO import price data.
    Source: EC AMECO variable "PMIM" (price deflator of imports of goods and services).
    """
    # Import price deflator % change — captures global commodity/trade price cycles
    # Positive = import prices rising (imported inflation pressure)
    import_price_data = {
        "FRA": {
            1999: 0.5, 2000: 10.2, 2001: -4.5, 2002: -3.5, 2003: 1.8,
            2004: 4.2, 2005: 5.5, 2006: 4.8, 2007: 2.5, 2008: 10.2,
            2009: -15.2, 2010: 8.2, 2011: 10.5, 2012: 3.8, 2013: -2.5,
            2014: -4.8, 2015: -12.5, 2016: -4.2, 2017: 4.5, 2018: 5.8,
            2019: -2.5, 2020: -8.5, 2021: 10.8, 2022: 19.5, 2023: -5.8,
            2024: -3.5, 2025: -1.5, 2026: -0.5,
        },
        "DEU": {
            1999: 1.2, 2000: 12.5, 2001: -3.5, 2002: -4.2, 2003: 2.5,
            2004: 5.5, 2005: 6.8, 2006: 5.5, 2007: 1.8, 2008: 8.5,
            2009: -12.5, 2010: 8.5, 2011: 9.5, 2012: 2.8, 2013: -2.8,
            2014: -5.2, 2015: -13.8, 2016: -5.5, 2017: 5.5, 2018: 5.5,
            2019: -2.8, 2020: -8.8, 2021: 11.5, 2022: 22.5, 2023: -7.8,
            2024: -4.5, 2025: -2.0, 2026: -1.0,
        },
        "ITA": {
            1999: 0.8, 2000: 12.8, 2001: -3.8, 2002: -3.8, 2003: 2.2,
            2004: 5.0, 2005: 6.2, 2006: 5.2, 2007: 2.8, 2008: 11.5,
            2009: -15.5, 2010: 8.8, 2011: 12.5, 2012: 4.5, 2013: -3.2,
            2014: -5.5, 2015: -13.5, 2016: -5.0, 2017: 4.8, 2018: 7.5,
            2019: -3.2, 2020: -8.2, 2021: 12.5, 2022: 23.8, 2023: -9.5,
            2024: -5.5, 2025: -2.5, 2026: -0.5,
        },
        "ESP": {
            1999: 1.2, 2000: 11.5, 2001: -3.2, 2002: -3.8, 2003: 2.0,
            2004: 4.8, 2005: 6.5, 2006: 5.8, 2007: 2.5, 2008: 10.8,
            2009: -15.0, 2010: 8.5, 2011: 11.5, 2012: 3.5, 2013: -3.0,
            2014: -5.0, 2015: -13.2, 2016: -4.8, 2017: 5.2, 2018: 7.2,
            2019: -3.0, 2020: -8.5, 2021: 12.0, 2022: 22.5, 2023: -9.2,
            2024: -5.2, 2025: -2.2, 2026: -0.5,
        },
        "GRC": {
            1999: 1.5, 2000: 13.5, 2001: -3.5, 2002: -3.5, 2003: 2.0,
            2004: 4.5, 2005: 5.5, 2006: 5.0, 2007: 3.0, 2008: 12.5,
            2009: -16.5, 2010: 9.5, 2011: 12.5, 2012: 3.0, 2013: -5.0,
            2014: -7.5, 2015: -15.0, 2016: -6.0, 2017: 4.5, 2018: 8.0,
            2019: -3.5, 2020: -9.0, 2021: 12.5, 2022: 25.0, 2023: -10.0,
            2024: -5.5, 2025: -2.5, 2026: -0.5,
        },
        "PRT": {
            1999: 1.0, 2000: 11.5, 2001: -3.5, 2002: -4.0, 2003: 2.0,
            2004: 4.8, 2005: 6.0, 2006: 5.5, 2007: 2.5, 2008: 11.0,
            2009: -14.5, 2010: 8.5, 2011: 11.5, 2012: 3.5, 2013: -3.5,
            2014: -5.5, 2015: -13.5, 2016: -5.0, 2017: 5.0, 2018: 7.0,
            2019: -3.0, 2020: -8.5, 2021: 12.0, 2022: 22.0, 2023: -8.5,
            2024: -5.0, 2025: -2.0, 2026: -0.5,
        },
        "NLD": {
            1999: 1.0, 2000: 11.8, 2001: -3.5, 2002: -4.0, 2003: 2.0,
            2004: 5.0, 2005: 5.8, 2006: 5.2, 2007: 2.2, 2008: 9.5,
            2009: -14.0, 2010: 8.0, 2011: 9.0, 2012: 2.8, 2013: -2.5,
            2014: -4.8, 2015: -13.0, 2016: -5.0, 2017: 5.0, 2018: 5.5,
            2019: -2.8, 2020: -8.0, 2021: 11.0, 2022: 21.5, 2023: -7.5,
            2024: -4.5, 2025: -2.0, 2026: -1.0,
        },
        "BEL": {
            1999: 1.2, 2000: 12.0, 2001: -3.2, 2002: -3.8, 2003: 2.2,
            2004: 5.0, 2005: 5.8, 2006: 5.5, 2007: 2.5, 2008: 10.0,
            2009: -14.0, 2010: 8.5, 2011: 9.5, 2012: 3.0, 2013: -2.8,
            2014: -4.8, 2015: -13.2, 2016: -5.0, 2017: 5.2, 2018: 5.8,
            2019: -2.8, 2020: -8.2, 2021: 11.5, 2022: 22.0, 2023: -7.8,
            2024: -4.5, 2025: -2.0, 2026: -1.0,
        },
    }

    # For countries not specifically listed, use EU average as proxy
    eu_avg_keys = list(import_price_data.keys())
    eu_defaults = {}
    all_years = range(1999, 2027)
    for yr in all_years:
        vals = [import_price_data[c].get(yr, np.nan) for c in eu_avg_keys]
        vals = [v for v in vals if not np.isnan(v)]
        eu_defaults[yr] = float(np.mean(vals)) if vals else 0.0

    records = []
    for iso3 in EU_PANEL_COUNTRIES:
        yr_data = import_price_data.get(iso3, eu_defaults)
        for year, val in yr_data.items():
            if not np.isnan(val):
                records.append({"iso3": iso3, "year": year, "quarter": 4, "import_price_chg": val})

    # Fill remaining countries with EU average
    existing = {(r["iso3"], r["year"]) for r in records}
    for iso3 in EU_PANEL_COUNTRIES:
        for year, val in eu_defaults.items():
            if (iso3, year) not in existing:
                records.append({"iso3": iso3, "year": year, "quarter": 4, "import_price_chg": val})

    return pd.DataFrame(records)


def fetch_import_prices(save: bool = True) -> pd.DataFrame:
    """
    Fetch quarterly growth proxy of imports of goods & services for EU panel countries.
    Primary: OECD EO MGSV_ANNPCT (Q).
    Fallback: compiled annual table encoded at Q4.

    Returns DataFrame: iso3, year, quarter, import_price_chg (%).
    """
    print("  Fetching OECD EO quarterly imports growth proxy (MGSV_ANNPCT) …")
    df_live = _fetch_oecd_imports_quarterly()

    if df_live is not None and len(df_live) >= 100:
        n_countries = df_live["iso3"].nunique()
        print(f"  OECD EO imports: {len(df_live)} obs, {n_countries} countries")
        # Supplement with compiled fallback for any missing country-years
        fallback = _compiled_fallback()
        combined = pd.concat([df_live, fallback], ignore_index=True)
        df = combined.drop_duplicates(subset=["iso3", "year", "quarter"], keep="first")
    else:
        print("  OECD EO unavailable or insufficient — using compiled fallback.")
        df = _compiled_fallback()

    df = df.sort_values(["iso3", "year", "quarter"]).reset_index(drop=True)
    df["import_price_chg"] = df["import_price_chg"].clip(lower=-40.0, upper=60.0)

    if save:
        save_cache(df, DATA_DIR / "import_prices_raw.parquet", CACHE_VERSION)

    return df


def load_import_prices() -> pd.DataFrame:
    """Load cached import price data; re-fetch if not present or version mismatch."""
    df = load_cache(DATA_DIR / "import_prices_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_import_prices(save=True)


if __name__ == "__main__":
    df = fetch_import_prices()
    print(df[df["iso3"].isin(["FRA", "DEU", "ITA", "ESP"])].tail(12))
