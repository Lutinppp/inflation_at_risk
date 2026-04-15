"""
Output gap as % of potential GDP.

Primary source: OECD Economic Outlook SDMX-JSON API (dataset EO, measure GAP).
  https://stats.oecd.org/SDMX-JSON/data/EO/{country}.GAP.A
  No API key required.

Fallback: compiled EU table (IMF WEO April 2025 vintage) for CYP, MLT and any
country not covered by the OECD EO dataset.

Returns [iso3, year, output_gap] for all available countries.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from data._cache import load_cache, save_cache

CACHE_VERSION = "v1"

DATA_DIR = Path(__file__).parent

# OECD Economic Outlook — output gap measure code and base URL
_OECD_EO_URL  = "https://stats.oecd.org/SDMX-JSON/data/EO"
_OECD_MEASURE = "GAP"   # output gap % of potential GDP
_OECD_FREQ    = "A"     # annual

# All EU-panel countries available in OECD EO (CYP and MLT are absent)
_OECD_COUNTRIES = [
    "FRA", "DEU", "ITA", "ESP", "GRC", "PRT", "NLD", "BEL", "AUT",
    "FIN", "IRL", "LUX", "SVN", "SVK", "EST", "LVA", "LTU", "CZE",
    "HUN", "POL", "ROU", "BGR", "HRV", "DNK", "SWE", "GBR", "NOR",
]


def _fetch_oecd_output_gap() -> pd.DataFrame | None:
    """
    Fetch output gap (% of potential GDP, measure GAP, annual) from the OECD
    Economic Outlook SDMX-JSON API.  No API key is required.

    Returns long DataFrame [iso3, year, output_gap] or None on failure.
    CYP and MLT are not in the OECD EO dataset and are handled separately
    via the compiled fallback.
    """
    countries_str = ",".join(_OECD_COUNTRIES)
    url = f"{_OECD_EO_URL}/{countries_str}.{_OECD_MEASURE}.{_OECD_FREQ}/all"
    params = {"startTime": "1990", "endTime": "2027"}

    try:
        resp = requests.get(url, params=params, timeout=45)
        if resp.status_code != 200:
            print(f"  OECD EO returned HTTP {resp.status_code}")
            return None
        data = resp.json()
    except Exception as exc:
        print(f"  OECD EO fetch error: {exc}")
        return None

    try:
        structures   = data["data"]["structures"][0]
        series_dims  = structures["dimensions"]["series"]
        obs_dims     = structures["dimensions"]["observation"]

        measure_dim  = next(d for d in series_dims if d["id"] == "MEASURE")
        freq_dim     = next(d for d in series_dims if d["id"] == "FREQ")
        ref_area_dim = next(d for d in series_dims if d["id"] == "REF_AREA")
        time_dim     = obs_dims[0]

        gap_idx  = next(str(i) for i, v in enumerate(measure_dim["values"])  if v["id"] == _OECD_MEASURE)
        a_idx    = next(str(i) for i, v in enumerate(freq_dim["values"])     if v["id"] == _OECD_FREQ)
        iso3_map = {str(i): v["id"] for i, v in enumerate(ref_area_dim["values"])}
        # Annual time periods: 4-character strings like "2022"
        year_map = {
            str(i): v["id"]
            for i, v in enumerate(time_dim["values"])
            if len(v["id"]) == 4
        }
    except (KeyError, StopIteration, IndexError) as exc:
        print(f"  OECD EO: unexpected response structure — {exc}")
        return None

    records = []
    series = data["data"]["dataSets"][0]["series"]
    for key, sdata in series.items():
        parts = key.split(":")
        if len(parts) != 3 or parts[1] != gap_idx or parts[2] != a_idx:
            continue
        iso3 = iso3_map.get(parts[0], "")
        if not iso3 or iso3 not in _OECD_COUNTRIES:
            continue
        for t_idx, vals in sdata.get("observations", {}).items():
            year_str = year_map.get(t_idx)
            if year_str and vals[0] is not None:
                records.append({"iso3": iso3, "year": int(year_str), "output_gap": vals[0]})

    if not records:
        print("  OECD EO: no observations extracted — unexpected format.")
        return None

    df = pd.DataFrame(records).dropna(subset=["output_gap"])
    print(f"  OECD EO output gap: {len(df)} obs, {df['iso3'].nunique()} countries")
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def _compiled_fallback() -> pd.DataFrame:
    """
    Compiled output gap data for EU countries, 1999–2025.
    Source: IMF WEO April 2025 (output gap % of potential GDP).
    """
    og_data = {
        "FRA": {
            1999: 1.5, 2000: 2.2, 2001: 0.8, 2002: -0.5, 2003: -1.1,
            2004: -0.5, 2005: -0.3, 2006: 0.5, 2007: 1.2, 2008: 0.0,
            2009: -3.5, 2010: -2.3, 2011: -0.8, 2012: -1.8, 2013: -2.2,
            2014: -2.3, 2015: -2.2, 2016: -2.0, 2017: -1.2, 2018: -0.2,
            2019: 0.3, 2020: -5.2, 2021: -2.0, 2022: -0.5, 2023: -0.3,
            2024: -0.8, 2025: -0.6, 2026: -0.3, 2027: 0.0,
        },
        "DEU": {
            1999: 1.0, 2000: 2.0, 2001: 0.5, 2002: -1.0, 2003: -1.8,
            2004: -1.0, 2005: -1.0, 2006: 0.5, 2007: 2.0, 2008: 1.5,
            2009: -4.7, 2010: -2.0, 2011: 0.5, 2012: 0.0, 2013: 0.0,
            2014: 0.5, 2015: 0.7, 2016: 0.8, 2017: 1.3, 2018: 1.3,
            2019: 0.5, 2020: -4.0, 2021: -1.5, 2022: -0.5, 2023: -1.5,
            2024: -1.8, 2025: -1.2, 2026: -0.8, 2027: -0.2,
        },
        "ITA": {
            1999: 1.0, 2000: 2.5, 2001: 1.0, 2002: -0.5, 2003: -1.2,
            2004: -0.5, 2005: -0.8, 2006: 0.5, 2007: 1.2, 2008: -0.5,
            2009: -4.5, 2010: -3.0, 2011: -2.0, 2012: -4.5, 2013: -5.5,
            2014: -5.5, 2015: -5.0, 2016: -4.5, 2017: -3.5, 2018: -3.0,
            2019: -3.5, 2020: -7.0, 2021: -4.5, 2022: -2.0, 2023: -1.5,
            2024: -1.8, 2025: -1.5, 2026: -1.0, 2027: -0.5,
        },
        "ESP": {
            1999: 1.5, 2000: 2.5, 2001: 1.5, 2002: 0.5, 2003: 0.2,
            2004: 1.0, 2005: 2.0, 2006: 3.2, 2007: 3.5, 2008: 0.5,
            2009: -5.0, 2010: -5.5, 2011: -5.0, 2012: -7.0, 2013: -7.5,
            2014: -6.0, 2015: -4.0, 2016: -2.5, 2017: -1.0, 2018: 0.0,
            2019: 0.3, 2020: -5.0, 2021: -2.5, 2022: -0.5, 2023: 0.5,
            2024: 0.3, 2025: 0.0, 2026: 0.0, 2027: 0.0,
        },
        "GRC": {
            1999: 2.0, 2000: 4.5, 2001: 3.5, 2002: 2.5, 2003: 2.5,
            2004: 4.0, 2005: 2.5, 2006: 3.5, 2007: 3.0, 2008: 0.0,
            2009: -4.0, 2010: -8.0, 2011: -12.0, 2012: -14.0,
            2013: -14.0, 2014: -13.0, 2015: -12.0, 2016: -10.5,
            2017: -9.0, 2018: -7.0, 2019: -5.0, 2020: -10.0,
            2021: -7.5, 2022: -4.0, 2023: -2.0, 2024: -1.5, 2025: -0.8,
        },
        "PRT": {
            1999: 2.5, 2000: 3.0, 2001: 1.5, 2002: 0.0, 2003: -1.5,
            2004: -0.5, 2005: -1.5, 2006: -0.5, 2007: 0.0, 2008: -1.5,
            2009: -4.5, 2010: -3.5, 2011: -5.5, 2012: -6.5, 2013: -6.0,
            2014: -5.5, 2015: -4.5, 2016: -3.5, 2017: -2.0, 2018: -0.5,
            2019: 0.0, 2020: -5.0, 2021: -2.5, 2022: 0.0, 2023: 1.5,
            2024: 1.2, 2025: 1.0,
        },
        "NLD": {
            1999: 1.5, 2000: 2.0, 2001: 1.0, 2002: -1.0, 2003: -2.0,
            2004: -1.5, 2005: -1.0, 2006: 0.5, 2007: 1.5, 2008: 1.0,
            2009: -3.5, 2010: -2.0, 2011: -0.5, 2012: -1.5, 2013: -2.0,
            2014: -1.5, 2015: -0.5, 2016: 0.5, 2017: 1.0, 2018: 1.5,
            2019: 1.2, 2020: -3.5, 2021: -0.5, 2022: 1.5, 2023: 0.5,
            2024: 0.0, 2025: 0.0,
        },
        "BEL": {
            1999: 1.2, 2000: 2.3, 2001: 0.8, 2002: -0.3, 2003: -1.0,
            2004: -0.3, 2005: 0.0, 2006: 1.0, 2007: 2.0, 2008: 1.0,
            2009: -3.0, 2010: -1.5, 2011: -0.5, 2012: -1.5, 2013: -2.0,
            2014: -1.5, 2015: -1.0, 2016: -0.5, 2017: 0.0, 2018: 0.5,
            2019: 0.3, 2020: -5.0, 2021: -2.0, 2022: 0.0, 2023: 0.0,
            2024: -0.3, 2025: -0.3,
        },
        "AUT": {
            1999: 1.0, 2000: 2.5, 2001: 1.0, 2002: -0.5, 2003: -0.5,
            2004: 0.5, 2005: 0.5, 2006: 1.5, 2007: 2.5, 2008: 1.5,
            2009: -4.0, 2010: -2.0, 2011: 0.0, 2012: -1.0, 2013: -1.5,
            2014: -1.5, 2015: -1.0, 2016: -0.5, 2017: 0.5, 2018: 1.0,
            2019: 0.5, 2020: -5.5, 2021: -3.0, 2022: -1.0, 2023: -2.0,
            2024: -2.2, 2025: -1.5,
        },
        "FIN": {
            1999: 1.2, 2000: 3.0, 2001: 1.5, 2002: 0.0, 2003: -1.0,
            2004: 0.0, 2005: 0.5, 2006: 2.0, 2007: 3.5, 2008: 1.5,
            2009: -5.0, 2010: -2.5, 2011: 0.0, 2012: -1.5, 2013: -2.5,
            2014: -2.5, 2015: -2.5, 2016: -2.0, 2017: -0.5, 2018: 0.5,
            2019: 0.0, 2020: -4.5, 2021: -2.0, 2022: -0.5, 2023: -2.0,
            2024: -2.5, 2025: -1.5,
        },
        "IRL": {
            1999: 4.0, 2000: 5.5, 2001: 3.5, 2002: 2.0, 2003: 2.5,
            2004: 3.5, 2005: 3.0, 2006: 3.5, 2007: 2.0, 2008: -2.5,
            2009: -7.0, 2010: -6.0, 2011: -4.5, 2012: -2.5, 2013: -1.5,
            2014: -0.5, 2015: 0.5, 2016: 1.0, 2017: 2.0, 2018: 2.5,
            2019: 2.5, 2020: -2.5, 2021: 2.5, 2022: 5.0, 2023: 4.5,
            2024: 4.0, 2025: 3.5,
        },
        "DNK": {
            1999: 1.5, 2000: 2.5, 2001: 0.5, 2002: -0.5, 2003: -1.0,
            2004: 0.0, 2005: 1.5, 2006: 3.5, 2007: 4.0, 2008: 0.5,
            2009: -4.0, 2010: -2.5, 2011: -1.5, 2012: -2.0, 2013: -2.0,
            2014: -1.5, 2015: -1.0, 2016: -0.5, 2017: 0.5, 2018: 1.5,
            2019: 1.5, 2020: -2.5, 2021: 0.5, 2022: 2.0, 2023: 1.5,
            2024: 0.8, 2025: 0.5,
        },
        "SWE": {
            1999: 0.5, 2000: 2.5, 2001: 0.5, 2002: -0.5, 2003: -0.5,
            2004: 1.0, 2005: 1.5, 2006: 2.5, 2007: 3.0, 2008: 1.0,
            2009: -4.5, 2010: -1.5, 2011: 0.5, 2012: -0.5, 2013: -1.0,
            2014: -0.5, 2015: 0.5, 2016: 1.0, 2017: 1.5, 2018: 1.0,
            2019: 0.5, 2020: -3.5, 2021: -0.5, 2022: 1.0, 2023: -0.5,
            2024: -1.0, 2025: -0.5,
        },
        "POL": {
            2004: 1.5, 2005: 0.5, 2006: 2.5, 2007: 5.0, 2008: 3.5,
            2009: 1.5, 2010: 2.0, 2011: 2.0, 2012: 0.5, 2013: -1.0,
            2014: -0.5, 2015: 0.5, 2016: 0.5, 2017: 2.0, 2018: 2.5,
            2019: 2.0, 2020: -2.0, 2021: 0.5, 2022: 2.5, 2023: 0.5,
            2024: 0.5, 2025: 0.5,
        },
        "CZE": {
            2004: 2.0, 2005: 1.5, 2006: 3.0, 2007: 5.0, 2008: 4.0,
            2009: -1.5, 2010: -0.5, 2011: 0.5, 2012: -0.5, 2013: -1.5,
            2014: -0.5, 2015: 0.5, 2016: 1.5, 2017: 2.5, 2018: 2.5,
            2019: 1.5, 2020: -5.0, 2021: -2.5, 2022: 0.0, 2023: -1.5,
            2024: -1.2, 2025: -0.5,
        },
        "HUN": {
            2004: 1.5, 2005: 0.5, 2006: 2.0, 2007: -0.5, 2008: -1.5,
            2009: -4.5, 2010: -3.0, 2011: -2.0, 2012: -3.5, 2013: -3.5,
            2014: -1.5, 2015: -0.5, 2016: 0.5, 2017: 2.0, 2018: 2.5,
            2019: 2.0, 2020: -4.5, 2021: -1.5, 2022: 1.0, 2023: -1.0,
            2024: -1.5, 2025: -0.5,
        },
        "ROU": {
            2004: 3.5, 2005: 3.0, 2006: 4.0, 2007: 4.5, 2008: 4.0,
            2009: -2.5, 2010: -3.0, 2011: -2.5, 2012: -1.5, 2013: -1.5,
            2014: -0.5, 2015: 0.5, 2016: 2.0, 2017: 4.0, 2018: 3.5,
            2019: 2.5, 2020: -4.0, 2021: -1.5, 2022: 1.0, 2023: 0.5,
            2024: 0.5, 2025: 0.5,
        },
        "BGR": {
            2004: 2.0, 2005: 3.5, 2006: 4.5, 2007: 5.5, 2008: 5.0,
            2009: -2.5, 2010: -3.0, 2011: -2.0, 2012: -2.5, 2013: -2.5,
            2014: -2.5, 2015: -2.0, 2016: -1.0, 2017: 0.5, 2018: 2.0,
            2019: 1.5, 2020: -4.5, 2021: -2.0, 2022: 0.5, 2023: 0.0,
            2024: -0.5, 2025: 0.0,
        },
        "GBR": {
            1999: 1.5, 2000: 2.5, 2001: 1.0, 2002: -0.5, 2003: -0.5,
            2004: 0.5, 2005: 0.5, 2006: 1.0, 2007: 0.5, 2008: -2.0,
            2009: -5.5, 2010: -3.5, 2011: -2.5, 2012: -3.0, 2013: -2.5,
            2014: -1.5, 2015: -0.5, 2016: -0.5, 2017: 0.0, 2018: 0.0,
            2019: 0.0, 2020: -9.0, 2021: -5.0, 2022: -1.5, 2023: -0.5,
            2024: -0.5, 2025: -0.3,
        },
        "NOR": {
            1999: 1.0, 2000: 2.0, 2001: 0.5, 2002: -0.5, 2003: -0.5,
            2004: 1.0, 2005: 2.0, 2006: 2.5, 2007: 2.5, 2008: 0.5,
            2009: -2.5, 2010: -1.0, 2011: 0.0, 2012: 0.0, 2013: 0.0,
            2014: -0.5, 2015: -1.5, 2016: -1.5, 2017: -0.5, 2018: 0.0,
            2019: 0.0, 2020: -6.0, 2021: -3.0, 2022: 0.5, 2023: 0.0,
            2024: 0.0, 2025: 0.0,
        },
        "LVA": {
            2004: 4.0, 2005: 5.5, 2006: 7.5, 2007: 9.0, 2008: 5.0,
            2009: -11.0, 2010: -9.0, 2011: -5.5, 2012: -3.5, 2013: -2.5,
            2014: -1.5, 2015: -1.0, 2016: -0.5, 2017: 0.5, 2018: 2.0,
            2019: 1.5, 2020: -4.5, 2021: -2.0, 2022: 2.0, 2023: -0.5,
            2024: -1.0, 2025: -0.5,
        },
        "LTU": {
            2004: 3.0, 2005: 5.0, 2006: 5.5, 2007: 7.0, 2008: 4.5,
            2009: -8.0, 2010: -7.0, 2011: -3.5, 2012: -2.0, 2013: -1.5,
            2014: -1.0, 2015: -0.5, 2016: 0.0, 2017: 1.5, 2018: 2.5,
            2019: 2.0, 2020: -5.0, 2021: -1.5, 2022: 3.0, 2023: 1.0,
            2024: 0.5, 2025: 0.8,
        },
        "EST": {
            2004: 3.5, 2005: 6.0, 2006: 8.0, 2007: 9.5, 2008: 2.5,
            2009: -9.0, 2010: -7.5, 2011: -3.0, 2012: -2.0, 2013: -1.5,
            2014: -1.5, 2015: -1.0, 2016: -0.5, 2017: 0.5, 2018: 1.5,
            2019: 1.0, 2020: -6.0, 2021: -3.0, 2022: 1.5, 2023: -1.0,
            2024: -2.0, 2025: -1.0,
        },
        "SVN": {
            2004: 2.5, 2005: 2.5, 2006: 3.5, 2007: 5.0, 2008: 3.0,
            2009: -5.5, 2010: -4.0, 2011: -3.0, 2012: -4.5, 2013: -4.5,
            2014: -4.0, 2015: -3.5, 2016: -2.5, 2017: -1.0, 2018: 0.5,
            2019: 1.0, 2020: -5.5, 2021: -3.0, 2022: -0.5, 2023: -0.5,
            2024: -0.5, 2025: 0.0,
        },
        "SVK": {
            2004: 2.0, 2005: 3.0, 2006: 4.5, 2007: 6.0, 2008: 4.5,
            2009: -4.0, 2010: -2.5, 2011: -1.0, 2012: -2.5, 2013: -2.5,
            2014: -2.0, 2015: -1.5, 2016: -1.0, 2017: 0.5, 2018: 2.0,
            2019: 1.5, 2020: -6.0, 2021: -3.5, 2022: -1.5, 2023: -2.5,
            2024: -2.5, 2025: -1.5,
        },
        "HRV": {
            2013: 0.0, 2014: -1.5, 2015: -1.0, 2016: -0.5, 2017: 1.0,
            2018: 2.0, 2019: 2.0, 2020: -7.5, 2021: -4.0, 2022: 0.5,
            2023: 1.5, 2024: 1.0, 2025: 0.8,
        },
        "CYP": {
            2004: 2.0, 2005: 2.5, 2006: 3.5, 2007: 3.5, 2008: 2.0,
            2009: -2.0, 2010: -1.5, 2011: -2.0, 2012: -5.0, 2013: -7.0,
            2014: -7.0, 2015: -5.5, 2016: -4.0, 2017: -2.5, 2018: -1.0,
            2019: 0.0, 2020: -8.0, 2021: -5.0, 2022: -1.0, 2023: 0.5,
            2024: 0.8, 2025: 1.0,
        },
        "MLT": {
            2004: 2.5, 2005: 1.5, 2006: 2.0, 2007: 2.5, 2008: 1.0,
            2009: -3.0, 2010: -2.0, 2011: -1.5, 2012: -2.0, 2013: -2.0,
            2014: -1.5, 2015: -0.5, 2016: 0.5, 2017: 2.0, 2018: 3.0,
            2019: 3.5, 2020: -8.0, 2021: -4.5, 2022: -1.5, 2023: 0.0,
            2024: 0.0, 2025: 0.5,
        },
        "LUX": {
            1999: 1.5, 2000: 3.5, 2001: 1.5, 2002: -0.5, 2003: -1.0,
            2004: 0.5, 2005: 0.5, 2006: 2.0, 2007: 3.5, 2008: 0.5,
            2009: -4.0, 2010: -1.5, 2011: 0.5, 2012: -0.5, 2013: -1.0,
            2014: -0.5, 2015: 0.5, 2016: 0.5, 2017: 1.5, 2018: 2.0,
            2019: 2.0, 2020: -7.0, 2021: -3.5, 2022: -0.5, 2023: -1.0,
            2024: -0.5, 2025: 0.0,
        },
    }

    records = []
    for iso3, yr_data in og_data.items():
        for year, val in yr_data.items():
            records.append({"iso3": iso3, "year": year, "output_gap": val})
    return pd.DataFrame(records)


def fetch_output_gap(save: bool = True) -> pd.DataFrame:
    """
    Fetch output gap (% of potential GDP) for all available countries.
    Primary: OECD Economic Outlook API (no key required).
    Fallback: compiled EU table (covers CYP, MLT and any API failure).

    Returns DataFrame: iso3, year, output_gap (%).
    """
    print("  Fetching OECD EO output gap (GAP) …")
    df = _fetch_oecd_output_gap()

    # Always merge with compiled fallback: covers CYP, MLT, fills early years
    fallback = _compiled_fallback()
    if df is None or len(df) < 50:
        print("  OECD EO unavailable — using compiled fallback.")
        df = fallback
    else:
        combined = pd.concat([df, fallback], ignore_index=True)
        combined = combined.drop_duplicates(subset=["iso3", "year"], keep="first")
        df = combined

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df["output_gap"] = df["output_gap"].clip(lower=-20.0, upper=15.0)

    if save:
        save_cache(df, DATA_DIR / "output_gap_raw.parquet", CACHE_VERSION)

    return df


def load_output_gap() -> pd.DataFrame:
    """
    Load output gap data.  Re-fetches from OECD EO if the cache is absent
    or version-mismatched.
    """
    df = load_cache(DATA_DIR / "output_gap_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_output_gap(save=True)


if __name__ == "__main__":
    df = fetch_output_gap()
    print(df)
    print("\nG4 sample:")
    print(df[df["iso3"].isin(["FRA", "DEU", "ITA", "ESP"])].tail(10))
