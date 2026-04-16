"""
Eurostat HICP — dependent variable.

Source: Eurostat SDMX 3.0 API, dataset prc_hicp_minr (monthly % change vs same
month previous year, unit=RCH_A, coicop18=TOTAL). All monthly observations
for a given country-year are averaged, so partial years (e.g. 2026 with only
Jan-Apr available) still produce a usable annual estimate. Response format:
csvdata / formatVersion 2.0.

  https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/
  prc_hicp_minr/1.0/*.*.*.*?c[unit]=RCH_A&c[coicop18]=TOTAL&format=csvdata...

Returns [iso3, year, month, hicp] for EU member states + UK + Norway.
Saves to data/hicp_raw.parquet.
"""

import gzip
import requests
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path
from data._cache import load_cache, save_cache

CACHE_VERSION = "v2"

DATA_DIR = Path(__file__).resolve().parent.parent / "files"

# Eurostat SDMX 3.0 — monthly HICP rates of change (year-on-year), csvdata format
EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/"
    "prc_hicp_minr/1.0/*.*.*.*"
)

# Eurostat geo codes → ISO3
# Note: Eurostat uses EL for Greece and UK for the United Kingdom
GEO_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DK": "DNK", "EE": "EST", "ES": "ESP", "FI": "FIN",
    "FR": "FRA", "EL": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT",
    "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK",
    # Non-EU included countries
    "UK": "GBR", "NO": "NOR", "IS": "ISL",
}


def _fetch_eurostat_hicp() -> pd.DataFrame | None:
    """
    Download monthly HICP year-on-year rates (TOTAL, RCH_A) from Eurostat
    SDMX 3.0 API (csvdata / formatVersion 2.0) and keep native monthly
    frequency.

    Returns DataFrame with [iso3, year, month, hicp] or None on failure.
    """
    geo_list = [g for g in GEO_TO_ISO3 if g not in ("EA",)]
    params = {
        "c[freq]": "M",
        "c[unit]": "RCH_A",
        "c[coicop18]": "TOTAL",
        "c[geo]": ",".join(sorted(geo_list)),
        "startPeriod": "1996-01",
        "compress": "true",
        "format": "csvdata",
        "formatVersion": "2.0",
        "lang": "en",
    }
    try:
        resp = requests.get(EUROSTAT_URL, params=params, timeout=180)
        if resp.status_code != 200:
            print(f"  Eurostat HICP returned HTTP {resp.status_code}")
            return None

        content = resp.content
        try:
            content = gzip.decompress(content)
        except Exception:
            pass  # already decompressed by requests or not gzip

        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as exc:
        print(f"  Eurostat HICP fetch error: {exc}")
        return None

    if df.empty:
        return None

    # Normalise column names: strip whitespace, uppercase, drop '\TIME_PERIOD' suffix
    df.columns = [c.strip().upper().split("\\")[0] for c in df.columns]

    time_col = next((c for c in df.columns if c == "TIME_PERIOD"), None)
    geo_col  = next((c for c in df.columns if c == "GEO"), None)
    val_col  = next((c for c in df.columns if c == "OBS_VALUE"), None)

    if not all([time_col, geo_col, val_col]):
        print(f"  Unexpected CSV columns: {df.columns.tolist()}")
        return None

    df["iso3"]  = df[geo_col].map(GEO_TO_ISO3)
    df["year"]  = pd.to_numeric(df[time_col].str[:4], errors="coerce")
    df["month"] = pd.to_numeric(df[time_col].str[5:7], errors="coerce")
    df["hicp"] = pd.to_numeric(df[val_col], errors="coerce")

    df = df.dropna(subset=["iso3", "year", "month", "hicp"])
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    return (
        df[["iso3", "year", "month", "hicp"]]
        .sort_values(["iso3", "year", "month"])
        .reset_index(drop=True)
    )


def _compiled_fallback() -> pd.DataFrame:
    """
    Compiled HICP annual % change for EU G4 + key EU states.
    Used only when the Eurostat API is unavailable.
    Source: Eurostat / ECB publications.
    """
    hicp_data = {
        "FRA": {
            1999: 0.6, 2000: 1.8, 2001: 1.8, 2002: 1.9, 2003: 2.2,
            2004: 2.3, 2005: 1.9, 2006: 1.9, 2007: 1.6, 2008: 3.2,
            2009: 0.1, 2010: 1.7, 2011: 2.3, 2012: 2.2, 2013: 1.0,
            2014: 0.6, 2015: 0.1, 2016: 0.3, 2017: 1.2, 2018: 2.1,
            2019: 1.3, 2020: 0.5, 2021: 2.1, 2022: 5.9, 2023: 5.7,
            2024: 2.6, 2025: 1.4,
        },
        "DEU": {
            1999: 0.6, 2000: 1.4, 2001: 1.9, 2002: 1.4, 2003: 1.0,
            2004: 1.8, 2005: 1.9, 2006: 1.8, 2007: 2.3, 2008: 2.8,
            2009: 0.2, 2010: 1.2, 2011: 2.5, 2012: 2.1, 2013: 1.6,
            2014: 0.8, 2015: 0.7, 2016: 0.4, 2017: 1.7, 2018: 1.9,
            2019: 1.4, 2020: 0.4, 2021: 3.2, 2022: 8.7, 2023: 6.0,
            2024: 2.5, 2025: 2.2,
        },
        "ITA": {
            1999: 1.7, 2000: 2.6, 2001: 2.3, 2002: 2.6, 2003: 2.8,
            2004: 2.3, 2005: 2.2, 2006: 2.2, 2007: 2.0, 2008: 3.5,
            2009: 0.8, 2010: 1.6, 2011: 2.9, 2012: 3.3, 2013: 1.2,
            2014: 0.2, 2015: 0.1, 2016: -0.1, 2017: 1.3, 2018: 1.3,
            2019: 0.6, 2020: -0.1, 2021: 1.9, 2022: 8.7, 2023: 5.9,
            2024: 1.0, 2025: 1.5,
        },
        "ESP": {
            1999: 2.2, 2000: 3.5, 2001: 2.8, 2002: 3.6, 2003: 3.1,
            2004: 3.1, 2005: 3.4, 2006: 3.6, 2007: 2.8, 2008: 4.1,
            2009: -0.2, 2010: 2.0, 2011: 3.1, 2012: 2.4, 2013: 1.5,
            2014: -0.2, 2015: -0.6, 2016: -0.3, 2017: 2.0, 2018: 1.7,
            2019: 0.8, 2020: -0.3, 2021: 3.1, 2022: 8.4, 2023: 3.4,
            2024: 2.9, 2025: 2.5,
        },
        "GRC": {
            1999: 2.1, 2000: 2.9, 2001: 3.7, 2002: 3.9, 2003: 3.4,
            2004: 3.0, 2005: 3.5, 2006: 3.3, 2007: 3.0, 2008: 4.2,
            2009: 1.3, 2010: 4.7, 2011: 3.1, 2012: 1.0, 2013: -0.9,
            2014: -1.4, 2015: -1.1, 2016: 0.0, 2017: 1.1, 2018: 0.8,
            2019: 0.5, 2020: -1.3, 2021: 0.6, 2022: 9.3, 2023: 4.2,
            2024: 3.0, 2025: 2.8,
        },
        "PRT": {
            1999: 2.2, 2000: 2.9, 2001: 4.4, 2002: 3.6, 2003: 3.3,
            2004: 2.4, 2005: 2.1, 2006: 3.0, 2007: 2.4, 2008: 2.7,
            2009: -0.9, 2010: 1.4, 2011: 3.6, 2012: 2.8, 2013: 0.4,
            2014: -0.2, 2015: 0.5, 2016: 0.6, 2017: 1.6, 2018: 1.2,
            2019: 0.3, 2020: -0.1, 2021: 0.9, 2022: 8.1, 2023: 4.3,
            2024: 2.4, 2025: 2.0,
        },
        "NLD": {
            1999: 2.0, 2000: 2.3, 2001: 5.1, 2002: 3.9, 2003: 2.2,
            2004: 1.4, 2005: 1.5, 2006: 1.7, 2007: 1.6, 2008: 2.2,
            2009: 1.0, 2010: 0.9, 2011: 2.5, 2012: 2.8, 2013: 2.6,
            2014: 0.3, 2015: 0.2, 2016: 0.1, 2017: 1.3, 2018: 1.6,
            2019: 2.7, 2020: 1.1, 2021: 2.7, 2022: 11.6, 2023: 4.1,
            2024: 2.9, 2025: 2.5,
        },
        "BEL": {
            1999: 1.1, 2000: 2.7, 2001: 2.4, 2002: 1.6, 2003: 1.5,
            2004: 1.9, 2005: 2.5, 2006: 2.3, 2007: 1.8, 2008: 4.5,
            2009: 0.0, 2010: 2.3, 2011: 3.4, 2012: 2.6, 2013: 1.2,
            2014: 0.5, 2015: 0.6, 2016: 1.8, 2017: 2.2, 2018: 2.3,
            2019: 1.2, 2020: 0.4, 2021: 3.2, 2022: 10.3, 2023: 2.3,
            2024: 4.5, 2025: 3.0,
        },
        "AUT": {
            1999: 0.5, 2000: 2.0, 2001: 2.3, 2002: 1.7, 2003: 1.3,
            2004: 2.0, 2005: 2.1, 2006: 1.7, 2007: 2.2, 2008: 3.2,
            2009: 0.4, 2010: 1.7, 2011: 3.6, 2012: 2.6, 2013: 2.1,
            2014: 1.5, 2015: 0.8, 2016: 1.0, 2017: 2.2, 2018: 2.1,
            2019: 1.5, 2020: 1.4, 2021: 2.8, 2022: 8.6, 2023: 7.7,
            2024: 3.0, 2025: 2.5,
        },
        "FIN": {
            1999: 1.3, 2000: 3.0, 2001: 2.7, 2002: 2.0, 2003: 1.3,
            2004: 0.1, 2005: 0.8, 2006: 1.3, 2007: 1.6, 2008: 3.9,
            2009: 1.6, 2010: 1.7, 2011: 3.3, 2012: 3.2, 2013: 2.2,
            2014: 1.2, 2015: -0.2, 2016: 0.4, 2017: 0.8, 2018: 1.2,
            2019: 1.1, 2020: 0.4, 2021: 2.1, 2022: 7.2, 2023: 4.3,
            2024: 1.5, 2025: 1.5,
        },
        "IRL": {
            1999: 2.5, 2000: 5.3, 2001: 4.0, 2002: 4.7, 2003: 4.0,
            2004: 2.3, 2005: 2.2, 2006: 2.7, 2007: 2.9, 2008: 3.1,
            2009: -1.7, 2010: -1.6, 2011: 1.2, 2012: 1.9, 2013: 0.5,
            2014: 0.3, 2015: 0.0, 2016: -0.2, 2017: 0.3, 2018: 0.7,
            2019: 1.3, 2020: -0.5, 2021: 2.4, 2022: 8.2, 2023: 5.2,
            2024: 1.6, 2025: 1.8,
        },
        "LUX": {
            1999: 1.0, 2000: 3.8, 2001: 2.4, 2002: 2.1, 2003: 2.5,
            2004: 3.2, 2005: 3.8, 2006: 3.0, 2007: 2.7, 2008: 4.1,
            2009: 0.0, 2010: 2.8, 2011: 3.7, 2012: 2.9, 2013: 1.7,
            2014: 0.7, 2015: 0.1, 2016: 0.0, 2017: 2.1, 2018: 2.0,
            2019: 1.6, 2020: 0.0, 2021: 3.5, 2022: 8.3, 2023: 2.9,
            2024: 2.4, 2025: 2.2,
        },
        "SVN": {
            1999: 6.1, 2000: 8.9, 2001: 8.6, 2002: 7.5, 2003: 5.7,
            2004: 3.7, 2005: 2.5, 2006: 2.5, 2007: 3.8, 2008: 5.5,
            2009: 0.9, 2010: 2.1, 2011: 2.1, 2012: 2.8, 2013: 1.9,
            2014: 0.4, 2015: -0.8, 2016: -0.2, 2017: 1.6, 2018: 1.9,
            2019: 1.7, 2020: -0.1, 2021: 2.0, 2022: 9.3, 2023: 7.4,
            2024: 2.5, 2025: 2.2,
        },
        "SVK": {
            1999: 10.4, 2000: 12.2, 2001: 7.3, 2002: 3.5, 2003: 8.4,
            2004: 7.5, 2005: 2.8, 2006: 4.3, 2007: 1.9, 2008: 3.9,
            2009: 0.9, 2010: 0.7, 2011: 4.1, 2012: 3.7, 2013: 1.5,
            2014: -0.1, 2015: -0.3, 2016: -0.5, 2017: 1.4, 2018: 2.5,
            2019: 2.8, 2020: 2.0, 2021: 2.8, 2022: 12.1, 2023: 10.5,
            2024: 3.2, 2025: 3.0,
        },
        "EST": {
            2004: 3.0, 2005: 4.1, 2006: 4.4, 2007: 6.7, 2008: 10.6,
            2009: 0.2, 2010: 2.7, 2011: 5.1, 2012: 4.2, 2013: 3.2,
            2014: 0.5, 2015: 0.1, 2016: 0.8, 2017: 3.7, 2018: 3.4,
            2019: 2.3, 2020: -0.6, 2021: 4.5, 2022: 19.4, 2023: 9.2,
            2024: 3.4, 2025: 3.0,
        },
        "LVA": {
            2004: 6.2, 2005: 6.9, 2006: 6.6, 2007: 10.1, 2008: 15.3,
            2009: 3.3, 2010: -1.2, 2011: 4.2, 2012: 2.3, 2013: 0.0,
            2014: 0.7, 2015: 0.2, 2016: 0.1, 2017: 2.9, 2018: 2.6,
            2019: 2.7, 2020: 0.1, 2021: 3.2, 2022: 17.2, 2023: 8.9,
            2024: 2.2, 2025: 2.5,
        },
        "LTU": {
            2004: 1.2, 2005: 2.7, 2006: 3.8, 2007: 5.8, 2008: 11.1,
            2009: 4.2, 2010: 1.2, 2011: 4.1, 2012: 3.2, 2013: 1.2,
            2014: 0.2, 2015: -0.7, 2016: 0.7, 2017: 3.7, 2018: 2.5,
            2019: 2.2, 2020: 1.1, 2021: 3.8, 2022: 18.9, 2023: 8.7,
            2024: 0.8, 2025: 2.2,
        },
        "CZE": {
            2004: 2.6, 2005: 1.6, 2006: 2.1, 2007: 3.0, 2008: 6.3,
            2009: 0.6, 2010: 1.2, 2011: 2.1, 2012: 3.5, 2013: 1.4,
            2014: 0.4, 2015: 0.3, 2016: 0.6, 2017: 2.4, 2018: 2.0,
            2019: 2.6, 2020: 3.2, 2021: 3.8, 2022: 15.1, 2023: 10.7,
            2024: 2.6, 2025: 2.5,
        },
        "HUN": {
            2004: 6.8, 2005: 3.5, 2006: 4.0, 2007: 7.9, 2008: 6.0,
            2009: 4.0, 2010: 4.7, 2011: 3.9, 2012: 5.7, 2013: 1.7,
            2014: 0.0, 2015: 0.1, 2016: 0.4, 2017: 2.4, 2018: 2.9,
            2019: 3.4, 2020: 3.4, 2021: 5.1, 2022: 15.3, 2023: 17.6,
            2024: 3.7, 2025: 4.5,
        },
        "POL": {
            2004: 3.4, 2005: 2.2, 2006: 1.3, 2007: 2.6, 2008: 4.2,
            2009: 4.0, 2010: 2.6, 2011: 3.9, 2012: 3.7, 2013: 0.8,
            2014: 0.1, 2015: -0.7, 2016: -0.2, 2017: 2.0, 2018: 1.2,
            2019: 2.1, 2020: 3.7, 2021: 5.1, 2022: 13.2, 2023: 11.6,
            2024: 3.6, 2025: 4.2,
        },
        "ROU": {
            2004: 11.9, 2005: 9.0, 2006: 6.6, 2007: 4.9, 2008: 7.9,
            2009: 5.6, 2010: 6.1, 2011: 5.8, 2012: 3.4, 2013: 3.2,
            2014: 1.4, 2015: -0.4, 2016: -1.1, 2017: 1.3, 2018: 4.6,
            2019: 3.9, 2020: 2.3, 2021: 5.0, 2022: 13.8, 2023: 9.7,
            2024: 5.8, 2025: 5.5,
        },
        "BGR": {
            2004: 5.7, 2005: 5.0, 2006: 7.3, 2007: 7.6, 2008: 12.0,
            2009: 2.5, 2010: 3.0, 2011: 3.4, 2012: 2.4, 2013: 0.4,
            2014: -1.6, 2015: -1.1, 2016: -1.3, 2017: 1.2, 2018: 2.6,
            2019: 2.5, 2020: 1.7, 2021: 3.3, 2022: 13.0, 2023: 8.6,
            2024: 3.0, 2025: 3.5,
        },
        "HRV": {
            2013: 2.2, 2014: 0.2, 2015: -0.5, 2016: -1.1, 2017: 1.3,
            2018: 1.5, 2019: 0.8, 2020: 0.0, 2021: 2.7, 2022: 10.7,
            2023: 7.9, 2024: 4.0, 2025: 3.5,
        },
        "CYP": {
            2004: 1.9, 2005: 2.0, 2006: 2.2, 2007: 2.2, 2008: 4.4,
            2009: 0.2, 2010: 2.6, 2011: 3.5, 2012: 3.1, 2013: 0.4,
            2014: -0.3, 2015: -1.5, 2016: -1.2, 2017: 0.7, 2018: 0.8,
            2019: 0.5, 2020: -1.1, 2021: 2.3, 2022: 8.1, 2023: 3.9,
            2024: 2.3, 2025: 2.0,
        },
        "MLT": {
            2004: 2.7, 2005: 2.5, 2006: 2.6, 2007: 0.7, 2008: 4.7,
            2009: 1.8, 2010: 2.0, 2011: 2.5, 2012: 3.2, 2013: 1.0,
            2014: 0.8, 2015: 1.2, 2016: 0.9, 2017: 1.3, 2018: 1.2,
            2019: 1.5, 2020: 0.8, 2021: 0.7, 2022: 6.2, 2023: 5.1,
            2024: 2.4, 2025: 2.5,
        },
        "DNK": {
            1999: 2.1, 2000: 2.9, 2001: 2.4, 2002: 2.4, 2003: 2.0,
            2004: 0.9, 2005: 1.7, 2006: 1.9, 2007: 1.7, 2008: 3.6,
            2009: 1.1, 2010: 2.2, 2011: 2.7, 2012: 2.4, 2013: 0.8,
            2014: 0.6, 2015: 0.2, 2016: 0.3, 2017: 1.1, 2018: 0.7,
            2019: 0.7, 2020: 0.4, 2021: 2.3, 2022: 8.5, 2023: 3.4,
            2024: 2.3, 2025: 2.5,
        },
        "SWE": {
            1999: 0.5, 2000: 1.3, 2001: 2.7, 2002: 2.0, 2003: 1.9,
            2004: 0.4, 2005: 0.5, 2006: 1.5, 2007: 1.7, 2008: 3.3,
            2009: 1.9, 2010: 1.9, 2011: 1.4, 2012: 0.9, 2013: 0.4,
            2014: 0.2, 2015: 0.7, 2016: 1.1, 2017: 1.9, 2018: 2.0,
            2019: 1.7, 2020: 0.5, 2021: 2.7, 2022: 8.4, 2023: 5.9,
            2024: 2.2, 2025: 1.8,
        },
        "GBR": {
            1999: 1.4, 2000: 0.8, 2001: 1.2, 2002: 1.3, 2003: 1.4,
            2004: 1.3, 2005: 2.1, 2006: 2.3, 2007: 2.3, 2008: 3.6,
            2009: 2.1, 2010: 3.3, 2011: 4.5, 2012: 2.8, 2013: 2.6,
            2014: 1.5, 2015: 0.0, 2016: 0.7, 2017: 2.7, 2018: 2.5,
            2019: 1.8, 2020: 0.9, 2021: 2.6, 2022: 9.1, 2023: 7.3,
            2024: 2.5, 2025: 2.8,
        },
        "NOR": {
            1999: 2.3, 2000: 3.1, 2001: 3.0, 2002: 1.3, 2003: 2.5,
            2004: 0.4, 2005: 1.5, 2006: 2.3, 2007: 0.8, 2008: 3.8,
            2009: 2.3, 2010: 2.4, 2011: 1.3, 2012: 0.7, 2013: 2.1,
            2014: 2.0, 2015: 2.2, 2016: 3.6, 2017: 1.9, 2018: 2.7,
            2019: 2.3, 2020: 1.3, 2021: 3.5, 2022: 5.8, 2023: 5.5,
            2024: 3.0, 2025: 2.5,
        },
    }

    records = []
    for iso3, yr_data in hicp_data.items():
        for year, hicp in yr_data.items():
            records.append({"iso3": iso3, "year": year, "hicp": hicp})
    return pd.DataFrame(records)


def fetch_hicp(save: bool = True) -> pd.DataFrame:
    """
    Fetch monthly HICP % change for EU + UK + Norway.
    Primary: Eurostat REST API.  Fallback: compiled historical table.

    Returns DataFrame: iso3, year, month, hicp (percentage points).
    """
    print("  Fetching Eurostat HICP monthly % change (prc_hicp_minr, RCH_A) …")
    df = _fetch_eurostat_hicp()

    if df is None or len(df) < 100:
        print("  Eurostat API unavailable or insufficient data — using compiled fallback.")
        df = _compiled_fallback()
        # Fallback table is annual; encode it as December observations.
        df["month"] = 12
    else:
        print(f"  Eurostat HICP: {len(df)} rows, {df['iso3'].nunique()} countries")

    # Keep only geo codes with valid ISO3 mappings
    df = df[df["iso3"].notna()].copy()
    df = df.sort_values(["iso3", "year", "month"]).reset_index(drop=True)

    # Winsorize extreme outliers (hyperinflation countries excluded from panel anyway)
    df["hicp"] = df["hicp"].clip(lower=-5.0, upper=50.0)

    if save:
        save_cache(df, DATA_DIR / "hicp_raw.parquet", CACHE_VERSION)

    return df


def load_hicp() -> pd.DataFrame:
    """Load cached HICP data; re-fetch if not present or version mismatch."""
    df = load_cache(DATA_DIR / "hicp_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_hicp(save=True)


if __name__ == "__main__":
    df = fetch_hicp()
    print(df)
    print("\nG4 sample:")
    print(df[df["iso3"].isin(["FRA", "DEU", "ITA", "ESP"])].tail(12))
