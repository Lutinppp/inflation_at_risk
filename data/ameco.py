"""
Inflation expectations for EU panel countries.

Primary: OECD Economic Outlook SDMX-JSON API, measure CPIH_YTYPCT (HICP
y/y % change, annual).  Covers 27 of 29 EU panel countries (CYP and MLT
are absent from OECD EO).  No API key required.
  https://stats.oecd.org/SDMX-JSON/data/EO/{countries}.CPIH_YTYPCT.A/all

Override: ECB Consumer Expectations Survey (CES) — 12-month ahead mean,
monthly averaged to annual.  Covers DE, FR, IT, ES from 2020.
  https://data-api.ecb.europa.eu/service/data/ECB_CES1/

Fallback: compiled table for CYP and MLT (1999–2025).

Returns [iso3, year, infl_expectations].
Saves to data/ameco_raw.parquet.
"""

import io
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from data._cache import load_cache, save_cache

CACHE_VERSION = "v1"

DATA_DIR = Path(__file__).parent

# ── OECD Economic Outlook — HICP inflation proxy ────────────────────────────
_OECD_EO_URL     = "https://stats.oecd.org/SDMX-JSON/data/EO"
_OECD_MEASURE    = "CPIH_YTYPCT"   # HICP y/y % change (annual), forecast + actuals
_OECD_FREQ       = "A"

# All EU panel countries available in OECD EO (CYP and MLT are absent)
_OECD_COUNTRIES = [
    "FRA", "DEU", "ITA", "ESP", "GRC", "PRT", "NLD", "BEL", "AUT",
    "FIN", "IRL", "LUX", "SVN", "SVK", "EST", "LVA", "LTU", "CZE",
    "HUN", "POL", "ROU", "BGR", "HRV", "DNK", "SWE", "GBR", "NOR",
]

# G4 country → AMECO country suffix (kept for legacy / bulk-parse fallback)
AMECO_CODES = {
    "FRA": "AYFRF",
    "DEU": "AYDBF",
    "ITA": "AYITF",
    "ESP": "AYESF",
}

AMECO_VAR = "ZCPIH"  # Price deflator of private final consumption exp.
AMECO_VAR_ALT = "ZCPIN"  # Alternative: private consumption deflator

AMECO_REST_URL = (
    "https://ec.europa.eu/economy_finance/ameco/user/serie/ResultSerie.cfm"
)

AMECO_BULK_URL = (
    "https://ec.europa.eu/economy_finance/ameco/user/serie/SelectSerie.cfm"
)

ISO3_TO_NAME = {
    "AUT": "Austria",    "BEL": "Belgium",    "BGR": "Bulgaria",
    "CYP": "Cyprus",     "CZE": "Czechia",    "DEU": "Germany",
    "DNK": "Denmark",    "ESP": "Spain",      "EST": "Estonia",
    "FIN": "Finland",    "FRA": "France",     "GBR": "United Kingdom",
    "GRC": "Greece",     "HRV": "Croatia",    "HUN": "Hungary",
    "IRL": "Ireland",    "ITA": "Italy",      "LTU": "Lithuania",
    "LUX": "Luxembourg", "LVA": "Latvia",     "MLT": "Malta",
    "NLD": "Netherlands","NOR": "Norway",     "POL": "Poland",
    "PRT": "Portugal",   "ROU": "Romania",    "SVK": "Slovakia",
    "SVN": "Slovenia",   "SWE": "Sweden",
}


def _build_ameco_series_codes(var: str = AMECO_VAR) -> str:
    """Build the series codes query string for AMECO REST API."""
    # Format: ZCPIH.1.0.0.0.AYFRF+ZCPIH.1.0.0.0.AYDBF+...
    codes = "+".join(f"{var}.1.0.0.0.{suffix}" for suffix in AMECO_CODES.values())
    return codes


# ECB CES: 12-month ahead mean inflation expectation
# Dimensions (order): FREQ . REF_AREA . CES_BREAKDOWN . CES_CUSTOM . CES_VARIABLE . CES_ANSWER . CES_DENOM
# Try multiple likely variable/denom codes since the exact codes vary across CES releases.
_CES_SERIES_CANDIDATES = [
    "M.DE+FR+IT+ES.TOT.A.INFL_NEXT12M.MEAN.PC",
    "M.DE+FR+IT+ES.TOT.A.INFL_NEXT12M.MEAN.PCT",
    "M.DE+FR+IT+ES.TOT.N.INFL_NEXT12M.MEAN.PC",
    "M.DE+FR+IT+ES.TOT.A.IP12M.MEAN.PC",
]
_ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data/ECB_CES1"

# ECB 2-letter → ISO3
_ECB_ISO2_TO_ISO3 = {"DE": "DEU", "FR": "FRA", "IT": "ITA", "ES": "ESP"}


def _fetch_oecd_cpih() -> pd.DataFrame | None:
    """
    Fetch HICP y/y % change from the OECD Economic Outlook.
    Uses CPIH_YTYPCT (harmonized CPI) where available; falls back to
    CPI_YTYPCT (national CPI) for countries that lack the harmonized series.
    No API key required.  Covers all 27 OECD EU panel countries.

    Returns long DataFrame [iso3, year, infl_expectations] or None on failure.
    """
    countries_str = ",".join(_OECD_COUNTRIES)

    def _fetch_measure(measure: str) -> pd.DataFrame | None:
        url = f"{_OECD_EO_URL}/{countries_str}.{measure}.{_OECD_FREQ}/all"
        try:
            resp = requests.get(url, params={"startTime": "1990", "endTime": "2027"}, timeout=60)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception as exc:
            print(f"  OECD EO {measure} fetch error: {exc}")
            return None

        try:
            structures   = data["data"]["structures"][0]
            series_dims  = structures["dimensions"]["series"]
            obs_dims     = structures["dimensions"]["observation"]

            measure_dim  = next(d for d in series_dims if d["id"] == "MEASURE")
            freq_dim     = next(d for d in series_dims if d["id"] == "FREQ")
            ref_area_dim = next(d for d in series_dims if d["id"] == "REF_AREA")
            time_dim     = obs_dims[0]

            meas_idx = next(str(i) for i, v in enumerate(measure_dim["values"])
                            if v["id"] == measure)
            a_idx    = next(str(i) for i, v in enumerate(freq_dim["values"])
                            if v["id"] == _OECD_FREQ)
            iso3_map = {str(i): v["id"] for i, v in enumerate(ref_area_dim["values"])}
            year_map = {
                str(i): v["id"]
                for i, v in enumerate(time_dim["values"])
                if len(v["id"]) == 4
            }
        except (KeyError, StopIteration, IndexError) as exc:
            print(f"  OECD EO {measure}: unexpected response structure — {exc}")
            return None

        records = []
        for key, sdata in data["data"]["dataSets"][0]["series"].items():
            parts = key.split(":")
            if len(parts) != 3 or parts[1] != meas_idx or parts[2] != a_idx:
                continue
            iso3 = iso3_map.get(parts[0], "")
            if not iso3 or iso3 not in _OECD_COUNTRIES:
                continue
            for t_idx, vals in sdata.get("observations", {}).items():
                year_str = year_map.get(t_idx)
                if year_str and vals[0] is not None:
                    records.append({"iso3": iso3, "year": int(year_str),
                                     "infl_expectations": vals[0]})
        return pd.DataFrame(records) if records else None

    # Step 1: CPIH_YTYPCT (harmonized HICP) — available for euro area + some others
    df_hicp = _fetch_measure("CPIH_YTYPCT")
    if df_hicp is None or df_hicp.empty:
        print("  OECD EO CPIH_YTYPCT unavailable — trying CPI_YTYPCT fallback.")
        df_cpi = _fetch_measure("CPI_YTYPCT")
        if df_cpi is None or df_cpi.empty:
            return None
        df = df_cpi
    else:
        # Step 2: for countries missing from CPIH, use CPI_YTYPCT
        covered = set(df_hicp["iso3"].unique())
        missing = [c for c in _OECD_COUNTRIES if c not in covered]
        if missing:
            df_cpi = _fetch_measure("CPI_YTYPCT")
            if df_cpi is not None and not df_cpi.empty:
                filler = df_cpi[df_cpi["iso3"].isin(missing)]
                df = pd.concat([df_hicp, filler], ignore_index=True)
                print(f"  OECD EO CPIH_YTYPCT: {df_hicp['iso3'].nunique()} countries; "
                      f"CPI_YTYPCT supplement: {missing}")
            else:
                df = df_hicp
        else:
            df = df_hicp

    df = df.dropna(subset=["infl_expectations"])
    print(f"  OECD EO inflation: {len(df)} obs, {df['iso3'].nunique()} countries")
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)


def _fetch_ecb_ces() -> pd.DataFrame | None:
    """
    Fetch 12-month-ahead inflation expectations (mean) from the ECB Consumer
    Expectations Survey (CES).  Monthly observations are averaged within each
    calendar year so partial years are handled automatically.

    Returns long DataFrame [iso3, year, infl_expectations] or None on failure.
    CES data starts in 2020; for earlier years use AMECO fallback.
    """
    for key in _CES_SERIES_CANDIDATES:
        url = f"{_ECB_API_BASE}/{key}"
        try:
            resp = requests.get(
                url,
                params={"format": "csvdata", "startPeriod": "2020"},
                timeout=45,
            )
            if resp.status_code != 200:
                continue

            content = resp.text.strip()
            if not content or len(content) < 50:
                continue

            # ECB CSV format: KEY,FREQ,REF_AREA,...,TIME_PERIOD,OBS_VALUE,...
            df = pd.read_csv(io.StringIO(content))
            if df.empty or "OBS_VALUE" not in df.columns:
                continue

            # Identify country and time columns (case-insensitive)
            col_map = {c.upper(): c for c in df.columns}
            area_col = col_map.get("REF_AREA")
            time_col = col_map.get("TIME_PERIOD")
            if area_col is None or time_col is None:
                continue

            df = df[[area_col, time_col, "OBS_VALUE"]].copy()
            df.columns = ["ref_area", "time_period", "value"]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])

            # Extract year from YYYY-MM time period
            df["year"] = df["time_period"].str[:4]
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)

            # Map ECB country code → ISO3
            df["iso3"] = df["ref_area"].str.upper().map(_ECB_ISO2_TO_ISO3)
            df = df.dropna(subset=["iso3"])

            if df.empty:
                continue

            # Annual average of monthly survey values
            annual = (
                df.groupby(["iso3", "year"])["value"]
                .mean()
                .reset_index()
                .rename(columns={"value": "infl_expectations"})
            )
            print(f"  ECB CES loaded: {len(annual)} country-year obs (key: {key})")
            return annual.sort_values(["iso3", "year"]).reset_index(drop=True)

        except Exception as exc:
            print(f"  ECB CES attempt failed ({key}): {exc}")
            continue

    return None


def _fetch_ameco_rest(var: str = AMECO_VAR) -> pd.DataFrame | None:
    """
    Attempt AMECO REST API fetch for G4 inflation expectations.
    Returns long DataFrame or None on failure.
    """
    series_codes = _build_ameco_series_codes(var)
    params = {"ZCPIH": series_codes}  # query format may vary
    url = f"{AMECO_REST_URL}?{series_codes}"
    try:
        resp = requests.get(url, params={}, timeout=30)
        if resp.status_code != 200:
            print(f"  AMECO REST returned HTTP {resp.status_code}")
            return None

        # Try to parse as CSV or HTML table
        content = resp.text
        if "," in content[:200]:
            df = pd.read_csv(io.StringIO(content))
        else:
            # Try HTML tables
            tables = pd.read_html(io.StringIO(content))
            if not tables:
                return None
            df = tables[0]

        # Look for year columns and country identifiers
        if df.empty or len(df.columns) < 3:
            return None

        return _parse_ameco_wide(df)

    except Exception as exc:
        print(f"  AMECO REST fetch failed: {exc}")
        return None


def _fetch_ameco_bulk() -> pd.DataFrame | None:
    """
    Attempt AMECO bulk Excel download and parse ZCPIH for G4.
    Returns long DataFrame or None on failure.
    """
    bulk_urls = [
        "https://ec.europa.eu/economy_finance/ameco/user/serie/AMECO11.xls",
        "https://ec.europa.eu/economy_finance/ameco/user/serie/AMECO10.xls",
    ]
    for url in bulk_urls:
        try:
            print(f"  Trying AMECO bulk: {url}")
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                continue
            xl = pd.read_excel(io.BytesIO(resp.content), sheet_name=None, engine="xlrd")
            for sheet_name, df_raw in xl.items():
                if df_raw.empty:
                    continue
                # Look for rows containing our variable codes
                df_str = df_raw.astype(str)
                mask = df_str.apply(
                    lambda col: col.str.contains(AMECO_VAR, na=False)
                ).any(axis=1)
                relevant = df_raw[mask]
                if len(relevant) > 0:
                    return _parse_ameco_bulk_sheet(relevant)
        except Exception as exc:
            print(f"  AMECO bulk failed ({url}): {exc}")
            continue
    return None


def _parse_ameco_wide(df: pd.DataFrame) -> pd.DataFrame | None:
    """Parse a wide-format AMECO table into long format."""
    try:
        # Find year columns (4-digit numeric column names)
        year_cols = [c for c in df.columns if str(c).strip().isdigit()
                     and 1990 <= int(str(c).strip()) <= 2030]
        if not year_cols:
            return None

        # Find country identifier column
        id_col = df.columns[0]
        records = []
        for _, row in df.iterrows():
            row_id = str(row[id_col])
            iso3 = None
            for i3, suffix in AMECO_CODES.items():
                if suffix in row_id or ISO3_TO_NAME[i3].lower() in row_id.lower():
                    iso3 = i3
                    break
            if iso3 is None:
                continue
            for yc in year_cols:
                val = pd.to_numeric(row[yc], errors="coerce")
                if pd.notna(val):
                    records.append({"iso3": iso3, "year": int(str(yc).strip()),
                                    "infl_expectations": float(val)})
        if records:
            df_out = pd.DataFrame(records)
            # AMECO ZCPIH is an index (2005=100) or % change — check scale
            # If values > 20, likely index → compute % change
            if df_out["infl_expectations"].median() > 20:
                df_out = _convert_index_to_pct(df_out)
            return df_out
    except Exception as exc:
        print(f"  Wide parse failed: {exc}")
    return None


def _parse_ameco_bulk_sheet(df: pd.DataFrame) -> pd.DataFrame | None:
    """Parse a single AMECO bulk Excel sheet for G4 ZCPIH."""
    try:
        year_cols = [c for c in df.columns if str(c).strip().replace(".", "").isdigit()
                     and 1990 <= int(str(c).strip().split(".")[0]) <= 2030]
        records = []
        for _, row in df.iterrows():
            row_str = " ".join(str(v) for v in row.values)
            iso3 = None
            for i3, suffix in AMECO_CODES.items():
                if suffix in row_str:
                    iso3 = i3
                    break
            if iso3 is None:
                continue
            for yc in year_cols:
                val = pd.to_numeric(row[yc], errors="coerce")
                if pd.notna(val):
                    records.append({"iso3": iso3, "year": int(str(yc).strip()),
                                    "infl_expectations": float(val)})
        if records:
            df_out = pd.DataFrame(records)
            if df_out["infl_expectations"].median() > 20:
                df_out = _convert_index_to_pct(df_out)
            return df_out
    except Exception as exc:
        print(f"  Bulk sheet parse failed: {exc}")
    return None


def _convert_index_to_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Convert AMECO price index (2010=100 style) to annual % change."""
    dfs = []
    for iso3, grp in df.groupby("iso3"):
        grp = grp.sort_values("year").copy()
        grp["infl_expectations"] = grp["infl_expectations"].pct_change() * 100
        dfs.append(grp)
    return pd.concat(dfs, ignore_index=True).dropna(subset=["infl_expectations"])


def _compiled_fallback() -> pd.DataFrame:
    """
    Compiled HICP % change for G4 + CYP + MLT (countries not covered by OECD EO).
    G4 values: EC Economic Forecast publications + ECB March 2026 projections.
    CYP / MLT values: Eurostat HICP actuals + EC Spring 2025 forecast.
    Used only to supplement OECD EO for CYP and MLT.
    """
    ameco_data = {
        "FRA": {
            1999: 0.5, 2000: 1.5, 2001: 1.6, 2002: 1.7, 2003: 1.9,
            2004: 1.8, 2005: 1.5, 2006: 1.8, 2007: 1.3, 2008: 2.7,
            2009: 0.0, 2010: 1.5, 2011: 2.0, 2012: 2.0, 2013: 0.8,
            2014: 0.5, 2015: 0.1, 2016: 0.3, 2017: 1.1, 2018: 1.7,
            2019: 1.1, 2020: 0.4, 2021: 1.8, 2022: 5.4, 2023: 5.2,
            2024: 2.4, 2025: 1.5, 2026: 1.7, 2027: 2.0,
        },
        "DEU": {
            1999: 0.5, 2000: 1.2, 2001: 1.6, 2002: 1.1, 2003: 0.9,
            2004: 1.4, 2005: 1.5, 2006: 1.5, 2007: 2.0, 2008: 2.3,
            2009: 0.2, 2010: 1.0, 2011: 2.2, 2012: 1.7, 2013: 1.3,
            2014: 0.7, 2015: 0.6, 2016: 0.4, 2017: 1.5, 2018: 1.6,
            2019: 1.2, 2020: 0.3, 2021: 2.8, 2022: 8.0, 2023: 5.7,
            2024: 2.3, 2025: 2.4, 2026: 2.3, 2027: 2.2,
        },
        "ITA": {
            1999: 1.5, 2000: 2.2, 2001: 2.0, 2002: 2.3, 2003: 2.5,
            2004: 2.0, 2005: 1.8, 2006: 1.8, 2007: 1.6, 2008: 2.8,
            2009: 0.5, 2010: 1.2, 2011: 2.5, 2012: 2.8, 2013: 1.0,
            2014: 0.2, 2015: 0.1, 2016: -0.1, 2017: 1.0, 2018: 1.1,
            2019: 0.5, 2020: -0.1, 2021: 1.6, 2022: 7.9, 2023: 5.5,
            2024: 1.0, 2025: 1.4, 2026: 1.6, 2027: 1.9,
        },
        "ESP": {
            1999: 1.9, 2000: 3.0, 2001: 2.5, 2002: 3.2, 2003: 2.8,
            2004: 2.7, 2005: 3.0, 2006: 3.1, 2007: 2.4, 2008: 3.3,
            2009: -0.2, 2010: 1.6, 2011: 2.7, 2012: 2.0, 2013: 1.2,
            2014: -0.2, 2015: -0.5, 2016: -0.3, 2017: 1.7, 2018: 1.4,
            2019: 0.6, 2020: -0.2, 2021: 2.8, 2022: 7.7, 2023: 3.2,
            2024: 2.7, 2025: 2.2, 2026: 2.0, 2027: 2.1,
        },
        # Cyprus — Eurostat HICP actuals; joined euro area 2008
        "CYP": {
            1999: 1.1, 2000: 4.9, 2001: 2.0, 2002: 2.8, 2003: 4.0,
            2004: 1.9, 2005: 2.0, 2006: 2.2, 2007: 2.2, 2008: 4.4,
            2009: 0.2, 2010: 2.6, 2011: 3.5, 2012: 3.1, 2013: 0.4,
            2014: -0.3, 2015: -1.5, 2016: -1.2, 2017: 0.7, 2018: 0.8,
            2019: 0.5, 2020: -1.1, 2021: 2.3, 2022: 8.1, 2023: 3.9,
            2024: 2.2, 2025: 1.8, 2026: 2.0, 2027: 2.0,
        },
        # Malta — Eurostat HICP actuals; joined euro area 2008
        "MLT": {
            1999: 2.3, 2000: 3.0, 2001: 2.5, 2002: 2.2, 2003: 1.9,
            2004: 2.7, 2005: 2.5, 2006: 2.6, 2007: 0.7, 2008: 4.7,
            2009: 1.8, 2010: 2.0, 2011: 2.5, 2012: 3.2, 2013: 1.0,
            2014: 0.8, 2015: 1.2, 2016: 0.9, 2017: 1.3, 2018: 1.7,
            2019: 1.5, 2020: 0.8, 2021: 0.7, 2022: 6.1, 2023: 5.6,
            2024: 2.4, 2025: 2.0, 2026: 2.1, 2027: 2.1,
        },
    }

    records = []
    for iso3, yr_data in ameco_data.items():
        for year, val in yr_data.items():
            records.append({"iso3": iso3, "year": year, "infl_expectations": val})
    return pd.DataFrame(records)


def fetch_ameco(save: bool = True) -> pd.DataFrame:
    """
    Fetch inflation expectations for all EU panel countries.
    1. OECD EO CPIH_YTYPCT (1990–2027): primary source for 27/29 countries.
    2. ECB CES (2020+): overrides G4 (DE/FR/IT/ES) with survey expectations.
    3. Compiled fallback: supplements CYP and MLT not covered by OECD EO.

    Returns DataFrame: iso3, year, infl_expectations (% change p.a.).
    """
    print("  Fetching inflation expectations (OECD EO CPIH_YTYPCT, all EU panel) …")

    # ── Step 1: OECD EO as primary source for 27 countries ──────────────────
    base_df = _fetch_oecd_cpih()

    if base_df is None or base_df.empty:
        print("  OECD EO unavailable — falling back to compiled G4 table.")
        base_df = _compiled_fallback()
    else:
        # Supplement with compiled fallback for CYP and MLT (not in OECD EO)
        fallback = _compiled_fallback()
        cyp_mlt = fallback[fallback["iso3"].isin(["CYP", "MLT"])]
        if not cyp_mlt.empty:
            base_df = pd.concat([base_df, cyp_mlt], ignore_index=True) \
                        .drop_duplicates(subset=["iso3", "year"], keep="first")

    # ── Step 2: ECB CES overrides G4 for 2020+ ──────────────────────────────
    ces_df = _fetch_ecb_ces()
    if ces_df is not None and not ces_df.empty:
        combined = pd.concat([ces_df, base_df], ignore_index=True)
        df = combined.drop_duplicates(subset=["iso3", "year"], keep="first")
        print("  Blended ECB CES (2020+ G4) with OECD EO history.")
    else:
        print("  ECB CES unavailable — using OECD EO data only.")
        df = base_df

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df["infl_expectations"] = df["infl_expectations"].clip(lower=-5.0, upper=30.0)

    if save:
        save_cache(df, DATA_DIR / "ameco_raw.parquet", CACHE_VERSION)

    n_countries = df["iso3"].nunique()
    sample_2024 = df[df["year"] == 2024].set_index("iso3")["infl_expectations"]
    print(f"  Inflation expectations: {len(df)} obs, {n_countries} countries")
    print(f"  2024 sample: {sample_2024.to_dict()}")

    return df


def load_ameco() -> pd.DataFrame:
    """Load cached AMECO data; re-fetch if not present or version mismatch."""
    df = load_cache(DATA_DIR / "ameco_raw.parquet", CACHE_VERSION)
    if df is not None:
        return df
    return fetch_ameco(save=True)


if __name__ == "__main__":
    df = fetch_ameco()
    print(df)
