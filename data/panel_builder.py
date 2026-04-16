"""
Monthly panel builder for EU Inflation-at-Risk.

Builds a monthly panel from mixed-frequency source series:
- Monthly series are kept at native frequency.
- Quarterly series are anchored at quarter-end months (3/6/9/12) and then
  propagated to missing months within each year.
- Annual series are anchored at December and then propagated to all months
  within each year.

This follows the rule requested in the notebook workflow:
"for lower-frequency data, place values at period-end and use the same value
for missing months".

Saves: data/files/panel.parquet
Columns:
  iso3, year, month, hicp, hicp_lag,
  output_gap, infl_expectations, energy_price_chg,
  import_price_chg, clifs, spread_10y, wui,
  hicp_fwd1, hicp_fwd2, hicp_fwd4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "files"

# Countries intended for the estimation panel
EU_CORE = [
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA",
    "MLT", "NLD", "NOR", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
]


def _monthlyize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Convert a series to monthly frequency with period-end anchoring.

    Accepted input schemas:
    - monthly : iso3, year, month, value_col
    - quarterly: iso3, year, quarter, value_col
    - annual  : iso3, year, value_col

    Output schema:
    - iso3, year, month, value_col

    For lower-frequency inputs, values are placed at period end and then
    propagated to missing months within each year.
    """
    cols = set(df.columns)
    work = df.copy()

    if {"iso3", "year", "month", value_col}.issubset(cols):
        out = work[["iso3", "year", "month", value_col]].copy()
    elif {"iso3", "year", "quarter", value_col}.issubset(cols):
        out = work[["iso3", "year", "quarter", value_col]].copy()
        out["month"] = out["quarter"].astype(int) * 3
        out = out[["iso3", "year", "month", value_col]]
    elif {"iso3", "year", value_col}.issubset(cols):
        out = work[["iso3", "year", value_col]].copy()
        out["month"] = 12
        out = out[["iso3", "year", "month", value_col]]
    else:
        return pd.DataFrame(columns=["iso3", "year", "month", value_col])

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["iso3", "year", "month"]) 
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out = out[out["month"].between(1, 12)]

    # Collapse accidental duplicates at same monthly point
    out = (
        out.groupby(["iso3", "year", "month"], as_index=False)[value_col]
        .mean()
    )

    # Expand to full monthly grid per country-year
    years = out["year"].unique().tolist()
    countries = out["iso3"].unique().tolist()
    full = pd.MultiIndex.from_product(
        [countries, sorted(years), range(1, 13)],
        names=["iso3", "year", "month"],
    ).to_frame(index=False)

    out = full.merge(out, on=["iso3", "year", "month"], how="left")

    # Keep period-end convention but fill missing months with same value.
    # bfill first (to fill months before period-end anchor), then ffill
    # for trailing gaps in the same year.
    out[value_col] = (
        out.sort_values(["iso3", "year", "month"])
        .groupby(["iso3", "year"])[value_col]
        .transform(lambda s: s.bfill().ffill())
    )

    return out


def _has_wui_col(df: pd.DataFrame) -> bool:
    return "wui" in df.columns


def _make_wui_stub() -> pd.DataFrame:
    return pd.DataFrame(columns=["iso3", "year", "quarter", "wui"])


def build_panel(
    min_obs_years: int = 15,
    start_year: int = 1999,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    Build monthly estimation panel from all available data modules.

    Parameters
    ----------
    min_obs_years : minimum required years of non-missing HICP
    start_year    : first year to include
    end_year      : last year to include
    """
    from data.modules.hicp import load_hicp
    from data.modules.ameco import load_ameco
    from data.modules.output_gap import load_output_gap
    from data.modules.energy_prices import load_energy_prices
    from data.modules.import_prices import load_import_prices
    from data.modules.imf_fsi import load_fsi
    from data.modules.ecb_spreads import load_spreads
    from data.modules.wui import load_wui

    print("Building IaR monthly estimation panel ...")

    # Load source data
    hicp_raw = load_hicp()[["iso3", "year", "month", "hicp"]]
    ameco_raw = load_ameco()[["iso3", "year", "quarter", "infl_expectations"]]
    og_raw = load_output_gap()[["iso3", "year", "quarter", "output_gap"]]
    energy_raw = load_energy_prices()[["iso3", "year", "month", "energy_price_chg"]]
    import_raw = load_import_prices()[["iso3", "year", "quarter", "import_price_chg"]]
    fsi_raw = load_fsi()[["iso3", "year", "month", "clifs"]]
    spreads_raw = load_spreads()[["iso3", "year", "month", "spread_10y"]]
    wui_loaded = load_wui()
    wui_raw = (
        wui_loaded[["iso3", "year", "quarter", "wui"]]
        if _has_wui_col(wui_loaded)
        else _make_wui_stub()
    )

    # Convert everything to monthly with period-end anchoring + fill
    hicp = _monthlyize(hicp_raw, "hicp")
    ameco = _monthlyize(ameco_raw, "infl_expectations")
    og = _monthlyize(og_raw, "output_gap")
    energy = _monthlyize(energy_raw, "energy_price_chg")
    imports = _monthlyize(import_raw, "import_price_chg")
    fsi = _monthlyize(fsi_raw, "clifs")
    spreads = _monthlyize(spreads_raw, "spread_10y")
    wui = _monthlyize(wui_raw, "wui")

    # Base grid from HICP coverage
    panel = hicp.copy()

    # Merge monthlyized regressors
    panel = (
        panel
        .merge(og, on=["iso3", "year", "month"], how="left")
        .merge(ameco, on=["iso3", "year", "month"], how="left")
        .merge(energy, on=["iso3", "year", "month"], how="left")
        .merge(imports, on=["iso3", "year", "month"], how="left")
        .merge(fsi, on=["iso3", "year", "month"], how="left")
        .merge(spreads, on=["iso3", "year", "month"], how="left")
        .merge(wui, on=["iso3", "year", "month"], how="left")
    )

    panel = panel[
        (panel["year"] >= start_year) & (panel["year"] <= end_year)
    ].copy()

    # Keep EU_CORE countries and coverage threshold on monthly HICP
    hicp_counts = (
        panel.groupby("iso3")["hicp"]
        .apply(lambda x: x.notna().sum())
        .reset_index()
        .rename(columns={"hicp": "n_hicp"})
    )
    min_obs_months = int(min_obs_years * 12)
    good_countries = hicp_counts.loc[
        (hicp_counts["n_hicp"] >= min_obs_months)
        & (hicp_counts["iso3"].isin(EU_CORE)),
        "iso3",
    ]
    panel = panel[panel["iso3"].isin(good_countries)].copy()

    # Winsorization / robust clipping
    panel["hicp"] = panel["hicp"].clip(lower=-3.0, upper=25.0)
    panel["output_gap"] = panel["output_gap"].clip(lower=-15.0, upper=10.0)
    panel["infl_expectations"] = panel["infl_expectations"].fillna(panel["hicp"])
    panel["infl_expectations"] = panel["infl_expectations"].clip(lower=-2.0, upper=15.0)
    panel["energy_price_chg"] = panel["energy_price_chg"].clip(lower=-80.0, upper=120.0)
    panel["import_price_chg"] = panel["import_price_chg"].clip(lower=-40.0, upper=60.0)
    panel["spread_10y"] = panel["spread_10y"].clip(lower=-5.0, upper=25.0)
    panel["wui"] = panel["wui"].clip(lower=0.0, upper=2.0)

    panel = panel.sort_values(["iso3", "year", "month"]).reset_index(drop=True)

    # Lagged monthly HICP
    panel["hicp_lag"] = panel.groupby("iso3")["hicp"].shift(1)

    # Forward HICP means for horizons h in years over monthly data
    # h=1 -> next 12 months, h=2 -> next 24, h=4 -> next 48
    for h in [1, 2, 4]:
        window = 12 * h
        vals = []
        for iso3, grp in panel.groupby("iso3"):
            grp = grp.sort_values(["year", "month"]).copy()
            arr = grp["hicp"].values
            fwd = np.full(len(arr), np.nan)
            for i in range(len(arr)):
                win = arr[i + 1: i + 1 + window]
                if len(win) == window and not np.isnan(win).any():
                    fwd[i] = float(np.mean(win))
            grp[f"hicp_fwd{h}"] = fwd
            vals.append(grp[["iso3", "year", "month", f"hicp_fwd{h}"]])
        fwd_df = pd.concat(vals, ignore_index=True)
        panel = panel.merge(fwd_df, on=["iso3", "year", "month"], how="left")

    # Month-end timestamp convenience column
    panel["date"] = pd.to_datetime(
        panel["year"].astype(str) + "-" + panel["month"].astype(str).str.zfill(2) + "-01"
    ) + pd.offsets.MonthEnd(0)

    panel = panel.sort_values(["iso3", "year", "month"]).reset_index(drop=True)

    # Coverage summary
    n_countries = panel["iso3"].nunique()
    ym_min = panel[["year", "month"]].dropna().sort_values(["year", "month"]).iloc[0]
    ym_max = panel[["year", "month"]].dropna().sort_values(["year", "month"]).iloc[-1]
    print(f"\nPanel: {panel.shape[0]} rows x {panel.shape[1]} cols")
    print(f"Countries: {n_countries}")
    print(f"Period range: {int(ym_min['year'])}-{int(ym_min['month']):02d} to {int(ym_max['year'])}-{int(ym_max['month']):02d}")

    out_path = DATA_DIR / "panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")

    return panel


def load_panel() -> pd.DataFrame:
    """Load cached panel; rebuild if not present."""
    cache = DATA_DIR / "panel.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return build_panel()


if __name__ == "__main__":
    df = build_panel()
    print(df.head())
    print(df.describe(include="all"))
