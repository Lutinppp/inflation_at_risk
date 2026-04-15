"""
Panel builder — assembles clean estimation panel from all IaR data sources.

Saves: data/panel.parquet
Columns: iso3, year, hicp, hicp_lag, output_gap, infl_expectations,
         energy_price_chg, import_price_chg, clifs, spread_10y, wui
         + forward HICP rolling means: hicp_fwd1, hicp_fwd2, hicp_fwd4

Estimation panel: EU member states with continuous HICP from 1999 onward
(target 25–30 countries). Non-G4 countries are used to estimate the model;
G4 is the focus for projection.

Forward variable: h-period-ahead HICP is the rolling mean of the next h
annual HICP values — consistent with López-Salido & Loria (2024, JME).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

# EU member states that joined by 2007 (sufficient HICP history from 1999)
EU_CORE = [
    "AUT", "BEL", "BGR", "CYP", "CZE", "DEU", "DNK", "ESP", "EST", "FIN",
    "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ITA", "LTU", "LUX", "LVA",
    "MLT", "NLD", "NOR", "POL", "PRT", "ROU", "SVK", "SVN", "SWE",
]


def build_panel(
    min_obs: int = 15,
    start_year: int = 1999,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    Merge all IaR data sources into a clean estimation panel.

    Parameters
    ----------
    min_obs    : minimum non-missing HICP observations per country
    start_year : first year to include
    end_year   : last year to include

    Returns
    -------
    Clean panel DataFrame saved to data/panel.parquet.
    """
    from data.hicp          import load_hicp
    from data.ameco         import load_ameco
    from data.output_gap    import load_output_gap
    from data.energy_prices import load_energy_prices
    from data.import_prices import load_import_prices
    from data.imf_fsi       import load_fsi
    from data.ecb_spreads   import load_spreads
    from data.wui           import load_wui

    print("Building IaR estimation panel …")

    # ── Load all components ────────────────────────────────────────────────
    hicp      = load_hicp()[["iso3", "year", "hicp"]]
    ameco     = load_ameco()[["iso3", "year", "infl_expectations"]]
    og        = load_output_gap()[["iso3", "year", "output_gap"]]
    energy    = load_energy_prices()[["iso3", "year", "energy_price_chg"]]
    imports   = load_import_prices()[["iso3", "year", "import_price_chg"]]
    fsi       = load_fsi()[["iso3", "year", "clifs"]]
    spreads   = load_spreads()[["iso3", "year", "spread_10y"]]
    wui       = load_wui()[["iso3", "year", "wui"]] if _has_wui_col(load_wui()) \
                else _make_wui_stub()

    # ── Merge on [iso3, year] ─────────────────────────────────────────────
    panel = (
        hicp
        .merge(og,      on=["iso3", "year"], how="left")
        .merge(ameco,   on=["iso3", "year"], how="left")
        .merge(energy,  on=["iso3", "year"], how="left")
        .merge(imports, on=["iso3", "year"], how="left")
        .merge(fsi,     on=["iso3", "year"], how="left")
        .merge(spreads, on=["iso3", "year"], how="left")
        .merge(wui,     on=["iso3", "year"], how="left")
    )

    # ── Year / country filter ──────────────────────────────────────────────
    panel = panel[
        (panel["year"] >= start_year) & (panel["year"] <= end_year)
    ].copy()

    # Keep only EU_CORE countries with at least min_obs HICP observations
    hicp_counts = (
        panel.groupby("iso3")["hicp"]
        .apply(lambda x: x.notna().sum())
        .reset_index()
        .rename(columns={"hicp": "n_hicp"})
    )
    good_countries = hicp_counts.loc[
        (hicp_counts["n_hicp"] >= min_obs) &
        (hicp_counts["iso3"].isin(EU_CORE)),
        "iso3",
    ]
    panel = panel[panel["iso3"].isin(good_countries)].copy()

    # ── Winsorize extremes ─────────────────────────────────────────────────
    panel["hicp"]              = panel["hicp"].clip(lower=-3.0, upper=25.0)
    panel["output_gap"]        = panel["output_gap"].clip(lower=-15.0, upper=10.0)
    # Fill infl_expectations gaps with current-year HICP as adaptive-expectations
    # proxy (standard practice for countries without dedicated expectation surveys).
    panel["infl_expectations"] = panel["infl_expectations"].fillna(panel["hicp"])
    panel["infl_expectations"] = panel["infl_expectations"].clip(lower=-2.0, upper=15.0)
    panel["energy_price_chg"]  = panel["energy_price_chg"].clip(lower=-60.0, upper=80.0)
    panel["import_price_chg"]  = panel["import_price_chg"].clip(lower=-40.0, upper=50.0)
    panel["spread_10y"]        = panel["spread_10y"].clip(lower=-1.0, upper=20.0)
    panel["wui"]               = panel["wui"].clip(lower=0.0, upper=2.0)

    # ── Lag HICP — AR persistence control ─────────────────────────────────
    panel = panel.sort_values(["iso3", "year"])
    panel["hicp_lag"] = panel.groupby("iso3")["hicp"].shift(1)

    # ── Forward HICP: rolling mean over next h years ───────────────────────
    # h=1: just next year's HICP
    # h=2: mean of next 2 years
    # h=4: mean of next 4 years
    for h in [1, 2, 4]:
        fwd_vals = []
        for iso3, grp in panel.groupby("iso3"):
            grp = grp.sort_values("year").copy()
            hicp_arr = grp["hicp"].values
            fwd = np.full(len(hicp_arr), np.nan)
            for i in range(len(hicp_arr)):
                window = hicp_arr[i + 1: i + 1 + h]
                if len(window) == h and not any(np.isnan(window)):
                    fwd[i] = float(np.mean(window))
            grp[f"hicp_fwd{h}"] = fwd
            fwd_vals.append(grp[["iso3", "year", f"hicp_fwd{h}"]])
        fwd_df = pd.concat(fwd_vals, ignore_index=True)
        panel = panel.merge(fwd_df, on=["iso3", "year"], how="left")

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)

    # ── Coverage summary ─────────────────────────────────────────────────
    n_countries = panel["iso3"].nunique()
    year_min    = int(panel["year"].min())
    year_max    = int(panel["year"].max())
    print(f"\nPanel: {panel.shape[0]} rows × {panel.shape[1]} cols")
    print(f"Countries: {n_countries}")
    print(f"Year range: {year_min}–{year_max}")
    print("\nNon-null counts per variable:")
    cond_cols = ["hicp_lag", "output_gap", "infl_expectations",
                 "energy_price_chg", "import_price_chg", "clifs",
                 "spread_10y", "wui"]
    for col in cond_cols:
        if col in panel.columns:
            print(f"  {col:25s}: {panel[col].notna().sum():5d}")

    out_path = DATA_DIR / "panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")

    return panel


def _has_wui_col(df: pd.DataFrame) -> bool:
    """Check that the WUI DataFrame has the 'wui' column."""
    return "wui" in df.columns


def _make_wui_stub() -> pd.DataFrame:
    """Return empty stub if WUI data has unexpected schema."""
    return pd.DataFrame(columns=["iso3", "year", "wui"])


def load_panel() -> pd.DataFrame:
    """Load cached panel; rebuild if not present."""
    cache = DATA_DIR / "panel.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    return build_panel()


if __name__ == "__main__":
    df = build_panel()
    print(df.describe())
    print("\nG4 sample:")
    g4 = df[df["iso3"].isin(["FRA", "DEU", "ITA", "ESP"])].tail(20)
    print(g4[["iso3", "year", "hicp", "hicp_lag", "output_gap",
              "infl_expectations", "energy_price_chg", "clifs"]].to_string())
