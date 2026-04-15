"""
Inflation-at-Risk extraction and ECB baseline re-centering.

Extracts from the pooled distribution:
  IaR      = Q95 (Inflation-at-Risk — 95th percentile)
  Upside   = Q95 − Q50 (upside inflation risk in pp)
  Downside = Q50 − Q05 (downside deflation risk in pp)

Re-centers the pooled median to the ECB's published HICP forecast for 2027
(AMECO infl_expectations value for 2027 per country, or ECB March 2026 staff
projections as fallback).

Focus countries: FR, DE, IT, ES (EU G4)
Focus horizon  : h=2

Reference: López-Salido & Loria (2024), JME.
           Furceri et al. (2025), IMF WP/25/86.
           Korobilis et al. (2021), ECB WP 2600.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from pathlib import Path

RISK_DIR  = Path(__file__).parent
MODEL_DIR = RISK_DIR.parent / "model"

G4_COUNTRIES = ["FRA", "DEU", "ITA", "ESP"]

# ECB March 2026 staff projections — fallback baselines for 2027
# Used when AMECO 2027 forecast is unavailable
ECB_BASELINE_2027 = {
    "FRA": 2.0,  # France
    "DEU": 2.2,  # Germany
    "ITA": 1.9,  # Italy
    "ESP": 2.1,  # Spain
}


def _fst_quantile(tau: float, xi: float, omega: float, alpha: float, nu: float) -> float:
    """Fernandez-Steel skewed-t quantile function."""
    from scipy.stats import t as t_dist
    gamma = np.exp(np.clip(alpha, -5.0, 5.0))
    nu    = max(float(nu), 2.01)
    omega = max(float(omega), 1e-6)
    p_mid = 1.0 / (1.0 + gamma)
    try:
        if tau < p_mid:
            q_t = t_dist.ppf(max(tau * (1.0 + gamma) / 2.0, 1e-9), df=nu)
            return xi + omega * q_t / gamma
        else:
            arg = (tau * (1.0 + gamma) - (gamma - 1.0)) / 2.0
            q_t = t_dist.ppf(min(max(arg, 1e-9), 1 - 1e-9), df=nu)
            return xi + omega * q_t * gamma
    except Exception:
        return float(xi)


def _pooled_quantile(
    tau: float,
    component_params: list[dict],
    weights: list[float],
) -> float:
    """
    Compute quantile of weighted mixture density via CDF inversion.
    The pooled CDF is: F*(x) = Σ_k w_k · F_k(x)
    """
    from scipy.stats import t as t_dist

    def fst_cdf(x, xi, omega, alpha, nu):
        """CDF of Fernandez-Steel skewed-t."""
        gamma = np.exp(np.clip(alpha, -5.0, 5.0))
        nu    = max(nu, 2.01)
        omega = max(omega, 1e-6)
        z = (x - xi) / omega
        if z < 0:
            return t_dist.cdf(-z / gamma, df=nu) / (1.0 + gamma)
        else:
            return (1.0 / (1.0 + gamma) +
                    t_dist.cdf(z * gamma, df=nu) * gamma / (1.0 + gamma))

    def pooled_cdf(x):
        val = 0.0
        for params, w in zip(component_params, weights):
            xi, omega, alpha, nu = (params["xi"], params["omega"],
                                    params["alpha"], params["nu"])
            if any(pd.isna([xi, omega, alpha, nu])):
                continue
            try:
                val += w * fst_cdf(x, xi, omega, alpha, nu)
            except Exception:
                pass
        return val

    # Find bounds
    all_q01 = [_fst_quantile(0.01, p["xi"], p["omega"], p["alpha"], p["nu"])
               for p in component_params
               if not any(pd.isna([p["xi"], p["omega"], p["alpha"], p["nu"]]))]
    all_q99 = [_fst_quantile(0.99, p["xi"], p["omega"], p["alpha"], p["nu"])
               for p in component_params
               if not any(pd.isna([p["xi"], p["omega"], p["alpha"], p["nu"]]))]

    if not all_q01 or not all_q99:
        return np.nan

    lo = min(all_q01) - 5.0
    hi = max(all_q99) + 5.0

    try:
        return brentq(lambda x: pooled_cdf(x) - tau, lo, hi, xtol=0.01, maxiter=200)
    except Exception:
        # Fallback: weighted average of component quantiles
        qs, ws = [], []
        for params, w in zip(component_params, weights):
            xi, omega, alpha, nu = (params["xi"], params["omega"],
                                    params["alpha"], params["nu"])
            if not any(pd.isna([xi, omega, alpha, nu])):
                qs.append(_fst_quantile(tau, xi, omega, alpha, nu))
                ws.append(w)
        if not qs:
            return np.nan
        ws = np.array(ws)
        ws /= ws.sum()
        return float(np.dot(ws, qs))


def _get_ecb_baseline(iso3: str, ameco_df: pd.DataFrame | None, year: int = 2027) -> float:
    """
    Look up the ECB/AMECO 2027 inflation expectation for a G4 country.
    Falls back to ECB March 2026 staff projections if AMECO unavailable.
    """
    if ameco_df is not None and not ameco_df.empty:
        row = ameco_df[(ameco_df["iso3"] == iso3) & (ameco_df["year"] == year)]
        if not row.empty and pd.notna(row.iloc[0]["infl_expectations"]):
            return float(row.iloc[0]["infl_expectations"])
    return ECB_BASELINE_2027.get(iso3, 2.0)


def compute_iar(
    skt_params: dict | pd.DataFrame,
    weights: dict | pd.DataFrame,
    ameco_df: pd.DataFrame | None = None,
    panel: pd.DataFrame | None = None,
    countries: list | None = None,
    horizon: int = 2,
    base_year: int = 2024,
    recenter: bool = True,
) -> pd.DataFrame:
    """
    Compute pooled Inflation-at-Risk for G4 countries.

    Parameters
    ----------
    skt_params : dict {h → DataFrame with iso3, year, cond_var, xi, omega, alpha, nu}
                 OR legacy single DataFrame with a 'horizon' column
    weights    : dict {iso3 → np.ndarray of horizon weights [w_h1, w_h2, ...]}
                 OR legacy DataFrame with iso3, cond_var, weight columns
    ameco_df   : AMECO inflation expectations DataFrame (for 2027 re-centring anchor)
    panel      : estimation panel (optional, unused but accepted for API compatibility)
    countries  : list of iso3 codes to process (default: G4_COUNTRIES)
    horizon    : primary forecast horizon used for per-driver waterfall decomposition
    base_year  : last data year for projection base
    recenter   : re-centre pooled median to ECB/AMECO 2027 baseline

    Returns
    -------
    DataFrame: iso3, Q05, Q50, Q95, IaR, Upside, Downside, ecb_baseline,
               + upside_{cond_var} waterfall decomposition columns
    """
    taus_to_extract = [0.05, 0.50, 0.95]

    if countries is None:
        countries = G4_COUNTRIES

    # Normalise skt_params to dict format {h: DataFrame}
    if isinstance(skt_params, pd.DataFrame):
        skt_dict: dict = {
            int(h): skt_params[skt_params["horizon"] == h]
            for h in skt_params["horizon"].unique()
        }
    else:
        skt_dict = {int(h): df for h, df in skt_params.items()}

    horizons = sorted(skt_dict.keys())

    results = []

    for iso3 in countries:
        # Latest year available across all horizons
        latest_years = []
        for h, df in skt_dict.items():
            sub = df[df["iso3"] == iso3]
            if not sub.empty:
                latest_years.append(int(sub["year"].max()))
        if not latest_years:
            print(f"  Warning: No skt params for {iso3}")
            continue
        latest_year = max(latest_years)

        # Horizon weights for this country
        if isinstance(weights, dict):
            w_arr = np.asarray(
                weights.get(iso3, np.ones(len(horizons)) / len(horizons)),
                dtype=float,
            )
        else:
            # Legacy DataFrame — equal horizon weights
            w_arr = np.ones(len(horizons)) / float(len(horizons))

        if len(w_arr) != len(horizons):
            w_arr = np.ones(len(horizons)) / float(len(horizons))
        w_arr = np.maximum(w_arr, 0.0)
        w_arr /= w_arr.sum()

        # For each horizon: average params across cond_vars at the latest year
        component_params: list[dict] = []
        component_weights: list[float] = []

        for h, w_h in zip(horizons, w_arr):
            df_h = skt_dict[h]
            params_h = df_h[
                (df_h["iso3"] == iso3) & (df_h["year"] == latest_year)
            ].dropna(subset=["xi", "omega", "alpha", "nu"])

            if params_h.empty:
                # Fall back to latest available year for this horizon
                df_h_country = df_h[df_h["iso3"] == iso3].dropna(
                    subset=["xi", "omega", "alpha", "nu"]
                )
                if df_h_country.empty:
                    continue
                params_h = df_h_country[
                    df_h_country["year"] == df_h_country["year"].max()
                ]

            if params_h.empty:
                continue

            xi    = float(params_h["xi"].mean())
            omega = float(params_h["omega"].mean())
            alpha = float(params_h["alpha"].mean())
            nu    = float(params_h["nu"].mean())

            component_params.append({"xi": xi, "omega": omega, "alpha": alpha, "nu": nu})
            component_weights.append(float(w_h))

        if not component_params:
            print(f"  Warning: No valid SKT components for {iso3}")
            continue

        wt = np.array(component_weights)
        wt /= wt.sum()

        # Extract pooled quantiles (pooled across horizons)
        q_values: dict[float, float] = {}
        for tau in taus_to_extract:
            q_values[tau] = _pooled_quantile(tau, component_params, wt.tolist())

        q05, q50, q95 = q_values[0.05], q_values[0.50], q_values[0.95]

        # Re-centre to ECB/AMECO 2027 baseline
        ecb_med = _get_ecb_baseline(iso3, ameco_df, year=2027)
        shift   = (ecb_med - q50) if recenter and np.isfinite(q50) else 0.0
        q05 += shift
        q50 += shift
        q95 += shift

        iar      = q95
        upside   = q95 - q50
        downside = q50 - q05

        row_out = {
            "iso3":         iso3,
            "year":         latest_year,
            "horizon":      horizon,
            "proj_year":    latest_year + horizon,
            "Q05":          round(q05, 2),
            "Q50":          round(q50, 2),
            "Q95":          round(q95, 2),
            "IaR":          round(iar, 2),
            "Upside":       round(upside, 2),
            "Downside":     round(downside, 2),
            "ecb_baseline": ecb_med,
        }
        results.append(row_out)

    out = pd.DataFrame(results)

    # ── Per-driver upside decomposition (waterfall) ──────────────────────────
    # Use the primary horizon's cond_var predictions (equal-weight across cond_vars)
    primary_df = skt_dict.get(horizon, next(iter(skt_dict.values())))
    cond_vars  = primary_df["cond_var"].unique().tolist() if "cond_var" in primary_df.columns else []

    for cv in cond_vars:
        out[f"upside_{cv}"] = np.nan

    for idx, row in out.iterrows():
        iso3         = row["iso3"]
        total_upside = row["Upside"]
        country_h    = primary_df[
            (primary_df["iso3"] == iso3) &
            (primary_df["year"] == row["year"])
        ]
        driver_upsides: dict[str, float] = {}
        for cv in cond_vars:
            cv_row = country_h[country_h["cond_var"] == cv]
            if cv_row.empty:
                continue
            r = cv_row.iloc[0]
            if any(pd.isna([r["xi"], r["omega"], r["alpha"], r["nu"]])):
                continue
            try:
                q95_cv = _fst_quantile(0.95, r["xi"], r["omega"], r["alpha"], r["nu"])
                q50_cv = _fst_quantile(0.50, r["xi"], r["omega"], r["alpha"], r["nu"])
                driver_upsides[cv] = max(q95_cv - q50_cv, 0.0)
            except Exception:
                pass

        total_driver = sum(driver_upsides.values())
        for cv, du in driver_upsides.items():
            share = (du / total_driver * total_upside) if total_driver > 0 else 0.0
            out.loc[idx, f"upside_{cv}"] = round(share, 2)

    out_path = RISK_DIR / "iar_results.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved IaR results → {out_path}")
    print(out[["iso3", "Q05", "Q50", "Q95", "IaR", "Upside",
               "Downside", "ecb_baseline"]])
    return out


def load_iar(horizon: int = 2) -> pd.DataFrame:
    """Load cached IaR results."""
    cache = RISK_DIR / "iar_results.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from model.quantile_fit import load_skt_params
    from data.panel_builder import build_panel
    from data.ameco import load_ameco
    skt   = load_skt_params()
    panel = build_panel()
    ameco = load_ameco()
    skt_dict = {h: skt[skt["horizon"] == h] for h in skt["horizon"].unique()}
    countries = G4_COUNTRIES
    equal_weights = {iso3: np.ones(3) / 3.0 for iso3 in countries}
    return compute_iar(skt_dict, equal_weights, ameco_df=ameco, horizon=horizon)


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from data.panel_builder import build_panel
    from data.ameco import load_ameco

    skt   = load_skt_params()
    panel = build_panel()
    ameco = load_ameco()
    skt_dict = {h: skt[skt["horizon"] == h] for h in skt["horizon"].unique()}
    equal_weights = {iso3: np.ones(3) / 3.0 for iso3 in G4_COUNTRIES}
    iar = compute_iar(skt_dict, equal_weights, ameco_df=ameco, horizon=2)
    print(iar.T)
