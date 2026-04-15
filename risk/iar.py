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
    skt_params: pd.DataFrame,
    weights: pd.DataFrame,
    horizon: int = 2,
    base_year: int = 2024,
    recenter: bool = True,
    ameco_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute pooled Inflation-at-Risk for G4 countries.

    Parameters
    ----------
    skt_params : DataFrame with skewed-t parameters (iso3, year, horizon, cond_var,
                 xi, omega, alpha, nu)
    weights    : DataFrame with country-specific pooling weights (iso3, cond_var, weight)
    horizon    : forecast horizon (default 2 → 2026 for assessment in 2024)
    base_year  : last data year for projection base
    recenter   : re-center pooled median to ECB/AMECO 2027 baseline
    ameco_df   : AMECO inflation expectations DataFrame (for 2027 re-centering anchor)

    Returns
    -------
    DataFrame: iso3, Q05, Q50, Q95, IaR, Upside, Downside, ecb_baseline,
               + per-driver upside decomposition columns
    """
    taus_to_extract = [0.05, 0.50, 0.95]
    cond_vars = skt_params["cond_var"].unique().tolist()

    results = []

    for iso3 in G4_COUNTRIES:
        country_skt = skt_params[
            (skt_params["iso3"] == iso3) &
            (skt_params["horizon"] == horizon)
        ]
        if country_skt.empty:
            print(f"  Warning: No skt params for {iso3} h={horizon}")
            continue

        latest_year = country_skt["year"].max()
        params_latest = country_skt[country_skt["year"] == latest_year]

        # Retrieve pooling weights
        country_weights_df = weights[weights["iso3"] == iso3]
        w_dict = dict(zip(country_weights_df["cond_var"], country_weights_df["weight"]))

        # Build component list
        component_params  = []
        component_weights = []
        for cv in cond_vars:
            row = params_latest[params_latest["cond_var"] == cv]
            if row.empty:
                continue
            r = row.iloc[0]
            if any(pd.isna([r["xi"], r["omega"], r["alpha"], r["nu"]])):
                continue
            component_params.append({
                "xi": float(r["xi"]), "omega": float(r["omega"]),
                "alpha": float(r["alpha"]), "nu": float(r["nu"]),
            })
            component_weights.append(w_dict.get(cv, 1.0 / len(cond_vars)))

        if not component_params:
            print(f"  Warning: No valid SKT components for {iso3}")
            continue

        # Normalise weights
        wt = np.array(component_weights)
        wt /= wt.sum()

        # Extract pooled quantiles
        q_values = {}
        for tau in taus_to_extract:
            q_values[tau] = _pooled_quantile(tau, component_params, wt.tolist())

        q05, q50, q95 = q_values[0.05], q_values[0.50], q_values[0.95]

        # Re-center to ECB/AMECO 2027 baseline
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

        # Per-driver weights
        for k, cv in enumerate(cond_vars):
            row_out[f"w_{cv}"] = float(wt[k]) if k < len(wt) else 0.0

        results.append(row_out)

    out = pd.DataFrame(results)

    # ── Driver upside decomposition (waterfall) ────────────────────────────
    cond_var_keys = [f.replace("w_", "") for f in out.columns if f.startswith("w_")]
    for cv in cond_var_keys:
        out[f"upside_{cv}"] = np.nan

    for idx, row in out.iterrows():
        iso3         = row["iso3"]
        total_upside = row["Upside"]
        country_skt  = skt_params[
            (skt_params["iso3"] == iso3) &
            (skt_params["horizon"] == horizon) &
            (skt_params["year"] == row["year"])
        ]
        driver_upsides = {}
        for cv in cond_var_keys:
            cv_row = country_skt[country_skt["cond_var"] == cv]
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
    from risk.pooling import load_pooling_weights
    skt     = load_skt_params()
    weights = load_pooling_weights(horizon=horizon)
    return compute_iar(skt, weights, horizon=horizon)


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from risk.pooling import load_pooling_weights
    from data.ameco import load_ameco

    skt     = load_skt_params()
    weights = load_pooling_weights(horizon=2)
    ameco   = load_ameco()
    iar     = compute_iar(skt, weights, horizon=2, ameco_df=ameco)
    print(iar.T)
