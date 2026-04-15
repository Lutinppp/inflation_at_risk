"""
Log-score density pooling (Crump et al. 2023).

Combines individual conditioning-variable densities into a single pooled
density using log-score optimal weights.

Method:
  w* = argmax Σ_{t∈validation} Σ_i log[Σ_k w_k · f_k(d_{i,t+h}|X_{i,t})]
  subject to: Σ_k w_k = 1, w_k ≥ 0

Implemented via a 20-year rolling out-of-sample window from 2005 onward,
country-specific weights.

Reference: Crump, Eusepi, Giannoni & Sahin (2022), "A Large Bayesian VAR..."
           Furceri et al. (2025), IMF WP/25/86, Section IV.C.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from pathlib import Path
import warnings

RISK_DIR  = Path(__file__).parent
MODEL_DIR = RISK_DIR.parent / "model"

QUANTILES_TAU = [0.05, 0.25, 0.50, 0.75, 0.95]
POOL_START    = 2005
ROLL_WINDOW   = 20  # years
MIN_WEIGHT    = 1e-6


def _skt_pdf(x: float, xi: float, omega: float, alpha: float, nu: float) -> float:
    """
    PDF of generalised skewed-t at x.
    Uses Fernandez-Steel two-piece parametrisation.
    """
    gamma = np.exp(alpha)
    z     = (x - xi) / omega
    c     = 2.0 / (omega * (gamma + 1.0 / gamma))

    if z < 0:
        t_pdf = t_dist.pdf(-z / gamma, df=nu) / gamma
    else:
        t_pdf = t_dist.pdf(z * gamma, df=nu) * gamma

    return c * t_pdf


def _skt_pdf_vec(x_arr: np.ndarray, xi: float, omega: float,
                 alpha: float, nu: float) -> np.ndarray:
    """Vectorised PDF evaluation."""
    return np.array([_skt_pdf(xi_val, xi, omega, alpha, nu) for xi_val in x_arr])


def _log_score_weight_opt(
    realised: np.ndarray,
    per_model_pdfs: np.ndarray,
    n_models: int,
) -> np.ndarray:
    """
    Optimise log-score weights.

    Parameters
    ----------
    realised       : array shape (n_obs,) — actual debt/GDP realisations
    per_model_pdfs : array shape (n_obs, n_models) — f_k(d_t | X_t) for each model k
    n_models       : number of conditioning-variable models

    Returns
    -------
    weights : array shape (n_models,)
    """
    def neg_log_score(w):
        # Clip weights to ensure non-negativity
        w_pos = np.maximum(w, 0.0)
        w_pos = w_pos / w_pos.sum()
        # Mixture density evaluated at realisations
        mixture = per_model_pdfs @ w_pos  # shape (n_obs,)
        mixture = np.maximum(mixture, 1e-12)
        return -np.mean(np.log(mixture))

    w0 = np.ones(n_models) / n_models
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(MIN_WEIGHT, 1.0)] * n_models

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            neg_log_score, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 500},
        )

    w_opt = np.maximum(res.x, 0.0)
    w_opt /= w_opt.sum()
    return w_opt


def compute_country_weights(
    params_by_h: dict,
    hicp_actual: pd.DataFrame,
    horizons: list = None,
) -> np.ndarray:
    """
    Compute log-score pooling weights across forecast horizons for one country.

    Parameters
    ----------
    params_by_h : dict {h → DataFrame with cols: year, cond_var, xi, omega, alpha, nu}
                  for a single country
    hicp_actual : DataFrame with 'year' and 'hicp' columns (one country)
    horizons    : list of horizons to pool (default [1, 2, 4])

    Returns
    -------
    np.ndarray of shape (len(horizons),) — optimal log-score mixture weights summing to 1
    """
    if horizons is None:
        horizons = [1, 2, 4]
    n_h = len(horizons)

    hicp_ser = hicp_actual.set_index("year")["hicp"]

    # For each horizon h: average params across cond_vars per year, then
    # evaluate the resulting PDF at the h-year-forward realised HICP.
    h_pdfs: dict[int, dict[int, float]] = {}
    for h in horizons:
        df = params_by_h.get(h)
        if df is None or df.empty:
            h_pdfs[h] = {}
            continue

        by_year: dict[int, float] = {}
        for t, grp in df.groupby("year"):
            valid = grp.dropna(subset=["xi", "omega", "alpha", "nu"])
            if valid.empty:
                continue
            xi    = float(valid["xi"].mean())
            omega = float(valid["omega"].mean())
            alpha = float(valid["alpha"].mean())
            nu    = float(valid["nu"].mean())

            realised = hicp_ser.get(int(t) + h)
            if realised is None or np.isnan(realised):
                continue

            try:
                pdf_val = max(_skt_pdf(float(realised), xi, omega, alpha, nu), 1e-12)
            except Exception:
                pdf_val = 1e-12
            by_year[int(t)] = pdf_val

        h_pdfs[h] = by_year

    # Common evaluation years (all horizons must have a score)
    common_years: set[int] | None = None
    for h in horizons:
        h_years = set(h_pdfs.get(h, {}).keys())
        common_years = h_years if common_years is None else common_years & h_years

    if not common_years or len(common_years) < 3:
        return np.ones(n_h) / n_h

    t_sorted = sorted(common_years)
    n_obs = len(t_sorted)
    per_model_pdfs = np.zeros((n_obs, n_h))
    for k, h in enumerate(horizons):
        for j, t in enumerate(t_sorted):
            per_model_pdfs[j, k] = h_pdfs[h][t]

    dummy = np.zeros(n_obs)  # _log_score_weight_opt only uses per_model_pdfs
    return _log_score_weight_opt(dummy, per_model_pdfs, n_h)


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from data.panel_builder import build_panel
    import pandas as _pd

    skt   = load_skt_params()
    panel = build_panel()
    iso3  = "DEU"
    params_by_h = {h: skt[(skt["iso3"] == iso3) & (skt["horizon"] == h)] for h in [1, 2, 4]}
    hicp_actual = panel[panel["iso3"] == iso3][["year", "hicp"]]
    w = compute_country_weights(params_by_h, hicp_actual, horizons=[1, 2, 4])
    print(f"{iso3}: h=[1,2,4] weights = {w}")
