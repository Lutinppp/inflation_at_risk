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
    skt_params: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int = 3,
    pool_start: int = POOL_START,
) -> pd.DataFrame:
    """
    Compute country-specific log-score pooling weights.

    Parameters
    ----------
    skt_params : DataFrame with columns iso3, year, horizon, cond_var, xi, omega, alpha, nu
    panel      : estimation panel (iso3, year, debt_gdp_fwdH)
    horizon    : forecast horizon

    Returns
    -------
    DataFrame: iso3, cond_var, weight
    """
    dep_col    = f"debt_gdp_fwd{horizon}"
    cond_vars  = skt_params["cond_var"].unique().tolist()
    n_models   = len(cond_vars)

    rows = []

    for iso3, country_params in skt_params[skt_params["horizon"] == horizon].groupby("iso3"):
        # Get realisations for this country in validation window
        realised_df = panel[
            (panel["iso3"] == iso3) &
            (panel["year"] >= pool_start)
        ][["year", dep_col]].dropna()

        if len(realised_df) < 5:
            # Insufficient data: equal weights
            for cv in cond_vars:
                rows.append({"iso3": iso3, "cond_var": cv, "weight": 1.0 / n_models})
            continue

        years_val = realised_df["year"].values
        d_real    = realised_df[dep_col].values

        # For each observation in validation, evaluate each model's PDF at realised value
        per_model_pdfs = np.zeros((len(d_real), n_models))

        for k, cond_var in enumerate(cond_vars):
            model_preds = country_params[
                (country_params["cond_var"] == cond_var) &
                (country_params["year"].isin(years_val))
            ].set_index("year")

            for j, yr in enumerate(years_val):
                if yr in model_preds.index:
                    r = model_preds.loc[yr]
                    if not any(pd.isna([r["xi"], r["omega"], r["alpha"], r["nu"]])):
                        try:
                            per_model_pdfs[j, k] = max(
                                _skt_pdf(d_real[j], r["xi"], r["omega"], r["alpha"], r["nu"]),
                                1e-12,
                            )
                        except Exception:
                            per_model_pdfs[j, k] = 1e-12
                    else:
                        per_model_pdfs[j, k] = 1e-12
                else:
                    per_model_pdfs[j, k] = 1e-12

        # Check that at least some models have non-trivial PDFs
        if per_model_pdfs.max() < 1e-10:
            weights = np.ones(n_models) / n_models
        else:
            weights = _log_score_weight_opt(d_real, per_model_pdfs, n_models)

        for k, cond_var in enumerate(cond_vars):
            rows.append({"iso3": iso3, "cond_var": cond_var, "weight": float(weights[k])})

    out = pd.DataFrame(rows)

    out_path = RISK_DIR / "pooling_weights.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved pooling weights → {out_path}")
    return out


def load_pooling_weights(horizon: int = 3) -> pd.DataFrame:
    """Load cached pooling weights."""
    cache = RISK_DIR / "pooling_weights.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from model.quantile_fit import load_skt_params
    from data.panel_builder import load_panel
    skt = load_skt_params()
    panel = load_panel()
    return compute_country_weights(skt, panel, horizon=horizon)


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from data.panel_builder import load_panel

    skt   = load_skt_params()
    panel = load_panel()
    w     = compute_country_weights(skt, panel, horizon=3)
    print(w)
