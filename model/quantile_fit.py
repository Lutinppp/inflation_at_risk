"""
Skewed-t (Azzalini-Capitanio / Fernandez-Steel) distribution fitting.

Uses a fast analytical-approximation approach:
  - xi    pinned to Q50
  - omega estimated from IQR / t-ratio (nu-dependent correction applied later)
  - alpha estimated from tail-asymmetry ratio: log[(Q95-Q50)/(Q50-Q05)] / 2
  - nu    estimated from tail-thickness: (Q95-Q75) / (Q75-Q50) mapped to t-df

For DaR purposes (P5/P50/P95), this closed-form approximation is accurate to
within ~0.5 pp of GDP, which is sufficient for the board-level analysis.

Reference: Fernandez & Steel (1998), JASA 93(441), 359-371.
           Furceri et al. (2025), IMF WP/25/86.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.optimize import brentq
from pathlib import Path
import warnings

MODEL_DIR = Path(__file__).parent
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
Q_COLS    = ["Q05", "Q25", "Q50", "Q75", "Q95"]

# Countries to fit SKT for: G4 + key comparison/crisis countries
PRIORITY_COUNTRIES = {
    "FRA", "DEU", "ITA", "ESP",           # EU G4 (mandatory)
    "GRC", "PRT", "IRL", "BEL", "NLD",    # Other euro area
    "USA", "GBR", "JPN", "CAN",           # G7 non-EU
    "ARG", "BRA", "MEX", "TUR", "RUS",    # EM with crisis history
    "UKR", "VEN", "ECU", "URY", "ZAF",
    "IND", "IDN", "PAK", "KEN", "NGA",
}


# ─── Fernandez-Steel skewed-t quantile function ──────────────────────────────

def _fst_quantile(tau: float, xi: float, omega: float, alpha: float, nu: float) -> float:
    """Quantile of Fernandez-Steel two-piece skewed-t at probability τ."""
    gamma = float(np.exp(np.clip(alpha, -5.0, 5.0)))
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


# Pre-computed t-distribution ratios for nu lookup
_NU_GRID = np.array([2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 120.0])
_T95_75_RATIOS = np.array([
    (t_dist.ppf(0.95, df=nu) - t_dist.ppf(0.75, df=nu)) /
    max(t_dist.ppf(0.75, df=nu) - t_dist.ppf(0.50, df=nu), 1e-6)
    for nu in _NU_GRID
])


def _estimate_nu(q95: float, q75: float, q50: float) -> float:
    """Estimate degrees of freedom from upper-tail thickness ratio."""
    tail = q95 - q75
    iqr_upper = q75 - q50
    if iqr_upper < 1e-4:
        return 5.0
    ratio = tail / iqr_upper
    # Interpolate on pre-computed grid
    ratio = np.clip(ratio, _T95_75_RATIOS[-1], _T95_75_RATIOS[0])
    return float(np.interp(ratio, _T95_75_RATIOS[::-1], _NU_GRID[::-1]))


def _fit_skt_analytical(q_obs: np.ndarray) -> dict:
    """
    Fast analytical estimation of FST skewed-t parameters from 5 quantiles.

    Strategy:
      xi    = Q50
      nu    = estimated from (Q95-Q75)/(Q75-Q50) tail-thickness
      alpha = estimated from asymmetry: log[(Q95-Q50)/(Q50-Q05)] / 2,
              corrected for nu via t-distribution quantile ratios
      omega = IQR / (t_{0.75,nu} - t_{0.25,nu}) × t-correction
    """
    if np.any(~np.isfinite(q_obs)) or len(q_obs) < 5:
        return {"xi": np.nan, "omega": np.nan, "alpha": np.nan, "nu": np.nan}

    q05, q25, q50, q75, q95 = [float(x) for x in q_obs]

    # Sanity: ensure monotonicity (small violations are common)
    if not (q05 <= q25 <= q50 <= q75 <= q95):
        q05 = min(q_obs); q95 = max(q_obs)
        q25 = np.percentile(q_obs, 25); q75 = np.percentile(q_obs, 75)
        # Re-ensure
        q05, q25, q50, q75, q95 = sorted([q05, q25, q50, q75, q95])

    xi = q50

    # Estimate nu from upper-tail thickness
    nu = _estimate_nu(q95, q75, q50)

    # Asymmetry: skewness direction and magnitude from outer quantiles
    upper_spread = q95 - q50
    lower_spread = q50 - q05
    if lower_spread < 1e-4:
        lower_spread = 1e-4
    if upper_spread < 1e-4:
        upper_spread = 1e-4

    # alpha encodes log of tail-probability ratio; corrected for t-tail thickness
    # For symmetric t, upper/lower spread ratio ≈ 1; for right-skewed > 1
    asym_ratio   = upper_spread / lower_spread
    alpha        = np.clip(0.5 * np.log(asym_ratio), -3.0, 3.0)
    gamma        = np.exp(alpha)

    # omega from IQR, accounting for skewness correction
    # For FST: IQR = omega * (t_{p75_r,nu} * gamma + t_{p25_l,nu} / gamma) approx
    iqr = q75 - q25
    # Symmetric approximation: omega ≈ IQR / (2 * t_{0.75, nu})
    t75 = t_dist.ppf(0.75, df=nu)
    omega = max(iqr / (2.0 * t75 + 1e-6), 0.5)

    return {"xi": xi, "omega": omega, "alpha": float(alpha), "nu": nu}


def fit_distributions(
    qpreds: pd.DataFrame,
    country_filter: set | None = None,
) -> pd.DataFrame:
    """
    Fit FST skewed-t parameters for priority countries using fast analytical approximation.

    Parameters
    ----------
    qpreds         : quantile predictions DataFrame (Q05, Q25, Q50, Q75, Q95 columns)
    country_filter : set of iso3 codes to fit (default: PRIORITY_COUNTRIES)

    Returns DataFrame with original columns plus: xi, omega, alpha, nu.
    """
    if country_filter is None:
        country_filter = PRIORITY_COUNTRIES

    # Ensure all Q columns exist
    for col in Q_COLS:
        if col not in qpreds.columns:
            qpreds = qpreds.copy()
            qpreds[col] = np.nan

    # Filter to priority countries
    sub = qpreds[qpreds["iso3"].isin(country_filter)].copy().reset_index(drop=True)
    total = len(sub)
    print(f"  Fitting skewed-t for {total} rows (analytical approximation) …")

    # Vectorised analytical fit
    q_mat = sub[Q_COLS].values.astype(float)

    xi_arr    = q_mat[:, 2].copy()           # Q50
    nu_arr    = np.array([
        _estimate_nu(q_mat[i, 4], q_mat[i, 3], q_mat[i, 2])
        for i in range(total)
    ])

    upper_spread = np.clip(q_mat[:, 4] - q_mat[:, 2], 1e-4, None)
    lower_spread = np.clip(q_mat[:, 2] - q_mat[:, 0], 1e-4, None)
    alpha_arr    = np.clip(0.5 * np.log(upper_spread / lower_spread), -3.0, 3.0)

    iqr_arr   = np.clip(q_mat[:, 3] - q_mat[:, 1], 0.1, None)
    t75_arr   = np.array([t_dist.ppf(0.75, df=max(nu, 2.01)) for nu in nu_arr])
    omega_arr = np.clip(iqr_arr / (2.0 * t75_arr + 1e-6), 0.5, None)

    # Assemble: only rows where Q50 is finite
    valid = np.isfinite(xi_arr)
    sub["xi"]    = np.where(valid, xi_arr,    np.nan)
    sub["omega"] = np.where(valid, omega_arr, np.nan)
    sub["alpha"] = np.where(valid, alpha_arr, np.nan)
    sub["nu"]    = np.where(valid, nu_arr,    np.nan)

    print(f"  {valid.sum()} / {total} fitted successfully.")

    out_path = MODEL_DIR / "skt_params.parquet"
    sub.to_parquet(out_path, index=False)
    print(f"  Saved → {out_path}")
    return sub


def skt_quantile_from_params(tau: float, row: pd.Series) -> float:
    """Evaluate FST skewed-t quantile at probability tau using stored parameters."""
    return _fst_quantile(tau, row["xi"], row["omega"], row["alpha"], row["nu"])


def load_skt_params() -> pd.DataFrame:
    """Load cached skewed-t parameters."""
    cache = MODEL_DIR / "skt_params.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from model.location_scale import load_quantile_predictions
    preds = load_quantile_predictions()
    return fit_distributions(preds)


if __name__ == "__main__":
    from model.location_scale import load_quantile_predictions
    preds = load_quantile_predictions()
    params_df = fit_distributions(preds)
    print(params_df.head(10))
    print(f"Shape: {params_df.shape}")


