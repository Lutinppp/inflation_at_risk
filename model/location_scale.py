"""
Machado-Santos Silva (2019) Location-Scale Quantile Regression estimator.
Adapted for Inflation-at-Risk (IaR).

Model:
    hicp_{i,t+h} = α_i + X'β + (δ_i + X'γ) · ε_{i,t+h}

where hicp_{i,t+h} is the rolling mean annual HICP over the next h years
(consistent with López-Salido & Loria 2024, JME; Banerjee et al. 2024, JIMF).

Three-step procedure (MSS 2019):
  Step 1: Fixed-effects OLS on hicp_{i,t+h} ~ [hicp_lag, X_{i,t}] → residuals ê
  Step 2: Fixed-effects OLS on |ê|        ~ [hicp_lag, X_{i,t}] → fitted scale ŝ
  Step 3: z = ê / ŝ → empirical quantiles q(τ)
  Predicted: Q(τ) = (α̂_i + δ̂_i·q(τ)) + [hicp_lag, X]'β̂ + [hicp_lag, X]'γ̂·q(τ)

Key difference from DaR:
  - Dependent variable: rolling-mean future HICP (not debt/GDP)
  - hicp_lag always included as AR control (analogous to initial_debt in DaR)
  - Horizons: h ∈ {1, 2, 4} years

Reference: Machado & Santos Silva (2019), JOE 213(1), 145-173.
           Furceri et al. (2025), IMF WP/25/86.
           López-Salido & Loria (2024), JME.
           Banerjee et al. (2024), JIMF.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

from linearmodels import PanelOLS

MODEL_DIR = Path(__file__).parent
DATA_DIR  = MODEL_DIR.parent / "data"

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
HORIZONS  = [1, 2, 4]

# Conditioning variables for Inflation-at-Risk
COND_VARS = {
    "hicp_lag":           "hicp_lag",
    "output_gap":         "output_gap",
    "infl_expectations":  "infl_expectations",
    "energy_price_chg":   "energy_price_chg",
    "import_price_chg":   "import_price_chg",
    "clifs":              "clifs",
    "spread_10y":         "spread_10y",
    "wui":                "wui",
    "food_price_chg":     "food_price_chg",
    "labour_cost_chg":    "labour_cost_chg",
    "neer_chg":           "neer_chg",
    "reer_chg":           "reer_chg",
}

# Expected coefficient signs for QA checks
EXPECTED_SIGNS = {
    # (locationβ_sign, scaleγ_sign)
    "hicp_lag":          ("+", "+"),   # persistence; fatter right tail when high
    "output_gap":        ("+", None),  # positive gap → higher inflation
    "infl_expectations": ("+", None),  # forward-looking anchor
    "energy_price_chg":  ("+", "+"),   # energy shocks → right tail
    "import_price_chg":  ("+", None),  # imported inflation
    "clifs":             ("-", "-"),   # stress → demand compression, left tail
    "spread_10y":        (None, None), # ambiguous for inflation
    "wui":               (None, "+"),  # uncertainty → fatter tails
    "food_price_chg":    ("+", "+"),   # food shocks → right tail
    "labour_cost_chg":   ("+", None),  # cost-push → higher inflation
    "neer_chg":          ("-", None),  # appreciation → cheaper imports
    "reer_chg":          ("-", None),  # real appreciation → dis-inflationary
}


def _prepare_panel(
    df: pd.DataFrame,
    cond_col: str,
    horizon: int,
) -> pd.DataFrame:
    """
    Prepare sub-panel for one conditioning variable × horizon.
    Returns DataFrame with MultiIndex (entity, time) for linearmodels.
    
    Note: hicp_lag is ALWAYS included as a control (AR persistence term).
    When cond_col == "hicp_lag", only hicp_lag is in indep_cols (no duplication).
    """
    dep_var   = f"hicp_fwd{horizon}"
    # AR control is always included; avoid duplicate if cond_col IS hicp_lag
    if cond_col == "hicp_lag":
        required = ["hicp_lag", dep_var]
    else:
        required = ["hicp_lag", cond_col, dep_var]

    sub = df[["iso3", "year"] + required].dropna(subset=required).copy()

    # The upstream panel may be monthly. MSS estimation here is annual,
    # so collapse to one observation per (iso3, year).
    if sub.duplicated(subset=["iso3", "year"]).any():
        sub = (
            sub.groupby(["iso3", "year"], as_index=False)[required]
            .mean()
        )

    sub = sub.set_index(["iso3", "year"])
    return sub


def _ols_fe(df: pd.DataFrame, dep: str, indep: list[str]) -> tuple:
    """
    Run fixed-effects OLS via linearmodels.PanelOLS.
    Returns (fitted_values, residuals, params_df, entity_effects).
    """
    formula = f"{dep} ~ 1 + {' + '.join(indep)} + EntityEffects"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model  = PanelOLS.from_formula(formula, data=df, drop_absorbed=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)

    fitted    = result.fitted_values.squeeze()
    residuals = result.resids.squeeze()
    params    = result.params
    fe_df     = (result.estimated_effects.copy()
                 if hasattr(result, "estimated_effects") else pd.DataFrame())

    return fitted, residuals, params, fe_df


def run_location_scale(
    panel: pd.DataFrame,
    cond_var_name: str,
    horizon: int,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run the MSS three-step estimator for one conditioning variable and horizon.

    Parameters
    ----------
    panel         : clean panel DataFrame (iso3, year columns + all regressors)
    cond_var_name : name key in COND_VARS dict
    horizon       : forecast horizon h ∈ {1, 2, 4}
    quantiles     : list of τ values (default: [0.05, 0.25, 0.50, 0.75, 0.95])

    Returns
    -------
    DataFrame with columns: iso3, year, horizon, cond_var,
                            Q05, Q25, Q50, Q75, Q95
    """
    if quantiles is None:
        quantiles = QUANTILES

    cond_col = COND_VARS[cond_var_name]
    dep_var  = f"hicp_fwd{horizon}"

    sub = _prepare_panel(panel, cond_col, horizon)
    if len(sub) < 30:
        print(f"    Skipping {cond_var_name} h={horizon}: insufficient data ({len(sub)} obs)")
        return pd.DataFrame()

    # Regressors: always include hicp_lag; add cond_col if it's not hicp_lag
    if cond_col == "hicp_lag":
        indep_cols = ["hicp_lag"]
    else:
        indep_cols = ["hicp_lag", cond_col]

    # ── Step 1: OLS on level ─────────────────────────────────────────────
    try:
        fitted1, resid1, params1, fe1 = _ols_fe(sub, dep_var, indep_cols)
    except Exception as exc:
        print(f"    Step 1 failed ({cond_var_name}, h={horizon}): {exc}")
        return pd.DataFrame()

    # Defensive guard: some backends can emit duplicate MultiIndex labels.
    if hasattr(resid1, "index") and not resid1.index.is_unique:
        resid1 = resid1.groupby(level=[0, 1]).mean()
    if hasattr(fitted1, "index") and not fitted1.index.is_unique:
        fitted1 = fitted1.groupby(level=[0, 1]).mean()

    # ── Step 2: OLS on absolute residuals ────────────────────────────────
    sub2 = sub.copy()
    sub2["__abs_resid__"] = np.abs(resid1.reindex(sub2.index).values)
    sub2 = sub2.dropna(subset=["__abs_resid__"])

    try:
        fitted2, _, params2, fe2 = _ols_fe(sub2, "__abs_resid__", indep_cols)
    except Exception as exc:
        print(f"    Step 2 failed ({cond_var_name}, h={horizon}): {exc}")
        return pd.DataFrame()

    if hasattr(fitted2, "index") and not fitted2.index.is_unique:
        fitted2 = fitted2.groupby(level=[0, 1]).mean()

    # ── Step 3: standardise residuals, get empirical quantiles ───────────
    scale_hat = fitted2.reindex(sub2.index).clip(lower=1e-6)
    common_idx = scale_hat.index.intersection(resid1.index)

    z = resid1.reindex(common_idx) / scale_hat.reindex(common_idx)
    z = z.dropna()

    if len(z) < 10:
        print(f"    Step 3 failed ({cond_var_name}, h={horizon}): too few z values")
        return pd.DataFrame()

    q_empirical = {tau: float(np.quantile(z, tau)) for tau in quantiles}

    # ── Predicted quantile: Q(τ | X_{i,t}) ─────────────────────────────
    # Q(τ) = μ̂_{i,t} + q(τ) · ŝ_{i,t}
    mu_hat = fitted1.reindex(common_idx)
    s_hat  = scale_hat.reindex(common_idx)

    records = []
    for idx in common_idx:
        iso3, year = idx
        mu  = float(mu_hat[idx])
        s   = float(s_hat[idx])
        row = {"iso3": iso3, "year": int(year), "horizon": horizon,
               "cond_var": cond_var_name}
        for tau, qz in q_empirical.items():
            col = f"Q{int(round(tau * 100)):02d}"
            row[col] = mu + qz * s
        records.append(row)

    result_df = pd.DataFrame(records)

    # ── QA: coefficient sign check ────────────────────────────────────────
    _check_signs(cond_var_name, horizon, params1, params2, cond_col)

    return result_df


def _check_signs(
    cond_var_name: str,
    horizon: int,
    params1: pd.Series,
    params2: pd.Series,
    cond_col: str,
) -> None:
    """Flag unexpected coefficient signs for QA."""
    expected = EXPECTED_SIGNS.get(cond_var_name, (None, None))
    exp_beta, exp_gamma = expected

    def _get_coeff(params: pd.Series, col: str) -> float | None:
        for key in params.index:
            if col in str(key):
                return float(params[key])
        return None

    beta  = _get_coeff(params1, cond_col)
    gamma = _get_coeff(params2, cond_col)

    flags = []
    if exp_beta == "+" and beta is not None and beta < 0:
        flags.append(f"β={beta:.3f} (expected +)")
    if exp_beta == "-" and beta is not None and beta > 0:
        flags.append(f"β={beta:.3f} (expected −)")
    if exp_gamma == "+" and gamma is not None and gamma < 0:
        flags.append(f"γ={gamma:.3f} (expected +)")
    if exp_gamma == "-" and gamma is not None and gamma > 0:
        flags.append(f"γ={gamma:.3f} (expected −)")

    if flags:
        print(f"    ⚠ Sign check for {cond_var_name} h={horizon}: {', '.join(flags)}")


def run_all(panel: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """
    Run MSS estimator for all conditioning variables × all horizons.
    Returns merged DataFrame of quantile predictions.
    """
    if horizons is None:
        horizons = HORIZONS

    results = []
    for h in horizons:
        print(f"\n  Horizon h={h}:")
        for var_name in COND_VARS:
            print(f"    {var_name} …", end=" ", flush=True)
            df_q = run_location_scale(panel, var_name, h)
            if not df_q.empty:
                results.append(df_q)
                print(f"{len(df_q)} rows")
            else:
                print("skipped")

    if not results:
        raise RuntimeError("No quantile predictions produced. Check panel data.")

    out = pd.concat(results, ignore_index=True)
    out_path = MODEL_DIR / "quantile_predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved quantile predictions → {out_path}")
    return out


def load_quantile_predictions() -> pd.DataFrame:
    """Load cached predictions; rerun if missing."""
    cache = MODEL_DIR / "quantile_predictions.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from data.panel_builder import load_panel
    return run_all(load_panel())


if __name__ == "__main__":
    from data.panel_builder import build_panel
    panel = build_panel()
    preds = run_all(panel)
    print(preds.head(20))
    print(f"Shape: {preds.shape}")
