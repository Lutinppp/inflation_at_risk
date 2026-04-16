"""
De-anchoring early-warning logit model.

De-anchoring definition:
  HICP > 3.0% for two consecutive years (t+1 and t+2).
  deanchoring_{i,t} = 1 if hicp_{i,t+1} > 3.0 AND hicp_{i,t+2} > 3.0

Panel logit:
  P(deanchoring_{i,t+1,t+2}) = Λ(β₀ + β₁ · (Q95 − Q50)_{i,t})

Run separately for each conditioning variable.
Upside inflation risk (Q95−Q50) should be positively correlated with
de-anchoring probability — a higher upside tail implies elevated risk of
sustained inflation above the ECB 3% threshold.

G4 de-anchoring scores produced for 2025 and 2026.

Reference: Furceri et al. (2025), IMF WP/25/86, Section IV.D (adapted).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

CRISIS_DIR = Path(__file__).parent
DATA_DIR   = CRISIS_DIR.parent / "data"
MODEL_DIR  = CRISIS_DIR.parent / "model"
RISK_DIR   = CRISIS_DIR.parent / "risk"

G4_COUNTRIES     = ["FRA", "DEU", "ITA", "ESP"]
DEANCHOR_THRESH  = 3.0   # % — sustained HICP above this = de-anchoring episode
DEANCHOR_WINDOW  = 2     # consecutive years required


def _build_deanchoring_variable(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct binary de-anchoring variable.

    deanchoring_{i,t} = 1 if hicp_{i,t+1} > 3.0 AND hicp_{i,t+2} > 3.0
                      = 0 otherwise (including missing)
    """
    records = []
    for iso3, grp in panel.groupby("iso3"):
        grp = grp.sort_values("year").copy()
        years    = grp["year"].values
        hicp_arr = grp["hicp"].values

        for i, year in enumerate(years):
            if i + DEANCHOR_WINDOW > len(years) - 1:
                continue  # not enough future data
            future_hicp = hicp_arr[i + 1: i + 1 + DEANCHOR_WINDOW]
            if len(future_hicp) < DEANCHOR_WINDOW:
                continue
            if any(np.isnan(future_hicp)):
                continue
            deanchor = int(all(v > DEANCHOR_THRESH for v in future_hicp))
            records.append({"iso3": iso3, "year": int(year), "deanchoring": deanchor})

    return pd.DataFrame(records)


def _compute_upside(
    skt_params: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Compute Q95−Q50 upside measure from SKT parameters.
    Returns DataFrame: iso3, year, cond_var, upside.
    """
    from model.quantile_fit import skt_quantile_from_params

    records = []
    for _, row in skt_params[skt_params["horizon"] == horizon].iterrows():
        if any(pd.isna([row["xi"], row["omega"], row["alpha"], row["nu"]])):
            continue
        try:
            q95 = skt_quantile_from_params(0.95, row)
            q50 = skt_quantile_from_params(0.50, row)
            upside = max(q95 - q50, 0.0)
            records.append({
                "iso3": row["iso3"],
                "year": int(row["year"]),
                "cond_var": row["cond_var"],
                "upside": upside,
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def run_deanchoring(
    skt_params=None,
    panel: pd.DataFrame = None,
    iar: pd.DataFrame = None,
    horizon: int = 2,
    countries: list | None = None,
    forecast_years: list[int] | None = None,
    qpreds: pd.DataFrame | None = None,
) -> tuple:
    """
    Run de-anchoring logit models and produce G4 probability scores.

    Parameters
    ----------
    skt_params     : dict {h → DataFrame} or flat DataFrame with 'horizon' column,
                     or None (uses iar for upside)
    panel          : estimation panel (with 'hicp' column)
    iar            : IaR DataFrame with 'iso3', 'Upside' columns (for forecast scores)
    horizon        : quantile forecast horizon
    countries      : iso3 codes to produce scores for (default: G4_COUNTRIES)
    forecast_years : years for which to produce G4 scores (default [2025, 2026])
    qpreds         : raw quantile predictions fallback

    Returns
    -------
    (logit_result, pooled_scores_df)
      logit_result   : statsmodels Logit result (pooled across cond_vars)
      pooled_scores  : DataFrame with iso3, year, pooled_prob
    """
    if forecast_years is None:
        forecast_years = [2025, 2026]
    if countries is None:
        countries = G4_COUNTRIES

    # Normalise skt_params to a flat DataFrame (iso3, year, horizon, cond_var, xi, omega, alpha, nu)
    if skt_params is None:
        skt_flat = pd.DataFrame()
    elif isinstance(skt_params, dict):
        frames = []
        for h, df in skt_params.items():
            tmp = df.copy()
            if "horizon" not in tmp.columns:
                tmp["horizon"] = int(h)
            frames.append(tmp)
        skt_flat = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        skt_flat = skt_params.copy()

    cond_vars = skt_flat["cond_var"].unique().tolist() if "cond_var" in skt_flat.columns and len(skt_flat) > 0 else []
    if not cond_vars and qpreds is not None:
        cond_vars = (qpreds["cond_var"].unique().tolist()
                     if "cond_var" in qpreds.columns else [])

    # ── 1. Build de-anchoring binary variable ─────────────────────────────
    deanchor_df = _build_deanchoring_variable(panel)

    base_rate = deanchor_df["deanchoring"].mean()
    print(f"  De-anchoring base rate: {base_rate:.3f} ({deanchor_df['deanchoring'].sum()} episodes)")

    # ── 2. Compute upside predictor ────────────────────────────────────────
    if len(skt_flat) > 0:
        upside_df = _compute_upside(skt_flat, horizon=horizon)
    elif qpreds is not None and "Q95" in qpreds.columns:
        sub_q = qpreds[qpreds["horizon"] == horizon].copy()
        sub_q["upside"] = (sub_q["Q95"] - sub_q["Q50"]).clip(lower=0)
        upside_df = sub_q[["iso3", "year", "cond_var", "upside"]].dropna()
    else:
        pooled_empty = pd.DataFrame(columns=["iso3", "year", "pooled_prob", "base_rate"])
        return None, pooled_empty

    # Merge upside with de-anchoring
    model_df = upside_df.merge(deanchor_df, on=["iso3", "year"], how="inner")
    model_df = model_df.dropna(subset=["upside", "deanchoring"])

    logit_results   = {}
    score_records   = []
    coeff_records   = []

    for cv in cond_vars:
        sub = model_df[model_df["cond_var"] == cv].copy()
        if len(sub) < 30 or sub["deanchoring"].sum() < 5:
            print(f"    {cv}: insufficient events ({sub['deanchoring'].sum()}), skipping logit")
            continue

        X = sm.add_constant(sub["upside"].values, has_constant="add")
        y = sub["deanchoring"].values.astype(float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model  = Logit(y, X)
                result = model.fit(disp=False, maxiter=200)
            logit_results[cv] = result

            beta_coeff = float(result.params[1]) if len(result.params) > 1 else np.nan
            beta_pval  = float(result.pvalues[1]) if len(result.pvalues) > 1 else np.nan
            print(f"    {cv}: β={beta_coeff:.3f}, p={beta_pval:.3f}")
            coeff_records.append({
                "cond_var": cv, "beta": beta_coeff, "pvalue": beta_pval,
            })

            # ── Predict scores for forecast years ─────────────────────────
            # For each forecast_year, select the horizon that best matches the
            # time distance from the latest available conditioning data:
            #   h = forecast_year − max_cond_year  (clamped to [1, 4])
            # This means 2026 uses h=1 upside (1-yr ahead from 2025 data) and
            # 2027 uses h=2 upside (2-yr ahead from 2025 data), capturing
            # genuinely different tail-risk widths.
            cv_skt = skt_flat[skt_flat["cond_var"] == cv]
            global_max_year = int(cv_skt["year"].max()) if len(cv_skt) > 0 else 2025

            available_horizons = sorted(skt_flat["horizon"].unique())

            for iso3 in countries:
                for fy in forecast_years:
                    # Horizon matched to forecast distance from latest data
                    h_match = max(1, min(int(fy - global_max_year), max(available_horizons)))
                    if h_match not in available_horizons:
                        h_match = available_horizons[-1]

                    cv_h_sub = skt_flat[
                        (skt_flat["iso3"] == iso3) &
                        (skt_flat["horizon"] == h_match) &
                        (skt_flat["cond_var"] == cv)
                    ]
                    if cv_h_sub.empty:
                        continue
                    latest = cv_h_sub.sort_values("year").iloc[-1]

                    upside_val = None
                    if not any(pd.isna([latest["xi"], latest["omega"],
                                        latest["alpha"], latest["nu"]])):
                        try:
                            from model.quantile_fit import skt_quantile_from_params
                            q95 = skt_quantile_from_params(0.95, latest)
                            q50 = skt_quantile_from_params(0.50, latest)
                            upside_val = max(q95 - q50, 0.0)
                        except Exception:
                            pass

                    # Fallback: use iar.Upside
                    if upside_val is None and iar is not None and not iar.empty:
                        row_iar = iar[iar["iso3"] == iso3]
                        if not row_iar.empty and pd.notna(row_iar.iloc[0].get("Upside")):
                            upside_val = float(row_iar.iloc[0]["Upside"])

                    if upside_val is None:
                        continue

                    X_pred = np.array([[1.0, upside_val]])
                    prob   = float(result.predict(X_pred)[0])
                    score_records.append({
                        "iso3":     iso3,
                        "year":     fy,
                        "cond_var": cv,
                        "prob":     prob,
                        "upside":   upside_val,
                    })
        except Exception as exc:
            print(f"    {cv}: logit failed — {exc}")
            continue

    scores_df = pd.DataFrame(score_records) if score_records else pd.DataFrame()
    coeffs_df = pd.DataFrame(coeff_records) if coeff_records else pd.DataFrame()

    # ── 3. Pooled scores: average across conditioning variables ───────────
    if not scores_df.empty:
        pooled = (
            scores_df.groupby(["iso3", "year"])["prob"]
            .mean()
            .reset_index()
            .rename(columns={"prob": "pooled_prob"})
        )
        pooled["base_rate"] = base_rate
    else:
        pooled = pd.DataFrame(columns=["iso3", "year", "pooled_prob", "base_rate"])

    # Save
    out_path = CRISIS_DIR / "deanchoring_scores.parquet"
    if not scores_df.empty:
        scores_df.to_parquet(out_path, index=False)
        print(f"  Saved de-anchoring scores → {out_path}")

    pooled_path = CRISIS_DIR / "deanchoring_pooled.parquet"
    if not pooled.empty:
        pooled.to_parquet(pooled_path, index=False)
        print(f"  Saved pooled scores → {pooled_path}")

    # ── 4. Pooled logit result (for .summary()) ───────────────────────────
    pooled_result = None
    if len(model_df) >= 30 and model_df["deanchoring"].sum() >= 5:
        try:
            X_pool = sm.add_constant(model_df["upside"].values, has_constant="add")
            y_pool = model_df["deanchoring"].values.astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pooled_result = Logit(y_pool, X_pool).fit(disp=False, maxiter=200)
        except Exception as exc:
            print(f"  Pooled logit failed: {exc}")

    if pooled_result is None and logit_results:
        # Fall back to best per-cond_var result by log-likelihood
        pooled_result = max(logit_results.values(), key=lambda r: r.llf)

    return pooled_result, pooled


def load_deanchoring_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached de-anchoring scores."""
    sc_path = CRISIS_DIR / "deanchoring_scores.parquet"
    po_path = CRISIS_DIR / "deanchoring_pooled.parquet"
    scores  = pd.read_parquet(sc_path)  if sc_path.exists()  else pd.DataFrame()
    pooled  = pd.read_parquet(po_path)  if po_path.exists()  else pd.DataFrame()
    return scores, pooled


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from data.panel_builder import build_panel

    skt   = load_skt_params()
    panel = build_panel()
    result, pooled = run_deanchoring(skt_params=skt, panel=panel, horizon=2)
    print(pooled)
