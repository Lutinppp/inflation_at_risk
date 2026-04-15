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
    skt_params: pd.DataFrame,
    panel: pd.DataFrame,
    horizon: int = 2,
    forecast_years: list[int] | None = None,
    qpreds: pd.DataFrame | None = None,
) -> dict:
    """
    Run de-anchoring logit models and produce G4 probability scores.

    Parameters
    ----------
    skt_params     : DataFrame with skewed-t parameters
    panel          : estimation panel (with 'hicp' column)
    horizon        : quantile forecast horizon
    forecast_years : years for which to produce G4 scores (default [2025, 2026])
    qpreds         : raw quantile predictions fallback

    Returns
    -------
    dict with keys:
      'logit_results'  : {cond_var: statsmodels Result}
      'deanchoring_scores' : DataFrame per-variable G4 scores
      'pooled_scores'      : DataFrame pooled G4 scores
    """
    if forecast_years is None:
        forecast_years = [2025, 2026]

    cond_vars = skt_params["cond_var"].unique().tolist()
    if not cond_vars and qpreds is not None:
        cond_vars = (qpreds["cond_var"].unique().tolist()
                     if "cond_var" in qpreds.columns else [])

    # ── 1. Build de-anchoring binary variable ─────────────────────────────
    deanchor_df = _build_deanchoring_variable(panel)

    base_rate = deanchor_df["deanchoring"].mean()
    print(f"  De-anchoring base rate: {base_rate:.3f} ({deanchor_df['deanchoring'].sum()} episodes)")

    # ── 2. Compute upside predictor ────────────────────────────────────────
    if len(skt_params) > 0:
        upside_df = _compute_upside(skt_params, horizon=horizon)
    elif qpreds is not None and "Q95" in qpreds.columns:
        sub_q = qpreds[qpreds["horizon"] == horizon].copy()
        sub_q["upside"] = (sub_q["Q95"] - sub_q["Q50"]).clip(lower=0)
        upside_df = sub_q[["iso3", "year", "cond_var", "upside"]].dropna()
    else:
        return {
            "logit_results": {},
            "deanchoring_scores": pd.DataFrame(),
            "pooled_scores": pd.DataFrame(),
            "base_rate": base_rate,
        }

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

            # ── Predict G4 scores for forecast years ─────────────────────
            for iso3 in G4_COUNTRIES:
                g4_sub = skt_params[
                    (skt_params["iso3"] == iso3) &
                    (skt_params["horizon"] == horizon) &
                    (skt_params["cond_var"] == cv)
                ]
                if g4_sub.empty:
                    continue

                for fy in forecast_years:
                    latest = g4_sub.sort_values("year").iloc[-1]
                    if any(pd.isna([latest["xi"], latest["omega"],
                                    latest["alpha"], latest["nu"]])):
                        continue
                    try:
                        from model.quantile_fit import skt_quantile_from_params
                        q95 = skt_quantile_from_params(0.95, latest)
                        q50 = skt_quantile_from_params(0.50, latest)
                    except Exception:
                        continue
                    upside_val = max(q95 - q50, 0.0)
                    X_pred     = np.array([[1.0, upside_val]])
                    prob       = float(result.predict(X_pred)[0])
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

    return {
        "logit_results":      logit_results,
        "deanchoring_scores": scores_df,
        "pooled_scores":      pooled,
        "base_rate":          base_rate,
        "coefficients":       coeffs_df,
    }


def load_deanchoring_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached de-anchoring scores."""
    sc_path = CRISIS_DIR / "deanchoring_scores.parquet"
    po_path = CRISIS_DIR / "deanchoring_pooled.parquet"
    scores  = pd.read_parquet(sc_path)  if sc_path.exists()  else pd.DataFrame()
    pooled  = pd.read_parquet(po_path)  if po_path.exists()  else pd.DataFrame()
    return scores, pooled


if __name__ == "__main__":
    from model.quantile_fit import load_skt_params
    from data.panel_builder import load_panel

    skt   = load_skt_params()
    panel = load_panel()
    out   = run_deanchoring(skt, panel, horizon=2)
    print(out["pooled_scores"])
