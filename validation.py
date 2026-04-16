"""
Econometric validation suite for the EU G4 Inflation-at-Risk pipeline.

The suite is designed to work with either:
1. the notebook-style in-memory objects
   (`panel`, `results`, `skt_params`, `weights`, `iar`, `pooled_scores`,
   `daresults`), or
2. the repository's cached flat parquet outputs.

Results are returned as a nested dictionary of DataFrames / Series so they can
be inspected in a notebook or exported into tables for publication appendices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy.stats import (
    anderson_ksamp,
    binomtest,
    chi2,
    chisquare,
    ks_2samp,
    norm,
    t as student_t,
)
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.tsa.stattools import adfuller

from model.quantile_fit import skt_quantile_from_params

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
HORIZONS = [1, 2, 4]
Q_COLS = {0.05: "Q05", 0.25: "Q25", 0.50: "Q50", 0.75: "Q75", 0.95: "Q95"}
EPS = 1e-12


@dataclass
class MSSComponents:
    step1: Any
    step2: Any
    q_std: dict[float, float]
    indep_cols: list[str]
    sample: pd.DataFrame


def annualize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse a monthly panel to annual frequency using within-year means."""
    work = panel.copy()
    if isinstance(work.index, pd.MultiIndex):
        work = work.reset_index()
    if "iso3" not in work.columns or "year" not in work.columns:
        raise ValueError("panel must contain 'iso3' and 'year' columns.")

    numeric_cols = [
        c for c in work.select_dtypes(include=[np.number]).columns.tolist() if c != "year"
    ]
    keep = ["iso3", "year"] + numeric_cols
    annual = (
        work[keep]
        .groupby(["iso3", "year"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["iso3", "year"])
    )
    return annual


def _conditioning_variables(panel_annual: pd.DataFrame) -> list[str]:
    excluded = {"year", "month", "hicp_lag"}
    return [
        c
        for c in panel_annual.columns
        if c not in {"iso3", "date"}
        and not c.startswith("hicp_fwd")
        and c not in excluded
        and pd.api.types.is_numeric_dtype(panel_annual[c])
    ]


def _stochastic_variables(panel_annual: pd.DataFrame) -> list[str]:
    candidates = ["hicp", "hicp_lag"] + _conditioning_variables(panel_annual)
    out = []
    for col in candidates:
        if col not in panel_annual.columns:
            continue
        vals = panel_annual[col].dropna()
        if len(vals) == 0:
            continue
        if vals.nunique() <= 2:
            continue
        out.append(col)
    return out


def _normalize_prediction_frame(results: Any) -> pd.DataFrame:
    """Return a flat quantile prediction frame."""
    if results is None:
        raise ValueError("results is required.")

    if isinstance(results, pd.DataFrame):
        frame = results.copy()
    elif isinstance(results, dict):
        flat_frames = []
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                tmp = value.copy()
                if "horizon" not in tmp.columns:
                    tmp["horizon"] = int(key)
                flat_frames.append(tmp)
            elif isinstance(value, dict):
                for source in ("pred_q_oos", "pred_q_is"):
                    pred = value.get(source)
                    if pred is None or not isinstance(pred, pd.DataFrame):
                        continue
                    tmp = pred.copy()
                    tmp = tmp.reset_index()
                    rename_map = {}
                    for tau, col in Q_COLS.items():
                        if tau in tmp.columns:
                            rename_map[tau] = col
                        elif str(tau) in tmp.columns:
                            rename_map[str(tau)] = col
                    tmp = tmp.rename(columns=rename_map)
                    tmp["horizon"] = int(key)
                    if "cond_var" not in tmp.columns:
                        tmp["cond_var"] = "pooled"
                    flat_frames.append(tmp)
                    break
        if not flat_frames:
            raise ValueError("results dict does not contain usable quantile predictions.")
        frame = pd.concat(flat_frames, ignore_index=True)
    else:
        raise TypeError("results must be a DataFrame or dict.")

    missing = {"iso3", "year", "horizon"} - set(frame.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")

    if "cond_var" not in frame.columns:
        frame["cond_var"] = "pooled"

    rename_map = {}
    for tau, col in Q_COLS.items():
        if tau in frame.columns:
            rename_map[tau] = col
        elif str(tau) in frame.columns:
            rename_map[str(tau)] = col
    frame = frame.rename(columns=rename_map)

    needed = list(Q_COLS.values())
    missing_q = [c for c in needed if c not in frame.columns]
    if missing_q:
        raise ValueError(f"results is missing quantile columns: {missing_q}")

    frame["year"] = frame["year"].astype(int)
    frame["horizon"] = frame["horizon"].astype(int)
    return frame[["iso3", "year", "horizon", "cond_var"] + needed].copy()


def _normalize_skt_params(skt_params: Any) -> pd.DataFrame:
    if skt_params is None:
        return pd.DataFrame(
            columns=["iso3", "year", "horizon", "cond_var", "xi", "omega", "alpha", "nu"]
        )

    if isinstance(skt_params, pd.DataFrame):
        frame = skt_params.copy()
    elif isinstance(skt_params, dict):
        frames = []
        for key, value in skt_params.items():
            tmp = value.copy()
            if "horizon" not in tmp.columns:
                tmp["horizon"] = int(key)
            frames.append(tmp)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        raise TypeError("skt_params must be a DataFrame or dict.")

    if frame.empty:
        return frame

    if "cond_var" not in frame.columns:
        frame["cond_var"] = "pooled"
    frame["year"] = frame["year"].astype(int)
    frame["horizon"] = frame["horizon"].astype(int)
    return frame


def _normalize_weights(weights: Any, horizons: tuple[int, ...] = (1, 2, 4)) -> pd.DataFrame:
    cols = list(horizons)
    if weights is None:
        return pd.DataFrame(columns=["iso3"] + cols)

    if isinstance(weights, dict):
        rows = []
        for iso3, values in weights.items():
            arr = np.asarray(values, dtype=float)
            if len(arr) != len(cols):
                continue
            rows.append({"iso3": iso3, **{h: arr[i] for i, h in enumerate(cols)}})
        frame = pd.DataFrame(rows)
    elif isinstance(weights, pd.DataFrame):
        frame = weights.copy()
        rename_map = {}
        for h in cols:
            if str(h) in frame.columns:
                rename_map[str(h)] = h
            if f"h{h}" in frame.columns:
                rename_map[f"h{h}"] = h
            if f"w_h{h}" in frame.columns:
                rename_map[f"w_h{h}"] = h
        frame = frame.rename(columns=rename_map)
    else:
        raise TypeError("weights must be a dict or DataFrame.")

    if frame.empty:
        return pd.DataFrame(columns=["iso3"] + cols)

    missing = {"iso3"} | set(cols)
    if not missing.issubset(frame.columns):
        raise ValueError(f"weights must contain columns {['iso3'] + cols}.")

    frame = frame[["iso3"] + cols].copy()
    frame[cols] = frame[cols].astype(float)
    frame[cols] = frame[cols].div(frame[cols].sum(axis=1).replace(0.0, np.nan), axis=0)
    return frame


def _fit_panel_ols(sample: pd.DataFrame, dep: str, indep_cols: list[str]) -> Any:
    formula = f"{dep} ~ 1 + {' + '.join(indep_cols)} + EntityEffects"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelOLS.from_formula(formula, data=sample, drop_absorbed=True)
        result = model.fit(cov_type="clustered", cluster_entity=True)
    return result


def _prepare_mss_sample(panel_annual: pd.DataFrame, cond_var: str, horizon: int) -> tuple[pd.DataFrame, list[str], str]:
    dep = f"hicp_fwd{horizon}"
    required = ["iso3", "year", "hicp_lag", dep]
    indep = ["hicp_lag"]
    if cond_var != "hicp_lag":
        required.append(cond_var)
        indep.append(cond_var)
    sample = panel_annual[required].dropna().copy()
    sample = sample.set_index(["iso3", "year"]).sort_index()
    return sample, indep, dep


def _fit_mss_components(panel_annual: pd.DataFrame, cond_var: str, horizon: int) -> MSSComponents | None:
    sample, indep_cols, dep = _prepare_mss_sample(panel_annual, cond_var, horizon)
    if len(sample) < 30:
        return None

    step1 = _fit_panel_ols(sample, dep, indep_cols)
    resid1 = step1.resids.squeeze()

    sample2 = sample.copy()
    sample2["__abs_resid__"] = np.abs(resid1.reindex(sample2.index))
    sample2 = sample2.dropna(subset=["__abs_resid__"])
    if len(sample2) < 20:
        return None

    step2 = _fit_panel_ols(sample2, "__abs_resid__", indep_cols)
    scale_hat = step2.fitted_values.squeeze().reindex(sample2.index).clip(lower=1e-6)
    common_idx = scale_hat.index.intersection(resid1.index)
    z = (resid1.reindex(common_idx) / scale_hat.reindex(common_idx)).dropna()
    if len(z) < 10:
        return None

    q_std = {tau: float(np.quantile(z, tau)) for tau in QUANTILES}
    return MSSComponents(step1=step1, step2=step2, q_std=q_std, indep_cols=indep_cols, sample=sample)


def _entity_effects(result: Any) -> pd.Series:
    effects = getattr(result, "estimated_effects", None)
    if effects is None or len(effects) == 0:
        return pd.Series(dtype=float)
    if isinstance(effects, pd.DataFrame):
        eff = effects.iloc[:, 0]
    else:
        eff = effects.squeeze()
    if isinstance(eff.index, pd.MultiIndex):
        eff = eff.groupby(level=0).mean()
    return eff.astype(float)


def _predict_fe(result: Any, data: pd.DataFrame, indep_cols: list[str]) -> pd.Series:
    params = result.params.copy()
    beta = params.reindex(["Intercept"] + indep_cols).fillna(0.0)
    entity_fe = _entity_effects(result)

    pred = pd.Series(0.0, index=data.index, dtype=float)
    pred += float(beta.get("Intercept", 0.0))
    for col in indep_cols:
        if col in data.columns:
            pred += float(beta.get(col, 0.0)) * data[col].astype(float)
    if not entity_fe.empty:
        pred += data.index.get_level_values(0).map(entity_fe).fillna(0.0).to_numpy()
    return pred


def _predict_quantiles(panel_annual: pd.DataFrame, cond_var: str, horizon: int, components: MSSComponents, years: list[int]) -> pd.DataFrame:
    sample, _, _ = _prepare_mss_sample(panel_annual, cond_var, horizon)
    pred_sample = sample[sample.index.get_level_values("year").isin(years)].copy()
    if pred_sample.empty:
        return pd.DataFrame()

    mu_hat = _predict_fe(components.step1, pred_sample, components.indep_cols)
    s_hat = _predict_fe(components.step2, pred_sample, components.indep_cols).clip(lower=1e-6)

    out = pred_sample.reset_index()[["iso3", "year"]].copy()
    out["horizon"] = horizon
    out["cond_var"] = cond_var
    for tau in QUANTILES:
        out[Q_COLS[tau]] = (mu_hat + components.q_std[tau] * s_hat).to_numpy()
    return out


def pesaran_cd_test(series: pd.Series) -> pd.Series:
    wide = series.unstack("iso3").sort_index()
    cols = [c for c in wide.columns if wide[c].notna().sum() >= 4]
    wide = wide[cols]

    corrs = []
    overlaps = []
    for i, ci in enumerate(wide.columns[:-1]):
        xi = wide[ci]
        for cj in wide.columns[i + 1 :]:
            pair = pd.concat([xi, wide[cj]], axis=1).dropna()
            if len(pair) < 4:
                continue
            if pair.iloc[:, 0].std(ddof=1) == 0 or pair.iloc[:, 1].std(ddof=1) == 0:
                continue
            rho = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if np.isfinite(rho):
                corrs.append(rho)
                overlaps.append(len(pair))

    corrs = np.asarray(corrs, dtype=float)
    overlaps = np.asarray(overlaps, dtype=float)
    if len(corrs) == 0:
        return pd.Series({"N_countries": wide.shape[1], "pairs_used": 0, "Tbar": np.nan, "mean_pair_corr": np.nan, "CD_stat": np.nan, "pvalue": np.nan})

    t_bar = overlaps.mean()
    cd_stat = np.sqrt(2.0 * t_bar / len(corrs)) * corrs.sum()
    pvalue = 2.0 * (1.0 - norm.cdf(abs(cd_stat)))
    return pd.Series(
        {
            "N_countries": wide.shape[1],
            "pairs_used": len(corrs),
            "Tbar": t_bar,
            "mean_pair_corr": corrs.mean(),
            "CD_stat": cd_stat,
            "pvalue": pvalue,
        }
    )


def ips_test(series: pd.Series, lags: int = 1, trend: str = "c", sims: int = 1000, seed: int = 20260416) -> pd.Series:
    rng = np.random.default_rng(seed)
    observed = []
    lengths = []
    for _, country_series in series.groupby(level="iso3"):
        y = country_series.droplevel("iso3").dropna().astype(float)
        if len(y) <= lags + 5:
            continue
        if y.std(ddof=1) == 0:
            continue
        observed.append(adfuller(y, maxlag=lags, regression=trend, autolag=None)[0])
        lengths.append(len(y))

    observed = np.asarray(observed, dtype=float)
    lengths = np.asarray(lengths, dtype=int)
    if len(observed) == 0:
        return pd.Series({"N_used": 0, "avg_adf_t": np.nan, "IPS_Z": np.nan, "pvalue": np.nan})

    cache: dict[int, tuple[float, float]] = {}

    def _null_moments(T: int) -> tuple[float, float]:
        if T in cache:
            return cache[T]
        sim_t = np.empty(sims)
        for r in range(sims):
            y0 = np.cumsum(rng.normal(size=T))
            sim_t[r] = adfuller(y0, maxlag=lags, regression=trend, autolag=None)[0]
        cache[T] = (sim_t.mean(), sim_t.var(ddof=1))
        return cache[T]

    mu = []
    var = []
    for T in lengths:
        null_mu, null_var = _null_moments(int(T))
        mu.append(null_mu)
        var.append(null_var)

    mu_arr = np.asarray(mu)
    var_arr = np.asarray(var)
    tbar = observed.mean()
    expected = mu_arr.mean()
    var_tbar = var_arr.sum() / (len(observed) ** 2)
    z = (tbar - expected) / np.sqrt(var_tbar)
    pvalue = norm.cdf(z)
    return pd.Series({"N_used": len(observed), "avg_adf_t": tbar, "IPS_Z": z, "pvalue": pvalue})


def _lag_block(panel_annual: pd.DataFrame, var: str, max_lag: int, include_l0: bool) -> pd.DataFrame:
    work = panel_annual.set_index(["iso3", "year"]).sort_index()
    group = work.groupby(level=0)[var]
    start = 0 if include_l0 else 1
    return pd.DataFrame(
        {f"{var}_l{lag}": group.shift(lag) for lag in range(start, max_lag + 1)},
        index=work.index,
    )


def _info_criteria(result: Any) -> tuple[float, float]:
    k = len(result.params)
    n = result.nobs
    return (-2.0 * result.loglik + 2.0 * k, -2.0 * result.loglik + np.log(n) * k)


def category1_panel_structure(panel: pd.DataFrame, key_regressors: list[str] | None = None, max_hicp_lag: int = 3, max_reg_lag: int = 2, ips_sims: int = 1000) -> dict[str, pd.DataFrame]:
    panel_annual = annualize_panel(panel)
    panel_idx = panel_annual.set_index(["iso3", "year"]).sort_index()

    conditioning_vars = _conditioning_variables(panel_annual)
    cross_dep = pd.DataFrame({col: pesaran_cd_test(panel_idx[col]) for col in ["hicp"] + conditioning_vars if col in panel_idx.columns}).T
    cross_dep.index.name = "variable"

    trend_map = {col: ("ct" if any(k in col for k in ("wui", "neer", "reer")) else "c") for col in _stochastic_variables(panel_annual)}
    stationarity = pd.DataFrame(
        {col: ips_test(panel_idx[col], lags=1, trend=trend_map[col], sims=ips_sims) for col in trend_map}
    ).T
    stationarity.index.name = "variable"

    if key_regressors is None:
        key_regressors = [c for c in conditioning_vars if c != "hicp"]

    hicp_rows = []
    reg_rows = []
    for horizon in HORIZONS:
        dep = f"hicp_fwd{horizon}"
        if dep not in panel_annual.columns:
            continue
        work = panel_annual.copy()
        work = work[["iso3", "year", "hicp", dep] + [c for c in key_regressors if c in work.columns]].copy()

        for p in range(1, max_hicp_lag + 1):
            X = _lag_block(work, "hicp", p, include_l0=False)
            sample = pd.concat([work.set_index(["iso3", "year"])[dep], X], axis=1).dropna()
            if len(sample) < 20:
                continue
            result = PanelOLS(sample[dep], sample.drop(columns=[dep]), entity_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
            aic, bic = _info_criteria(result)
            hicp_rows.append({"h": horizon, "hicp_lags": p, "nobs": result.nobs, "aic": aic, "bic": bic})

        if not hicp_rows:
            continue
        best_h = (
            pd.DataFrame([r for r in hicp_rows if r["h"] == horizon])
            .sort_values(["bic", "aic"])
            .iloc[0]["hicp_lags"]
        )
        for reg in key_regressors:
            if reg not in work.columns:
                continue
            X_h = _lag_block(work, "hicp", int(best_h), include_l0=False)
            for q in range(0, max_reg_lag + 1):
                X_r = _lag_block(work, reg, q, include_l0=True)
                sample = pd.concat([work.set_index(["iso3", "year"])[dep], X_h, X_r], axis=1).dropna()
                if len(sample) < 20:
                    continue
                result = PanelOLS(sample[dep], sample.drop(columns=[dep]), entity_effects=True, drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
                aic, bic = _info_criteria(result)
                reg_rows.append({"h": horizon, "variable": reg, "var_lags": q, "nobs": result.nobs, "aic": aic, "bic": bic})

    hicp_lags = pd.DataFrame(hicp_rows).sort_values(["h", "bic", "aic"]) if hicp_rows else pd.DataFrame()
    reg_lags = pd.DataFrame(reg_rows).sort_values(["h", "variable", "bic", "aic"]) if reg_rows else pd.DataFrame()
    return {"cross_sectional_dependence": cross_dep, "unit_root_ips": stationarity, "hicp_lag_selection": hicp_lags, "regressor_lag_selection": reg_lags}


def _tick_loss(y: Any, q: Any, tau: float) -> np.ndarray:
    err = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.where(err >= 0.0, tau * err, (tau - 1.0) * err)


def category2_mss_diagnostics(panel: pd.DataFrame, results: Any, oos_results: pd.DataFrame | None = None, tau_pairs: list[tuple[float, float]] | None = None) -> dict[str, pd.DataFrame]:
    if tau_pairs is None:
        tau_pairs = [(0.05, 0.95), (0.25, 0.75)]

    panel_annual = annualize_panel(panel)
    pred = _normalize_prediction_frame(results)
    actual_cols = ["iso3", "year"] + [c for c in panel_annual.columns if c.startswith("hicp_fwd")]
    merged = pred.merge(panel_annual[actual_cols], on=["iso3", "year"], how="left")

    pseudo_rows = []
    stability_rows = []
    serial_rows = []
    for (horizon, cond_var), grp in pred.groupby(["horizon", "cond_var"], sort=True):
        dep = f"hicp_fwd{horizon}"
        joined = merged[(merged["horizon"] == horizon) & (merged["cond_var"] == cond_var)].dropna(subset=[dep])
        if joined.empty:
            continue
        y = joined[dep].to_numpy()
        for tau in QUANTILES:
            col = Q_COLS[tau]
            model_loss = _tick_loss(y, joined[col].to_numpy(), tau).sum()
            null_q = np.quantile(y, tau)
            null_loss = _tick_loss(y, np.full(len(y), null_q), tau).sum()
            pseudo_rows.append(
                {
                    "h": horizon,
                    "cond_var": cond_var,
                    "tau": tau,
                    "pseudo_R2": 1.0 - model_loss / null_loss if null_loss > 0 else np.nan,
                    "model_loss": model_loss,
                    "null_loss": null_loss,
                }
            )

        components = _fit_mss_components(panel_annual, cond_var, horizon)
        if components is None:
            continue

        gamma = components.step2.params.reindex(components.indep_cols).dropna()
        cov_g = components.step2.cov.reindex(index=gamma.index, columns=gamma.index)
        if len(gamma) > 0:
            for tau_a, tau_b in tau_pairs:
                dq = components.q_std[tau_a] - components.q_std[tau_b]
                diff = dq * gamma.to_numpy()
                cov_diff = (dq ** 2) * cov_g.to_numpy()
                cov_inv = np.linalg.pinv(cov_diff)
                joint_stat = float(diff.T @ cov_inv @ diff)
                joint_p = float(chi2.sf(joint_stat, df=len(gamma)))
                for i, term in enumerate(gamma.index):
                    se = float(np.sqrt(max(cov_diff[i, i], 0.0)))
                    z = diff[i] / se if se > 0 else np.nan
                    p = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
                    stability_rows.append(
                        {
                            "h": horizon,
                            "cond_var": cond_var,
                            "tau_a": tau_a,
                            "tau_b": tau_b,
                            "term": term,
                            "coef_diff": diff[i],
                            "se": se,
                            "z": z,
                            "pvalue": p,
                            "joint_W": joint_stat,
                            "joint_pvalue": joint_p,
                        }
                    )

        resid = pd.Series(components.step1.resids).sort_index()
        diff_resid = resid.groupby(level=0).diff()
        diff_lag = diff_resid.groupby(level=0).shift(1)
        serial_df = pd.concat([diff_resid.rename("de"), diff_lag.rename("de_l1")], axis=1).dropna()
        if not serial_df.empty:
            model = sm.OLS(serial_df["de"], serial_df[["de_l1"]]).fit(
                cov_type="cluster",
                cov_kwds={"groups": serial_df.index.get_level_values(0)},
            )
            beta = float(model.params["de_l1"])
            se = float(model.bse["de_l1"])
            z = (beta + 0.5) / se if se > 0 else np.nan
            pvalue = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
            serial_rows.append({"h": horizon, "cond_var": cond_var, "beta_hat": beta, "se": se, "z_stat": z, "pvalue": pvalue, "n_obs": len(serial_df)})

    crossing_source = _normalize_prediction_frame(oos_results) if oos_results is not None else pred
    crossing_rows = []
    for (horizon, cond_var), grp in crossing_source.groupby(["horizon", "cond_var"], sort=True):
        arr = grp[[Q_COLS[t] for t in QUANTILES]].to_numpy()
        diffs = np.diff(arr, axis=1)
        crossing = (diffs < 0.0).any(axis=1)
        min_gap = np.minimum(diffs, 0.0).min(axis=1)
        crossing_rows.append(
            {
                "h": horizon,
                "cond_var": cond_var,
                "n_obs": len(grp),
                "n_crossing": int(crossing.sum()),
                "crossing_rate": float(crossing.mean()) if len(crossing) else np.nan,
                "worst_violation": float(min_gap.min()) if len(min_gap) else np.nan,
                "mean_violation_given_crossing": float(min_gap[crossing].mean()) if crossing.any() else 0.0,
            }
        )

    return {
        "pseudo_r2": pd.DataFrame(pseudo_rows).sort_values(["h", "cond_var", "tau"]),
        "coefficient_stability": pd.DataFrame(stability_rows).sort_values(["h", "cond_var", "tau_a", "tau_b", "term"]),
        "quantile_crossing": pd.DataFrame(crossing_rows).sort_values(["h", "cond_var"]),
        "serial_correlation": pd.DataFrame(serial_rows).sort_values(["h", "cond_var"]),
    }


def _fitted_quantiles_from_skt(skt_frame: pd.DataFrame) -> pd.DataFrame:
    out = skt_frame[["iso3", "year", "horizon", "cond_var"]].copy()
    for tau in QUANTILES:
        out[Q_COLS[tau]] = skt_frame.apply(lambda row: skt_quantile_from_params(tau, row), axis=1)
    return out


def category3_skewed_t_fit_quality(results: Any, skt_params: Any) -> dict[str, pd.DataFrame]:
    pred = _normalize_prediction_frame(results)
    skt = _normalize_skt_params(skt_params)
    if skt.empty:
        empty = pd.DataFrame()
        return {"distribution_tests": empty, "parameter_checks": empty, "bad_parameter_rows": empty, "maqe_by_tau": empty, "maqe_summary": empty}

    merged = pred.merge(skt, on=["iso3", "year", "horizon", "cond_var"], how="inner", suffixes=("_target", ""))
    if merged.empty:
        empty = pd.DataFrame()
        return {"distribution_tests": empty, "parameter_checks": empty, "bad_parameter_rows": empty, "maqe_by_tau": empty, "maqe_summary": empty}

    fit_q = _fitted_quantiles_from_skt(merged)
    joined = merged[["iso3", "year", "horizon", "cond_var"] + list(Q_COLS.values())].merge(
        fit_q,
        on=["iso3", "year", "horizon", "cond_var"],
        suffixes=("_target", "_fit"),
    )

    dist_rows = []
    for (horizon, cond_var), grp in joined.groupby(["horizon", "cond_var"], sort=True):
        for tau in QUANTILES:
            target = grp[f"{Q_COLS[tau]}_target"].dropna().to_numpy()
            fitted = grp[f"{Q_COLS[tau]}_fit"].dropna().to_numpy()
            if len(target) < 5 or len(fitted) < 5:
                continue
            ks = ks_2samp(target, fitted, alternative="two-sided", mode="auto")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ad = anderson_ksamp([target, fitted])
            dist_rows.append(
                {
                    "h": horizon,
                    "cond_var": cond_var,
                    "tau": tau,
                    "n": len(target),
                    "ks_stat": ks.statistic,
                    "ks_pvalue": ks.pvalue,
                    "ad_stat": ad.statistic,
                    "ad_pvalue": ad.significance_level / 100.0,
                }
            )

    checks = merged[["iso3", "year", "horizon", "cond_var", "xi", "omega", "alpha", "nu"]].copy()
    checks["sigma_ok"] = checks["omega"] > 0.0
    checks["nu_ok"] = checks["nu"] > 2.0
    checks["alpha_finite"] = np.isfinite(checks["alpha"])
    checks["converged_ok"] = True if "converged" not in merged.columns else merged["converged"].astype(bool)
    checks["all_ok"] = checks["sigma_ok"] & checks["nu_ok"] & checks["alpha_finite"] & checks["converged_ok"]
    summary = checks.groupby("horizon").agg(
        n_fits=("all_ok", "size"),
        n_fail_total=("all_ok", lambda s: int((~s).sum())),
        fail_rate=("all_ok", lambda s: float((~s).mean())),
        n_sigma_fail=("sigma_ok", lambda s: int((~s).sum())),
        n_nu_fail=("nu_ok", lambda s: int((~s).sum())),
        n_alpha_nonfinite=("alpha_finite", lambda s: int((~s).sum())),
        min_sigma=("omega", "min"),
        min_nu=("nu", "min"),
        max_abs_alpha=("alpha", lambda s: float(np.nanmax(np.abs(s)))),
    )

    abs_err_rows = []
    for tau in QUANTILES:
        col_t = f"{Q_COLS[tau]}_target"
        col_f = f"{Q_COLS[tau]}_fit"
        tmp = joined[["horizon", "cond_var", col_t, col_f]].copy()
        tmp["tau"] = tau
        tmp["abs_error"] = (tmp[col_f] - tmp[col_t]).abs()
        abs_err_rows.append(tmp[["horizon", "cond_var", "tau", "abs_error"]])
    abs_err = pd.concat(abs_err_rows, ignore_index=True)
    maqe_by_tau = (
        abs_err.groupby(["horizon", "cond_var", "tau"], as_index=False)["abs_error"]
        .mean()
        .rename(columns={"horizon": "h", "abs_error": "MAQE"})
        .sort_values(["h", "cond_var", "tau"])
    )
    maqe_summary = abs_err.groupby(["horizon", "cond_var"])["abs_error"].agg(["mean", "median", "max", lambda s: s.quantile(0.95)]).reset_index()
    maqe_summary = maqe_summary.rename(columns={"horizon": "h", "<lambda_0>": "p95"})

    return {
        "distribution_tests": pd.DataFrame(dist_rows).sort_values(["h", "cond_var", "tau"]),
        "parameter_checks": summary,
        "bad_parameter_rows": checks.loc[~checks["all_ok"]],
        "maqe_by_tau": maqe_by_tau,
        "maqe_summary": maqe_summary,
    }


def _skt_pdf(x: Any, xi: Any, omega: Any, alpha: Any, nu: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    omega = np.maximum(np.asarray(omega, dtype=float), 1e-6)
    gamma = np.exp(np.asarray(alpha, dtype=float))
    z = (x - xi) / omega
    c = 2.0 / (omega * (gamma + 1.0 / gamma))
    left = z < 0
    out = np.empty_like(z, dtype=float)
    out[left] = c[left] * (student_t.pdf(-z[left] / gamma[left], df=np.asarray(nu, dtype=float)[left]) / gamma[left])
    out[~left] = c[~left] * (student_t.pdf(z[~left] * gamma[~left], df=np.asarray(nu, dtype=float)[~left]) * gamma[~left])
    return np.maximum(out, EPS)


def _skt_cdf_scalar(x: float, xi: float, omega: float, alpha: float, nu: float) -> float:
    gamma = np.exp(alpha)
    z = (x - xi) / max(omega, 1e-6)
    if z < 0:
        return student_t.cdf(-z / gamma, df=nu) / (1.0 + gamma)
    return 1.0 / (1.0 + gamma) + student_t.cdf(z * gamma, df=nu) * gamma / (1.0 + gamma)


def build_density_scores(panel: pd.DataFrame, skt_params: Any, weights: Any) -> pd.DataFrame:
    panel_annual = annualize_panel(panel)
    skt = _normalize_skt_params(skt_params)
    w = _normalize_weights(weights)
    if skt.empty or w.empty:
        return pd.DataFrame()

    params = (
        skt.groupby(["iso3", "year", "horizon"], as_index=False)[["xi", "omega", "alpha", "nu"]]
        .mean()
        .sort_values(["iso3", "year", "horizon"])
    )

    wide = params.pivot(index=["iso3", "year"], columns="horizon", values=["xi", "omega", "alpha", "nu"])
    wide.columns = [f"{name}_h{h}" for name, h in wide.columns]
    wide = wide.reset_index()

    actual = panel_annual[["iso3", "year"] + [c for c in panel_annual.columns if c.startswith("hicp_fwd")]].copy()
    score = actual.merge(wide, on=["iso3", "year"], how="inner").merge(w, on="iso3", how="left")
    if score.empty:
        return score

    for h in HORIZONS:
        dep = f"hicp_fwd{h}"
        if dep not in score.columns or f"xi_h{h}" not in score.columns:
            continue
        score[f"pdf_h{h}"] = _skt_pdf(score[dep], score[f"xi_h{h}"], score[f"omega_h{h}"], score[f"alpha_h{h}"], score[f"nu_h{h}"])
        score[f"cdf_h{h}"] = [
            _skt_cdf_scalar(y, xi, omega, alpha, nu)
            for y, xi, omega, alpha, nu in zip(score[dep], score[f"xi_h{h}"], score[f"omega_h{h}"], score[f"alpha_h{h}"], score[f"nu_h{h}"])
        ]
        score[f"logscore_h{h}"] = np.log(np.clip(score[f"pdf_h{h}"], EPS, None))

    available_h = [h for h in HORIZONS if f"pdf_h{h}" in score.columns]
    if not available_h:
        return pd.DataFrame()

    weight_sum = score[available_h].sum(axis=1).replace(0.0, np.nan)
    for h in available_h:
        score[h] = score[h] / weight_sum

    score["pdf_pooled"] = sum(score[h] * score[f"pdf_h{h}"] for h in available_h)
    score["cdf_pooled"] = sum(score[h] * score[f"cdf_h{h}"] for h in available_h)
    score["logscore_pooled"] = np.log(np.clip(score["pdf_pooled"], EPS, None))
    return score.sort_values(["iso3", "year"]).reset_index(drop=True)


def _is_density_score_frame(frame: Any) -> bool:
    return isinstance(frame, pd.DataFrame) and {"logscore_pooled", "cdf_pooled"}.issubset(frame.columns)


def _newey_west_variance(x: pd.Series, lag: int | None = None) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr - arr.mean()
    T = len(arr)
    if T <= 1:
        return np.nan
    if lag is None:
        lag = int(np.floor(1.5 * T ** (1 / 3)))
    gamma0 = np.dot(arr, arr) / T
    var = gamma0
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1.0)
        gamma_l = np.dot(arr[l:], arr[:-l]) / T
        var += 2.0 * weight * gamma_l
    return var / T


def category4_density_pooling_validation(panel: pd.DataFrame, skt_params: Any, weights: Any, pooled_scores: pd.DataFrame | None = None) -> dict[str, pd.DataFrame | pd.Series]:
    weight_frame = _normalize_weights(weights)
    if pooled_scores is None or not _is_density_score_frame(pooled_scores):
        pooled_scores = build_density_scores(panel, skt_params, weights)

    if weight_frame.empty or pooled_scores.empty:
        empty = pd.DataFrame()
        return {"weight_summary": empty, "weight_dominance": pd.Series(dtype=float), "weight_by_country": empty, "diebold_mariano": empty, "pit_summary": pd.Series(dtype=float), "pit_histogram": empty, "density_scores": pooled_scores}

    available_h = [h for h in HORIZONS if h in weight_frame.columns and f"logscore_h{h}" in pooled_scores.columns]
    obs_summary = pd.DataFrame(index=weight_frame["iso3"])
    obs_summary["max_weight"] = weight_frame[available_h].max(axis=1).to_numpy()
    obs_summary["argmax_h"] = weight_frame[available_h].idxmax(axis=1).to_numpy()
    obs_summary["hhi"] = (weight_frame[available_h] ** 2).sum(axis=1).to_numpy()
    obs_summary["entropy"] = (-(weight_frame[available_h] * np.log(np.clip(weight_frame[available_h], EPS, None))).sum(axis=1) / np.log(len(available_h))).to_numpy()
    obs_summary["effective_horizons"] = 1.0 / obs_summary["hhi"]
    obs_summary.index.name = "iso3"

    dominance = pd.Series(
        {
            "n_obs": len(weight_frame),
            "share_max_weight_gt_0_80": float((obs_summary["max_weight"] > 0.80).mean()),
            "share_max_weight_gt_0_90": float((obs_summary["max_weight"] > 0.90).mean()),
            "mean_max_weight": float(obs_summary["max_weight"].mean()),
            "median_max_weight": float(obs_summary["max_weight"].median()),
            "mean_hhi": float(obs_summary["hhi"].mean()),
            "mean_entropy": float(obs_summary["entropy"].mean()),
            "mean_effective_horizons": float(obs_summary["effective_horizons"].mean()),
        }
    )

    dm_rows = []
    for h in available_h:
        diff = (pooled_scores["logscore_pooled"] - pooled_scores[f"logscore_h{h}"]).dropna()
        if len(diff) < 5:
            continue
        var_d = _newey_west_variance(diff, lag=max(1, h - 1))
        dm_stat = float(diff.mean() / np.sqrt(var_d)) if var_d and np.isfinite(var_d) and var_d > 0 else np.nan
        dm_rows.append(
            {
                "against_h": h,
                "n_obs": len(diff),
                "mean_logscore_gain": float(diff.mean()),
                "dm_stat": dm_stat,
                "pvalue_two_sided": float(2.0 * norm.sf(abs(dm_stat))) if np.isfinite(dm_stat) else np.nan,
                "pvalue_pooled_better": float(norm.sf(-dm_stat)) if np.isfinite(dm_stat) else np.nan,
            }
        )

    pit = pooled_scores["cdf_pooled"].dropna().clip(EPS, 1.0 - EPS)
    counts, edges = np.histogram(pit, bins=10, range=(0.0, 1.0))
    chi_stat, chi_p = chisquare(counts, np.full(10, len(pit) / 10.0))
    pit_summary = pd.Series(
        {
            "n_obs": len(pit),
            "mean_pit": float(pit.mean()),
            "var_pit": float(pit.var(ddof=1)),
            "uniform_mean_benchmark": 0.5,
            "uniform_var_benchmark": 1.0 / 12.0,
            "chi2_uniform_stat": float(chi_stat),
            "chi2_uniform_pvalue": float(chi_p),
        }
    )
    pit_hist = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "count": counts, "share": counts / counts.sum() if counts.sum() else np.nan})

    return {
        "weight_summary": weight_frame.set_index("iso3")[available_h].agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "h"}),
        "weight_dominance": dominance,
        "weight_by_country": obs_summary.reset_index(),
        "diebold_mariano": pd.DataFrame(dm_rows).sort_values("against_h"),
        "pit_summary": pit_summary,
        "pit_histogram": pit_hist,
        "density_scores": pooled_scores,
    }


def build_expanding_window_predictions(panel: pd.DataFrame, cond_vars: list[str] | None = None, horizons: list[int] | None = None, eval_start: int = 2015, eval_end: int = 2025) -> pd.DataFrame:
    panel_annual = annualize_panel(panel)
    if cond_vars is None:
        cond_vars = ["hicp_lag"] + _conditioning_variables(panel_annual)
    if horizons is None:
        horizons = HORIZONS

    rows = []
    for horizon in horizons:
        for cond_var in cond_vars:
            sample, indep_cols, dep = _prepare_mss_sample(panel_annual, cond_var, horizon)
            years_all = sorted(sample.index.get_level_values("year").unique())
            for forecast_year in range(eval_start, eval_end + 1):
                train = sample[sample.index.get_level_values("year") <= forecast_year - horizon].copy()
                pred = sample[sample.index.get_level_values("year") == forecast_year].copy()
                if len(train) < 30 or pred.empty:
                    continue
                try:
                    step1 = _fit_panel_ols(train, dep, indep_cols)
                    resid1 = step1.resids.squeeze()
                    train2 = train.copy()
                    train2["__abs_resid__"] = np.abs(resid1.reindex(train2.index))
                    train2 = train2.dropna(subset=["__abs_resid__"])
                    if len(train2) < 20:
                        continue
                    step2 = _fit_panel_ols(train2, "__abs_resid__", indep_cols)
                    scale_hat = step2.fitted_values.squeeze().reindex(train2.index).clip(lower=1e-6)
                    common_idx = scale_hat.index.intersection(resid1.index)
                    z = (resid1.reindex(common_idx) / scale_hat.reindex(common_idx)).dropna()
                    if len(z) < 10:
                        continue
                    q_std = {tau: float(np.quantile(z, tau)) for tau in QUANTILES}
                    comps = MSSComponents(step1=step1, step2=step2, q_std=q_std, indep_cols=indep_cols, sample=train)
                    pred_frame = _predict_quantiles(panel_annual, cond_var, horizon, comps, [forecast_year])
                    if not pred_frame.empty:
                        rows.append(pred_frame)
                except Exception:
                    continue

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["iso3", "year", "horizon", "cond_var"] + list(Q_COLS.values()))


def category5_out_of_sample_forecasting(panel: pd.DataFrame, results: Any | None = None, oos_results: pd.DataFrame | None = None, eval_start: int = 2015, eval_end: int = 2025) -> dict[str, pd.DataFrame]:
    panel_annual = annualize_panel(panel)
    if oos_results is None:
        if results is not None:
            oos_results = _normalize_prediction_frame(results)
            oos_results = oos_results[oos_results["year"].between(eval_start, eval_end)]
        else:
            oos_results = build_expanding_window_predictions(panel, eval_start=eval_start, eval_end=eval_end)
    else:
        oos_results = _normalize_prediction_frame(oos_results)

    eval_rows = []
    qs_rows = []
    cov_rows = []
    miss_rows = []
    if oos_results.empty:
        empty = pd.DataFrame()
        return {"evaluation_window": empty, "quantile_scores": empty, "coverage_90": empty, "interval_miss_decomposition": empty, "oos_predictions": empty}

    actual = panel_annual[["iso3", "year"] + [c for c in panel_annual.columns if c.startswith("hicp_fwd")]].copy()
    merged = oos_results.merge(actual, on=["iso3", "year"], how="left")

    for (horizon, cond_var), grp in merged.groupby(["horizon", "cond_var"], sort=True):
        dep = f"hicp_fwd{horizon}"
        sub = grp.dropna(subset=[dep]).copy()
        if sub.empty:
            continue
        eval_rows.append(
            {
                "h": horizon,
                "cond_var": cond_var,
                "n_obs": len(sub),
                "countries": sub["iso3"].nunique(),
                "start_year": int(sub["year"].min()),
                "end_year": int(sub["year"].max()),
            }
        )
        y = sub[dep].to_numpy()
        for tau in (0.05, 0.95):
            loss = _tick_loss(y, sub[Q_COLS[tau]].to_numpy(), tau)
            qs_rows.append(
                {
                    "h": horizon,
                    "cond_var": cond_var,
                    "tau": tau,
                    "n_obs": len(sub),
                    "mean_tick_loss": float(np.mean(loss)),
                    "median_tick_loss": float(np.median(loss)),
                    "total_tick_loss": float(np.sum(loss)),
                }
            )
        inside = ((sub[dep] >= sub["Q05"]) & (sub[dep] <= sub["Q95"])).astype(int)
        bt = binomtest(int(inside.sum()), n=len(inside), p=0.90, alternative="two-sided")
        cov_rows.append(
            {
                "h": horizon,
                "cond_var": cond_var,
                "n_obs": len(inside),
                "hits": int(inside.sum()),
                "coverage_rate": float(inside.mean()),
                "target_rate": 0.90,
                "coverage_error": float(inside.mean() - 0.90),
                "binom_pvalue": float(bt.pvalue),
            }
        )
        miss_rows.append(
            {
                "h": horizon,
                "cond_var": cond_var,
                "below_p05_rate": float((sub[dep] < sub["Q05"]).mean()),
                "above_p95_rate": float((sub[dep] > sub["Q95"]).mean()),
                "total_outside_rate": float(((sub[dep] < sub["Q05"]) | (sub[dep] > sub["Q95"])).mean()),
            }
        )

    return {
        "evaluation_window": pd.DataFrame(eval_rows).sort_values(["h", "cond_var"]),
        "quantile_scores": pd.DataFrame(qs_rows).sort_values(["h", "cond_var", "tau"]),
        "coverage_90": pd.DataFrame(cov_rows).sort_values(["h", "cond_var"]),
        "interval_miss_decomposition": pd.DataFrame(miss_rows).sort_values(["h", "cond_var"]),
        "oos_predictions": oos_results,
    }


def category6_deanchoring_logit(daresults: Any) -> dict[str, pd.DataFrame | pd.Series]:
    if daresults is None:
        empty = pd.DataFrame()
        return {"fit_metrics": pd.Series(dtype=float), "separation_check": pd.Series(dtype=object), "separation_symptoms": pd.Series(dtype=float), "hosmer_lemeshow": pd.Series(dtype=float), "hosmer_lemeshow_table": empty, "marginal_effect_upside_risk": pd.Series(dtype=float), "all_marginal_effects": empty}

    y = np.asarray(daresults.model.endog, dtype=float)
    X = pd.DataFrame(daresults.model.exog, columns=daresults.model.exog_names)
    p_hat = np.asarray(daresults.predict(X), dtype=float)

    fit_metrics = pd.Series(
        {
            "n_obs": len(y),
            "pseudo_R2_mcfadden": float(daresults.prsquared),
            "auc_roc": float(roc_auc_score(y, p_hat)),
            "brier_score": float(brier_score_loss(y, p_hat)),
            "brier_naive": float(brier_score_loss(y, np.full_like(y, y.mean(), dtype=float))),
            "loglik_model": float(daresults.llf),
            "loglik_null": float(daresults.llnull),
            "lr_stat": float(2.0 * (daresults.llf - daresults.llnull)),
            "lr_pvalue": float(chi2.sf(2.0 * (daresults.llf - daresults.llnull), df=int(daresults.df_model))),
        }
    )

    messages = []
    separated = False
    converged = None
    refit = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            refit = sm.Logit(y, X).fit(disp=False, maxiter=200)
        converged = bool(refit.mle_retvals.get("converged", True))
        for warning_msg in caught:
            msg = str(warning_msg.message)
            messages.append(msg)
            if "separation" in msg.lower():
                separated = True
    except PerfectSeparationError as exc:
        separated = True
        messages.append(str(exc))
    except Exception as exc:
        messages.append(str(exc))

    sep_check = pd.Series({"separation_flag": separated, "converged": converged, "messages": messages})
    sep_symptoms = pd.Series(
        {
            "share_p_lt_1e_6": float((p_hat < 1e-6).mean()),
            "share_p_gt_1_1e_6": float((p_hat > 1 - 1e-6).mean()),
            "max_abs_coef": float(np.max(np.abs(daresults.params))),
            "max_std_err": float(np.max(daresults.bse)),
            "converged": bool(daresults.mle_retvals.get("converged", True)),
        }
    )

    df = pd.DataFrame({"y": y, "p": np.clip(p_hat, 1e-8, 1 - 1e-8)}).sort_values("p")
    df["group"] = pd.qcut(df["p"], q=10, duplicates="drop")
    hl_table = df.groupby("group", observed=False).agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"), p_mean=("p", "mean"))
    hl_table["obs_non"] = hl_table["n"] - hl_table["observed"]
    hl_table["exp_non"] = hl_table["n"] - hl_table["expected"]
    hl_stat = (
        ((hl_table["observed"] - hl_table["expected"]) ** 2 / hl_table["expected"].clip(lower=1e-8))
        + ((hl_table["obs_non"] - hl_table["exp_non"]) ** 2 / hl_table["exp_non"].clip(lower=1e-8))
    ).sum()
    hl_summary = pd.Series({"groups_used": len(hl_table), "hl_stat": float(hl_stat), "df": int(len(hl_table) - 2), "pvalue": float(chi2.sf(hl_stat, len(hl_table) - 2))})

    margeff = daresults.get_margeff(at="mean").summary_frame()
    upside_effect = margeff.loc["Upside_risk"] if "Upside_risk" in margeff.index else pd.Series(dtype=float)
    return {
        "fit_metrics": fit_metrics,
        "separation_check": sep_check,
        "separation_symptoms": sep_symptoms,
        "hosmer_lemeshow": hl_summary,
        "hosmer_lemeshow_table": hl_table.reset_index(),
        "marginal_effect_upside_risk": upside_effect,
        "all_marginal_effects": margeff.reset_index().rename(columns={"index": "term"}),
    }


def run_validation_suite(panel: pd.DataFrame, results: Any, skt_params: Any | None = None, weights: Any | None = None, iar: pd.DataFrame | None = None, pooled_scores: pd.DataFrame | None = None, daresults: Any | None = None, oos_results: pd.DataFrame | None = None, ips_sims: int = 1000) -> dict[str, Any]:
    """
    Run the full six-category validation suite.

    Parameters match the notebook variable names so the suite can be called
    directly after the pipeline finishes.
    """
    out = {
        "category1": category1_panel_structure(panel, ips_sims=ips_sims),
        "category2": category2_mss_diagnostics(panel, results, oos_results=oos_results),
        "category3": category3_skewed_t_fit_quality(results, skt_params),
        "category4": category4_density_pooling_validation(panel, skt_params, weights, pooled_scores=pooled_scores),
        "category5": category5_out_of_sample_forecasting(panel, results=results, oos_results=oos_results),
        "category6": category6_deanchoring_logit(daresults),
    }
    if iar is not None:
        out["iar"] = iar.copy()
    return out
