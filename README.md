# EU G4 Inflation-at-Risk (IaR)

A Python implementation of the **Inflation-at-Risk** framework for the four largest EU economies (France, Germany, Italy, Spain), applying the **Machado–Santos Silva (2019) location-scale quantile regression** methodology.

---

## Deliverable

A single Jupyter notebook:

```
notebooks/eu_g4_iar.ipynb
```

When executed, it runs the full pipeline and produces inline charts plus a summary results table.

---

## Methodology

| Phase | Step | Detail |
|-------|------|--------|
| 1 | **Data** | 11 conditioning series for 29 EU/EEA countries; 1999–2025 |
| 2 | **MSS regression** | Three-step location-scale quantile estimator; τ ∈ {0.05, 0.25, 0.50, 0.75, 0.95}; h ∈ {1, 2, 4} years |
| 3 | **Distribution fit** | Fernández-Steel skewed-t fitted analytically to five quantile predictions per country-year |
| 4 | **Pooling** | Log-score density pooling across conditioning variables (Crump *et al.* 2022) |
| 5 | **IaR extraction** | P95 from pooled distribution; median re-centred to ECB March 2026 staff projection for 2027 |
| 6 | **De-anchoring** | Panel logit for P(HICP > 3% × 2 consecutive years); G4 scores for 2025/2026 |

**Dependent variable**: Rolling-mean forward annual HICP:

$$\widehat{\pi}_{i,t}^{(h)} = \frac{1}{h}\sum_{k=1}^{h} \pi_{i,t+k}$$

**IaR definition**:

$$\text{IaR}_{i,t}^{(h)} = Q_{0.95}\bigl(\pi_{i,t}^{(h)} \mid X_{i,t}\bigr)$$

**Upside / Downside asymmetry**:

$$\text{Upside} = Q_{0.95} - Q_{0.50}, \qquad \text{Downside} = Q_{0.50} - Q_{0.05}$$

---

## Data Sources

| Variable | Source | Endpoint / Series |
|----------|--------|-------------------|
| HICP inflation | Eurostat SDMX 3.0 API | `prc_hicp_minr`, TOTAL, RCH_A (monthly y/y %) |
| Inflation expectations | OECD Economic Outlook (primary) | SDMX-JSON, `EO`, `CPIH_YTYPCT`, annual; 27 countries |
| | ECB Consumer Expectations Survey (G4 2020+) | `ECB_CES1`, `INFL_NEXT12M.MEAN`, monthly averaged |
| | Compiled fallback | CYP and MLT only (Eurostat actuals + EC forecasts) |
| Output gap | OECD Economic Outlook | SDMX-JSON, `EO`, `GAP`, annual |
| Energy (Brent) | FRED / St. Louis Fed | `DCOILBRENTEU` (daily ICE Brent, USD/bbl → annual % change) |
| Import prices | ECB Statistical Data Warehouse (primary) | `MNA` dataflow, import deflator (SDMX 2.1) |
| | OECD Economic Outlook (fallback) | `EO`, import price measure |
| | EC AMECO (compiled fallback) | `PMIM` variable |
| Food prices | Eurostat SDMX 3.0 API | `prc_fsc_idx` (monthly food price index) |
| Labour costs | Eurostat SDMX 3.0 API | `lc_lci_r2_q` (quarterly labour cost index) |
| Effective exchange rate | EC DG ECFIN | `m42neer.xlsx` (NEER) / `m42reer.xlsx` (REER) |
| Financial stress | ECB CLIFS | `CLIFS` dataflow, annual average |
| Sovereign spread | Eurostat SDMX 2.1 API | `IRT_LT_MCBY_M` (monthly 10Y Maastricht bond yields) |
| Uncertainty | WUI Project | World Uncertainty Index (Ahir, Bloom, Furceri 2023), Excel download |

All series include compiled fallback tables so the pipeline runs fully offline.

---

## Conditioning Variables

```python
COND_VARS = {
    "hicp_lag":           "Lagged HICP (AR persistence)",
    "output_gap":         "% of potential GDP (OECD EO)",
    "infl_expectations":  "1-year-ahead HICP forecast (OECD EO / ECB CES)",
    "energy_price_chg":   "Brent crude % change (FRED DCOILBRENTEU)",
    "import_price_chg":   "Import deflator % change (ECB SDW MNA / OECD EO)",
    "food_price_chg":     "Food price index % change (Eurostat prc_fsc_idx)",
    "labour_cost_chg":    "Labour cost index % change (Eurostat lc_lci_r2_q)",
    "neer_chg":           "Nominal effective exchange rate % change (EC ECFIN)",
    "clifs":              "ECB CLIFS financial stress index",
    "spread_10y":         "10Y Maastricht yield (Eurostat IRT_LT_MCBY_M)",
    "wui":                "World Uncertainty Index (Ahir, Bloom, Furceri 2023)",
}
```

---

## Re-centring Anchor

The median of the pooled predictive distribution is shifted to the country-specific **ECB March 2026 staff projection for 2027 HICP** (sourced from the `infl_expectations` panel where available; below values used as fallback).

| Country | ECB baseline 2027 |
|---------|------------------|
| France | 2.0% |
| Germany | 2.2% |
| Italy | 1.9% |
| Spain | 2.1% |

---

## De-anchoring Signal

Binary outcome: `deanchor = 1` if **both** π_{t+1} > 3% **and** π_{t+2} > 3%.
Panel logit: P(deanchor) = Λ(β₀ + β₁ · Upside_risk)

Interpretation: a country with larger upside risk (wider right tail) is more likely to sustain inflation above target.

---

## Project Structure

```
inflation_at_risk/
├── data/
│   ├── modules/
│   │   ├── hicp.py           # Eurostat HICP (dep. variable; prc_hicp_minr)
│   │   ├── ameco.py          # Inflation expectations (OECD EO / ECB CES / fallback)
│   │   ├── output_gap.py     # OECD EO output gap (GAP measure)
│   │   ├── energy_prices.py  # Brent crude % change (FRED DCOILBRENTEU)
│   │   ├── import_prices.py  # Import deflator % change (ECB SDW MNA / OECD EO)
│   │   ├── food_prices.py    # Food price index % change (Eurostat prc_fsc_idx)
│   │   ├── labour_costs.py   # Labour cost index % change (Eurostat lc_lci_r2_q)
│   │   ├── EER.py            # NEER/REER (EC ECFIN m42neer.xlsx / m42reer.xlsx)
│   │   ├── imf_fsi.py        # ECB CLIFS financial stress index
│   │   ├── ecb_spreads.py    # 10Y Maastricht bond yields (Eurostat IRT_LT_MCBY_M)
│   │   └── wui.py            # World Uncertainty Index
│   ├── files/                # Compiled fallback parquet/ver caches
│   └── panel_builder.py      # Merge & forward variable construction (monthly panel)
├── model/
│   ├── location_scale.py     # MSS three-step estimator (IaR version)
│   └── quantile_fit.py       # Fernández-Steel skewed-t fitting
├── risk/
│   ├── pooling.py            # Log-score density pooling
│   └── iar.py                # IaR extraction + ECB baseline re-centring
├── crisis/
│   └── deanchoring_signal.py # De-anchoring early-warning logit
├── output/
│   └── charts.py             # Publication-quality charts
├── notebooks/
│   └── eu_g4_iar.ipynb       # Primary deliverable
├── validation.py             # Econometric validation suite
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook
cd notebooks
jupyter lab eu_g4_iar.ipynb
```

All data fetches have compiled fallback tables — the pipeline will execute correctly with no internet connection.

---

## Econometric validation suite

The repository now includes a reusable validation module for the full IaR pipeline:

```python
from validation import run_validation_suite

validation = run_validation_suite(
    panel=panel,
    results=results,
    skt_params=skt_params,
    weights=weights,
    iar=iar,
    pooled_scores=pooled_scores,
    daresults=daresults,
)
```

The returned object is a nested dictionary keyed by `category1` through `category6`,
with publication-ready tables for:

- panel structure
- MSS diagnostics
- skewed-t fit quality
- density pooling validation
- out-of-sample forecasting performance
- de-anchoring logit diagnostics

The validation functions accept either the notebook-style in-memory objects or the
repo's cached flat parquet outputs.

---

## References

- López-Salido, D. & Loria, F. (2024). *Inflation at Risk*. **Journal of Monetary Economics**, 105569.
- Korobilis, D., Landau, B., Musso, A. & Phella, A. (2021). *The Time-Varying Evolution of Inflation Risks*. ECB Working Paper No. 2591.
- Banerjee, R., Mehrotra, A. & Zampolli, F. (2024). *Inflation at Risk in Advanced and Emerging Economies*. **Journal of International Money and Finance**, 105813.
- Furceri, D., Loungani, P., Ostry, J. & Pizzuto, P. (2025). *Inflation-at-Risk*. IMF Working Paper WP/25/86.
- Crump, R., Eusepi, S., Giannoni, M. & Sahin, A. (2022). *The Unemployment–Inflation Trade-off Revisited*. NBER Working Paper 29186.
- Machado, J.A.F. & Santos Silva, J.M.C. (2019). *Quantiles via Moments*. **Journal of Econometrics**, 213(1), 145–173.
- Fernández, C. & Steel, M.F.J. (1998). *On Bayesian Modeling of Fat Tails and Skewness*. **JASA**, 93(441), 359–371.
