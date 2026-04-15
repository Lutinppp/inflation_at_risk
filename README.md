# EU G4 Inflation-at-Risk (IaR)

A Python implementation of the **Inflation-at-Risk** framework for the four largest EU economies (France, Germany, Italy, Spain), applying the same **Machado–Santos Silva (2019) location-scale quantile regression** methodology as the companion EU G4 Debt-at-Risk (`debt_at_risk`) project.

---

## Deliverable

A single Jupyter notebook:

```
notebooks/eu_g4_iar.ipynb
```

When executed, it runs the full five-phase pipeline and produces four inline charts plus a summary results table.

---

## Methodology

| Phase | Step | Detail |
|-------|------|--------|
| 1 | **Data** | 8 conditioning series for 29 EU/EEA countries; 1999–2025 |
| 2 | **MSS regression** | Three-step location-scale quantile estimator; τ ∈ {0.05, 0.25, 0.50, 0.75, 0.95}; h ∈ {1, 2, 4} years |
| 3 | **Distribution fit** | Fernández-Steel skewed-t fitted analytically to five quantile predictions per country-year |
| 4 | **Pooling** | Log-score density pooling across horizons (Crump *et al.* 2022) |
| 5 | **IaR extraction** | P95 from pooled distribution; median re-centred to ECB/AMECO 2027 country baseline |
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
| HICP inflation | Eurostat REST API | `prc_hicp_aind`, CP00, RCH_A |
| Inflation expectations | EC AMECO | ZCPIH variable; G4 only |
| Output gap | IMF DataMapper | NGAP_NPGDP |
| Energy (Brent) | IMF DataMapper | POILBRE_USD % change |
| Import prices | IMF DataMapper | PMPI / TM % change |
| Financial stress | ECB CLIFS | Annual average of CLIFS index |
| Sovereign spread | Eurostat | 10Y Maastricht bond yields (irt_lt_mcby_a) |
| Uncertainty | WUI Project | World Uncertainty Index (Excel download) |

All series include compiled fallback tables so the pipeline runs fully offline.

---

## Conditioning Variables

```python
COND_VARS = {
    "hicp_lag":          "Lagged HICP (AR persistence)",
    "output_gap":        "% of potential GDP (IMF)",
    "infl_expectations": "AMECO ZCPIH forecast",
    "energy_price_chg":  "Brent crude % change",
    "import_price_chg":  "Import deflator % change",
    "fsi":               "ECB CLIFS financial stress index",
    "spread_10y":        "10Y Maastricht yield",
    "wui":               "World Uncertainty Index",
}
```

---

## Re-centring Anchor

The median of the pooled predictive distribution is shifted to the country-specific ECB/AMECO **2027 HICP forecast** (ZCPIH variable), analogous to how the DaR project re-centres to the IMF WEO 2027 debt/GDP path.

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
│   ├── hicp.py               # Eurostat HICP (dep. variable)
│   ├── ameco.py              # EC AMECO inflation expectations
│   ├── output_gap.py         # IMF output gap
│   ├── energy_prices.py      # Brent crude % change
│   ├── import_prices.py      # Import deflator % change
│   ├── imf_fsi.py            # ECB CLIFS financial stress
│   ├── ecb_spreads.py        # 10Y Maastricht bond yields
│   ├── wui.py                # World Uncertainty Index
│   └── panel_builder.py      # Merge & forward variable construction
├── model/
│   ├── location_scale.py     # MSS three-step estimator (IaR version)
│   └── quantile_fit.py       # Fernández-Steel skewed-t fitting
├── risk/
│   ├── pooling.py            # Log-score density pooling
│   └── iar.py                # IaR extraction + driver decomposition
├── crisis/
│   └── deanchoring_signal.py # De-anchoring early-warning logit
├── output/
│   └── charts.py             # Four publication-quality charts
├── notebooks/
│   └── eu_g4_iar.ipynb       # Primary deliverable
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

## References

- López-Salido, D. & Loria, F. (2024). *Inflation at Risk*. **Journal of Monetary Economics**, 105569.
- Korobilis, D., Landau, B., Musso, A. & Phella, A. (2021). *The Time-Varying Evolution of Inflation Risks*. ECB Working Paper No. 2591.
- Banerjee, R., Mehrotra, A. & Zampolli, F. (2024). *Inflation at Risk in Advanced and Emerging Economies*. **Journal of International Money and Finance**, 105813.
- Furceri, D., Loungani, P., Ostry, J. & Pizzuto, P. (2025). *Inflation-at-Risk*. IMF Working Paper WP/25/86.
- Crump, R., Eusepi, S., Giannoni, M. & Sahin, A. (2022). *The Unemployment–Inflation Trade-off Revisited*. NBER Working Paper 29186.
- Machado, J.A.F. & Santos Silva, J.M.C. (2019). *Quantiles via Moments*. **Journal of Econometrics**, 213(1), 145–173.
- Fernández, C. & Steel, M.F.J. (1998). *On Bayesian Modeling of Fat Tails and Skewness*. **JASA**, 93(441), 359–371.

