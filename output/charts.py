"""
Chart generation for EU G4 Inflation-at-Risk presentation.

Charts produced:
  1. fig1_fan_charts.png    — 2×2 fan charts per G4 country: historical HICP
                              + P5/P50/P95 projection cone to 2028, ECB 2% reference
  2. fig2_asymmetry.png     — Upside (P95−P50) vs Downside (P50−P5) bar chart at h=2
  3. fig3_waterfall.png     — Upside risk decomposition by driver, per G4 country
  4. fig4_deanchoring.png   — De-anchoring probability scores for 2025 and 2026

All charts saved to output/ as .png (300 dpi).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY       = "#1B2A4A"
GOLD       = "#C8A951"
LIGHT_BLUE = "#7BAFD4"
RED        = "#C0392B"
GREEN      = "#27AE60"
GREY       = "#BDC3C7"
WHITE      = "#FFFFFF"
ORANGE     = "#E67E22"
PURPLE     = "#8E44AD"

G4_LABELS = {"FRA": "France", "DEU": "Germany", "ITA": "Italy", "ESP": "Spain"}
G4_ORDER  = ["FRA", "DEU", "ITA", "ESP"]

# ── Distinct colours for waterfall drivers (no two the same) ─────────────────
DRIVER_COLORS = {
    "hicp_lag":           "#E74C3C",   # red
    "output_gap":         "#2980B9",   # blue
    "infl_expectations":  "#27AE60",   # green
    "energy_price_chg":   "#F39C12",   # amber
    "import_price_chg":   "#8E44AD",   # purple
    "clifs":             "#16A085",   # teal
    "spread_10y":         "#D35400",   # burnt orange
    "wui":                "#2C3E50",   # dark slate
}

DRIVER_LABELS = {
    "hicp_lag":           "Lagged\nHICP",
    "output_gap":         "Output\nGap",
    "infl_expectations":  "Inflation\nExpect.",
    "energy_price_chg":   "Energy\nPrices",
    "import_price_chg":   "Import\nPrices",
    "clifs":             "Financial\nStress",
    "spread_10y":         "Sovereign\nSpread",
    "wui":                "Uncertainty\n(WUI)",
}

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.size":            10,
    "axes.titlesize":       12,
    "axes.titleweight":     "bold",
    "axes.labelsize":       10,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.edgecolor":       NAVY,
    "xtick.color":          NAVY,
    "ytick.color":          NAVY,
    "text.color":           NAVY,
    "figure.facecolor":     WHITE,
    "axes.facecolor":       WHITE,
})


# ──────────────────────────────────────────────────────────────────────────────
# 1. FAN CHARTS
# ──────────────────────────────────────────────────────────────────────────────

def fan_charts(
    panel: pd.DataFrame,
    iar: pd.DataFrame,
    horizon: int = 2,
    hist_start: int = 1999,
    proj_end: int = 2028,
) -> Path:
    """
    2×2 grid of HICP fan charts, one per G4 country.
    Historical HICP as solid line, P5/P50/P95 fan cone, ECB 2% target reference.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "EU G4 — HICP Inflation: Historical & Inflation-at-Risk Fan (2024–2028)",
        fontsize=14, fontweight="bold", color=NAVY, y=1.01,
    )

    for ax, iso3 in zip(axes.flat, G4_ORDER):
        # Historical
        hist = panel[
            (panel["iso3"] == iso3) & (panel["year"] >= hist_start)
        ].sort_values(["year", "month"])

        # Aggregate to annual (take last month of each year)
        hist_annual = hist.groupby("year")["hicp"].last().reset_index()

        ax.plot(hist_annual["year"], hist_annual["hicp"],
                color=NAVY, lw=2.0, label="Historical HICP", zorder=5)

        # IaR projection
        iar_row = iar[iar["iso3"] == iso3]
        if iar_row.empty:
            ax.set_title(G4_LABELS[iso3])
            continue

        dr = iar_row.iloc[0]

        # Anchor year: last historical observation
        anchor_year = int(hist_annual["year"].max()) if not hist_annual.empty else 2024
        anchor_hicp = float(hist_annual[hist_annual["year"] == anchor_year]["hicp"].values[0]) \
                      if not hist_annual.empty else dr["Q50"]

        proj_years = list(range(anchor_year, proj_end + 1))

        # Build fan via linear interpolation from anchor to projected quantiles
        n = len(proj_years)
        y_q05 = np.linspace(anchor_hicp, dr["Q05"], n)
        y_q50 = np.linspace(anchor_hicp, dr["Q50"], n)
        y_q95 = np.linspace(anchor_hicp, dr["Q95"], n)

        # Shaded fans
        ax.fill_between(proj_years, y_q05, y_q95,
                        color=GOLD, alpha=0.20, label="P5–P95 range", zorder=2)
        ax.fill_between(proj_years, y_q05, y_q50,
                        color=LIGHT_BLUE, alpha=0.25, label="P5–P50", zorder=3)

        ax.plot(proj_years, y_q50, color=GOLD,  lw=2.0, ls="--",
                label="Median P50", zorder=4)
        ax.plot(proj_years, y_q95, color=RED,   lw=1.5, ls=":",
                label="IaR P95", zorder=4)
        ax.plot(proj_years, y_q05, color=LIGHT_BLUE, lw=1.0, ls=":",
                zorder=4)

        # ECB 2% target reference line
        ax.axhline(2.0, color=GREEN, lw=1.2, ls="-.", alpha=0.85,
                   label="ECB 2% target")
        ax.text(hist_start + 1, 2.15, "ECB 2%", fontsize=7, color=GREEN, alpha=0.9)

        # Vertical divider at current year
        ax.axvline(anchor_year, color=GREY, lw=1.0, ls="--", zorder=1)
        ax.text(anchor_year + 0.1,
                ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 8.5,
                "Projection →", fontsize=7, color=GREY)

        # ECB baseline marker
        ax.axhline(dr["ecb_baseline"], color=ORANGE, lw=1.0, ls=":",
                   alpha=0.8, label=f"ECB baseline {dr['ecb_baseline']:.1f}%")

        ax.set_title(G4_LABELS[iso3], color=NAVY)
        ax.set_xlabel("Year")
        ax.set_ylabel("HICP % change")
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        ax.annotate(
            f"IaR P95: {dr['IaR']:.1f}%",
            xy=(proj_years[-1], y_q95[-1]),
            xytext=(-50, 8), textcoords="offset points",
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
        )

    out_path = OUTPUT_DIR / "fig1_fan_charts.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 2. ASYMMETRY BAR CHART
# ──────────────────────────────────────────────────────────────────────────────

def asymmetry_bar(iar: pd.DataFrame) -> Path:
    """
    Grouped bar chart: Upside (P95−P50) vs Downside (P50−P5) for each G4 country at h=2.
    Upside in red, downside in blue.
    Key board visual — shows whether distribution tilts toward inflation or deflation risk.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_title(
        "EU G4 — Inflation Risk Asymmetry at h=2 Years: Upside vs Downside",
        fontweight="bold", color=NAVY,
    )

    x      = np.arange(len(G4_ORDER))
    width  = 0.35
    labels = [G4_LABELS[c] for c in G4_ORDER]

    upside_vals   = []
    downside_vals = []
    for iso3 in G4_ORDER:
        row = iar[iar["iso3"] == iso3]
        if row.empty:
            upside_vals.append(0.0)
            downside_vals.append(0.0)
        else:
            r = row.iloc[0]
            upside_vals.append(float(r["Upside"]))
            downside_vals.append(float(r["Downside"]))

    bars_up   = ax.bar(x - width / 2, upside_vals,   width,
                       label="Upside risk (P95−P50)",   color=RED,        alpha=0.85,
                       edgecolor=NAVY, linewidth=0.5)
    bars_down = ax.bar(x + width / 2, downside_vals, width,
                       label="Downside risk (P50−P5)", color=LIGHT_BLUE, alpha=0.85,
                       edgecolor=NAVY, linewidth=0.5)

    # Value labels
    for bar in bars_up:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=RED, fontweight="bold")
    for bar in bars_down:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=NAVY, fontsize=11)
    ax.set_ylabel("Percentage points")
    ax.set_xlabel("")
    ax.legend(framealpha=0.8, loc="upper right")
    ax.set_ylim(0, max(max(upside_vals + downside_vals, default=4.0) * 1.3, 2.0))

    # Symmetry reference line
    avg_upside = np.mean(upside_vals) if upside_vals else 1.0
    ax.axhline(avg_upside, color=RED, lw=0.8, ls=":", alpha=0.5)
    ax.text(len(G4_ORDER) - 0.5, avg_upside + 0.05,
            f"Avg upside: {avg_upside:.2f}pp", fontsize=7, color=RED, ha="right")

    out_path = OUTPUT_DIR / "fig2_asymmetry.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 3. WATERFALL CHART — upside decomposition by driver
# ──────────────────────────────────────────────────────────────────────────────

def waterfall_charts(iar: pd.DataFrame) -> Path:
    """
    2×2 waterfall charts — decomposition of upside inflation risk (P95−P50)
    by driver for each G4 country.
    Each driver uses a DISTINCT colour (no two drivers share a colour).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(
        "EU G4 — Decomposition of Upside Inflation Risk by Driver (h=2)",
        fontsize=14, fontweight="bold", color=NAVY, y=1.01,
    )

    for ax, iso3 in zip(axes.flat, G4_ORDER):
        row = iar[iar["iso3"] == iso3]
        if row.empty:
            ax.set_title(G4_LABELS[iso3])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color=GREY)
            continue

        r = row.iloc[0]
        values, labels, colors = [], [], []

        for d_key, d_label in DRIVER_LABELS.items():
            col = f"upside_{d_key}"
            if col in r.index and pd.notna(r[col]) and r[col] > 0.005:
                values.append(float(r[col]))
                labels.append(d_label)
                colors.append(DRIVER_COLORS[d_key])

        if not values:
            ax.set_title(G4_LABELS[iso3])
            ax.text(0.5, 0.5, "No driver decomp", ha="center", va="center",
                    transform=ax.transAxes, color=GREY)
            continue

        # Sort descending
        order  = np.argsort(values)[::-1]
        values = [values[i] for i in order]
        labels = [labels[i] for i in order]
        colors = [colors[i] for i in order]

        # Cumulative running total for waterfall placement
        running = [0.0]
        for v in values:
            running.append(running[-1] + v)

        # Individual driver bars (waterfall style)
        for i, (v, c) in enumerate(zip(values, colors)):
            ax.bar(i, v, bottom=running[i], color=c, edgecolor=NAVY,
                   linewidth=0.5, alpha=0.88)
            ax.text(i, running[i] + v / 2, f"+{v:.2f}",
                    ha="center", va="center", fontsize=7.5, color=WHITE,
                    fontweight="bold")

        # Total bar (distinct dark red)
        total = sum(values)
        ax.bar(len(values), total, color=NAVY, edgecolor=NAVY,
               linewidth=0.8, alpha=0.92)
        ax.text(len(values), total / 2, f"{total:.2f}",
                ha="center", va="center", fontsize=9, color=WHITE, fontweight="bold")

        x_labels = labels + ["Total\nUpside"]
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=7.5, color=NAVY)
        ax.set_ylabel("pp HICP")
        ax.set_title(
            f"{G4_LABELS[iso3]} — Upside: {total:.2f} pp | P50={r['Q50']:.1f}% | IaR={r['IaR']:.1f}%",
            color=NAVY, fontsize=10,
        )

        # Colour legend patch per driver
        patches = [mpatches.Patch(color=DRIVER_COLORS[k], label=DRIVER_LABELS[k].replace("\n", " "))
                   for k in DRIVER_LABELS if f"upside_{k}" in r.index and pd.notna(r[f"upside_{k}"])
                   and r[f"upside_{k}"] > 0.005]
        if patches:
            ax.legend(handles=patches, fontsize=6.5, loc="upper right",
                      framealpha=0.8, ncol=2)

    out_path = OUTPUT_DIR / "fig3_waterfall.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# 4. DE-ANCHORING SIGNAL CHART
# ──────────────────────────────────────────────────────────────────────────────

def deanchoring_chart(pooled_scores: pd.DataFrame, base_rate: float = 0.20) -> Path:
    """
    Horizontal bar chart of G4 de-anchoring probability scores for 2026 and 2027.
    Sorted by probability. Red dashed line at historical base rate.
    """
    if pooled_scores.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No de-anchoring scores available",
                ha="center", va="center", transform=ax.transAxes)
        out_path = OUTPUT_DIR / "fig4_deanchoring.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        "EU G4 — De-anchoring Probability (HICP > 3% for 2 consecutive years)",
        fontsize=13, fontweight="bold", color=NAVY,
    )

    year_colors = {2026: RED, 2027: NAVY}

    for ax, year in zip(axes, [2026, 2027]):
        sub = pooled_scores[pooled_scores["year"] == year].copy()
        if sub.empty:
            ax.set_title(f"{year}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Sort by probability descending
        sub = sub.sort_values("pooled_prob", ascending=True)
        countries = [G4_LABELS.get(c, c) for c in sub["iso3"]]
        probs     = sub["pooled_prob"].values

        color = year_colors.get(year, NAVY)
        bars  = ax.barh(countries, probs,
                        color=color, alpha=0.80, edgecolor=NAVY, linewidth=0.5)

        # Value labels
        for bar, p in zip(bars, probs):
            ax.text(p + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1%}", va="center", fontsize=9, color=NAVY)

        # Base rate reference
        ax.axvline(base_rate, color=RED, lw=1.5, ls="--", alpha=0.8)
        ax.text(base_rate + 0.005, 0.05,
                f"Base rate\n{base_rate:.0%}", fontsize=7.5, color=RED,
                transform=ax.get_xaxis_transform())

        ax.set_title(f"{year} Projection", color=NAVY, fontweight="bold")
        ax.set_xlabel("De-anchoring probability")
        ax.set_xlim(0, min(max(probs.max() * 1.4, base_rate * 2.5, 0.5), 1.0))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    out_path = OUTPUT_DIR / "fig4_deanchoring.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Master chart generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_charts(
    panel: pd.DataFrame,
    iar: pd.DataFrame,
    pooled_scores: pd.DataFrame,
    base_rate: float = 0.20,
    horizon: int = 2,
) -> dict[str, Path]:
    """
    Generate all four IaR charts. Returns dict of chart_name → Path.
    """
    print("\nGenerating IaR charts …")
    paths = {}

    print("  Chart 1: Fan charts …")
    paths["fan"]          = fan_charts(panel, iar, horizon=horizon)

    print("  Chart 2: Asymmetry bar …")
    paths["asymmetry"]    = asymmetry_bar(iar)

    print("  Chart 3: Waterfall decomposition …")
    paths["waterfall"]    = waterfall_charts(iar)

    print("  Chart 4: De-anchoring signal …")
    paths["deanchoring"]  = deanchoring_chart(pooled_scores, base_rate=base_rate)

    print(f"  All charts saved to {OUTPUT_DIR}/")
    return paths


if __name__ == "__main__":
    from data.panel_builder   import load_panel
    from risk.iar             import load_iar
    from crisis.deanchoring_signal import load_deanchoring_scores

    panel   = load_panel()
    iar     = load_iar()
    _, pooled = load_deanchoring_scores()
    generate_all_charts(panel, iar, pooled)
