"""
Visualizations for the Sequence Model findings.
Updated for reclassified hold signals (3-way: portfolio / hold-rec / no-hold).
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
BASE = _ROOT / "data"
FIGS = _ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

C_PORT = "#2166ac"     # Portfolio (cramer_owns)
C_REC = "#7fbc41"      # Hold recommendation
C_NOHOLD = "#b2182b"   # No hold
C_NEUTRAL = "#636363"


def load():
    df = pd.read_csv(BASE / "cramer_sequenced.csv")
    seq = pd.read_csv(BASE / "sequence_summary.csv")
    sens = pd.read_csv(BASE / "sensitivity_results.csv")
    return df, seq, sens


def seq_category(row):
    if row.get("has_cramer_owns", False):
        return "portfolio"
    elif row.get("has_hold_recommendation", False):
        return "hold_rec"
    return "no_hold"


def vix_regime(v):
    if pd.isna(v): return None
    if v < 20: return "Low"
    elif v < 30: return "Moderate"
    return "High"


def ttest_alpha(series):
    s = series.dropna()
    if len(s) < 5:
        return (np.nan, np.nan, len(s), np.nan)
    t, p = stats.ttest_1samp(s, 0)
    se = float(s.std(ddof=1)) / np.sqrt(len(s))
    return (s.mean(), p, len(s), se)


CAT_COLORS = {"portfolio": C_PORT, "hold_rec": C_REC, "no_hold": C_NOHOLD}
CAT_LABELS = {"portfolio": "Portfolio pick\n(Cramer owns)", "hold_rec": "Hold-recommendation\n(advisory)", "no_hold": "Casual buy\n(no ownership)"}
CAT_SHORT = {"portfolio": "Portfolio pick", "hold_rec": "Hold-recommendation", "no_hold": "Casual buy"}


# ═══════════════════════════════════════════════════════════════════════
# Fig 1: Core finding — Three-way entry alpha comparison
# ═══════════════════════════════════════════════════════════════════════

def fig1_core_finding(df, seq):
    seq["seq_cat"] = seq.apply(seq_category, axis=1)
    d = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    d = d.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
    entries = d[d["is_sequence_start"]]

    categories = []
    for cat in ["portfolio", "hold_rec", "no_hold"]:
        sub = entries[entries["seq_cat"] == cat]["spy_relative_return"].dropna()
        categories.append((CAT_LABELS[cat], sub, CAT_COLORS[cat]))

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    # Tight title placement (caption in the document carries interpretation)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.28)

    bar_values = []
    for i, (label, data, color) in enumerate(categories):
        m, p, n, se = ttest_alpha(data)
        bar_values.append((m * 100, p, n, se * 100))
        ax.bar(i, m * 100, color=color, width=0.6, alpha=0.85,
               edgecolor="white", linewidth=1.5)
        # ±1.96·SE whisker (95% normal-approx CI on the per-position alpha mean).
        ax.errorbar(i, m * 100, yerr=1.96 * se * 100, fmt="none",
                    ecolor="black", capsize=4, linewidth=1.0, capthick=1.0)

    whisker_hi = [m + 1.96 * se for m, _, _, se in bar_values]
    whisker_lo = [m - 1.96 * se for m, _, _, se in bar_values]
    y_top = max(whisker_hi)
    y_bot = min(whisker_lo)
    # y limits: room above upper whisker; below lower whisker for 2-line label (va=top: text hangs
    # downward ~0.5–0.7 %-units — need extra pad so N= line clears x-axis spine / y=0 region in export)
    pad_top = 0.48
    pad_bot = 2.00
    ax.set_ylim(y_bot - pad_bot, y_top + pad_top)

    for i, (m, p, n, se) in enumerate(bar_values):
        stars = ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"

        label_txt = f"{m:+.2f}%{stars}\n(N={n})"
        w_top = m + 1.96 * se
        w_bot = m - 1.96 * se
        if m >= 0:
            ax.text(
                i,
                w_top + 0.1,
                label_txt,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="black",
                clip_on=False,
                zorder=20,
            )
        else:
            # Below lower whisker; anchor top of block with room for 2 lines + margin above axis line
            ax.text(
                i,
                w_bot - 0.38,
                label_txt,
                ha="center",
                va="top",
                fontsize=8.5,
                fontweight="bold",
                color="black",
                clip_on=False,
                zorder=20,
            )

    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c[0] for c in categories])
    ax.set_ylabel("SPY-relative alpha (%)")
    ax.set_title("First-of-sequence α by engagement tier (1Y)", pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, min_n_ticks=5))
    # Upper right: avoid overlap with the casual-buy lower whisker / label
    ax.text(
        0.99,
        0.98,
        "* p<0.05  ** p<0.01  *** p<0.001  ·  whiskers = ±1.96·SE",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="gray",
    )

    fig.savefig(FIGS / "fig1_hold_vs_nohold.png", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close()
    print("  Saved fig1_hold_vs_nohold.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 2: Sensitivity analysis — parameter stability
# ═══════════════════════════════════════════════════════════════════════

def fig2_sensitivity(sens):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Sweep hold_window
    hw_data = sens[(sens["non_hold_window"] == 60) & (sens["prune_gap"] == 60) &
                    (sens["prune_price"] == 0.10)].sort_values("hold_window")
    ax = axes[0]
    ax.plot(hw_data["hold_window"], hw_data["hold_entry_alpha"], "o-",
            color=C_PORT, label="Portfolio pick", linewidth=2, markersize=7)
    ax.plot(hw_data["hold_window"], hw_data["nohold_entry_alpha"], "s-",
            color=C_NOHOLD, label="Casual buy", linewidth=2, markersize=7)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("hold_window (days)")
    ax.set_ylabel("First-recommendation alpha (%)")
    ax.set_title("A. Portfolio-window sweep")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Sweep non_hold_window
    nhw_data = sens[(sens["hold_window"] == 120) & (sens["prune_gap"] == 60) &
                     (sens["prune_price"] == 0.10)].sort_values("non_hold_window")
    ax = axes[1]
    ax.plot(nhw_data["non_hold_window"], nhw_data["nohold_entry_alpha"], "s-",
            color=C_NOHOLD, label="Casual buy", linewidth=2, markersize=7)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("non_hold_window (days)")
    ax.set_title("B. Casual-buy-window sweep")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Casual-buy p-values across all 17 parameter combinations
    ax = axes[2]
    pvals = [r for r in sens["nohold_p"].values]
    colors = [C_NOHOLD if p < 0.05 else "#d9d9d9" for p in pvals]
    ax.bar(range(len(pvals)), [-np.log10(p) if p > 0 else 5 for p in pvals],
           color=colors, alpha=0.7)
    ax.axhline(-np.log10(0.05), color="black", linewidth=1, linestyle="--", label="p=0.05")
    ax.set_xlabel("Parameter combination (1 of 17)")
    ax.set_ylabel("-log10(p)")
    ax.set_title("C. Casual-buy p-values across all 17 sweeps")
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Sensitivity analysis: casual-buy underperformance stable across sequence-model parameters",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_sensitivity.png")
    plt.close()
    print("  Saved fig2_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 4: VIX regime × 3-way category
# ═══════════════════════════════════════════════════════════════════════

def fig4_vix_hold(df, seq):
    seq["seq_cat"] = seq.apply(seq_category, axis=1)
    d = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    d["vix_regime"] = d["vix_at_signal"].apply(vix_regime)
    d = d.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
    entries = d[d["is_sequence_start"]]

    vix_order = ["Low", "Moderate", "High"]
    cats = ["portfolio", "hold_rec", "no_hold"]
    y_cap, y_floor = 34, -24
    fig, ax = plt.subplots(figsize=(9.2, 3.4))

    bar_width = 0.25
    x = np.arange(len(vix_order))
    all_upper, all_lower = [], []

    for i, cat in enumerate(cats):
        alphas_v, ns_v, ps_v, ses_v = [], [], [], []
        for vix in vix_order:
            sub = entries[(entries["seq_cat"] == cat) & (entries["vix_regime"] == vix)]
            m, p, n, se = ttest_alpha(sub["spy_relative_return"])
            alphas_v.append(m * 100 if not np.isnan(m) else 0)
            ns_v.append(n)
            ps_v.append(p)
            ses_v.append(se * 100 if not np.isnan(se) else 0)

        offset = (i - 1) * bar_width
        whisker_yerr = [1.96 * s if n >= 10 else 0 for s, n in zip(ses_v, ns_v)]
        for a, w in zip(alphas_v, whisker_yerr):
            all_upper.append(a + w)
            all_lower.append(a - w)
        ax.bar(x + offset, alphas_v, bar_width, color=CAT_COLORS[cat],
               alpha=0.85, label=CAT_SHORT[cat], edgecolor="white", linewidth=1.0)
        ax.errorbar(x + offset, alphas_v, yerr=whisker_yerr, fmt="none",
                    ecolor="black", capsize=2.5, linewidth=0.75, capthick=0.75)

        for j, (a, n, p) in enumerate(zip(alphas_v, ns_v, ps_v)):
            stars = ""
            if p < 0.001: stars = "***"
            elif p < 0.01: stars = "**"
            elif p < 0.05: stars = "*"
            yoff = 0.3 if a > 0 else -0.3
            va = "bottom" if a > 0 else "top"
            if n >= 10:
                ax.text(x[j] + offset, a + yoff, f"{a:+.1f}%{stars}\n({n})",
                        ha="center", va=va, fontsize=6.5)

    ax.axhline(0, color="black", linewidth=0.8)
    pad = 1.0
    top = min(max(all_upper) + pad, y_cap)
    bottom = max(min(all_lower) - pad, y_floor)
    ax.set_ylim(bottom, top)
    ax.set_xticks(x)
    ax.set_xticklabels(["VIX <20", "VIX 20–30", "VIX >30"], fontsize=9)
    ax.set_ylabel("Entry alpha (SPY-rel., %)", fontsize=9.5)
    ax.legend(fontsize=7, ncol=3, loc="upper left", framealpha=0.95,
              columnspacing=0.6, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.2)
    fig.savefig(FIGS / "fig4_vix_hold.png", dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print("  Saved fig4_vix_hold.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 5: By-horizon performance (3-way)
# ═══════════════════════════════════════════════════════════════════════

def fig5_horizons(df, seq):
    seq["seq_cat"] = seq.apply(seq_category, axis=1)
    horizons = ["1m", "3m", "6m", "1y"]
    horizon_labels = ["1 Month", "3 Months", "6 Months", "1 Year"]
    cats = ["portfolio", "hold_rec", "no_hold"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_width = 0.25
    x = np.arange(len(horizons))

    for i, cat in enumerate(cats):
        alphas_h, ps_h, ses_h, ns_h = [], [], [], []
        for hz in horizons:
            dhz = df[(df["requested_horizon"] == hz) & (df["kept"] == True)].copy()
            dhz = dhz.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
            entries_hz = dhz[dhz["is_sequence_start"]]
            sub = entries_hz[entries_hz["seq_cat"] == cat]
            m, p, n, se = ttest_alpha(sub["annualized_spy_relative_return"])
            alphas_h.append(m * 100 if not np.isnan(m) else 0)
            ps_h.append(p)
            ses_h.append(se * 100 if not np.isnan(se) else 0)
            ns_h.append(n)

        offset = (i - 1) * bar_width
        ax.bar(x + offset, alphas_h, bar_width, color=CAT_COLORS[cat],
               alpha=0.85, label=CAT_SHORT[cat], edgecolor="white", linewidth=1.5)
        whisker_yerr = [1.96 * s if n >= 10 else 0 for s, n in zip(ses_h, ns_h)]
        ax.errorbar(x + offset, alphas_h, yerr=whisker_yerr, fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8, capthick=0.8)

        for j, (a, p) in enumerate(zip(alphas_h, ps_h)):
            stars = ""
            if p < 0.001: stars = "***"
            elif p < 0.01: stars = "**"
            elif p < 0.05: stars = "*"
            yoff = 0.15 if a > 0 else -0.15
            va = "bottom" if a > 0 else "top"
            ax.text(x[j] + offset, a + yoff, f"{a:+.1f}%{stars}",
                    ha="center", va=va, fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(horizon_labels)
    ax.set_ylabel("Entry alpha (annualized SPY-relative, %)")
    ax.set_title("Entry Performance by Engagement Tier Across Horizons (Annualized)")
    ax.legend(fontsize=9)
    ax.text(0.98, 0.02, "* p<0.05  ** p<0.01  *** p<0.001  ·  whiskers = ±1.96·SE",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGS / "fig5_horizons.png")
    plt.close()
    print("  Saved fig5_horizons.png")


# ═══════════════════════════════════════════════════════════════════════
# Fig 6: B-H heatmap — all 45 hypothesis cells, survivors highlighted
# ═══════════════════════════════════════════════════════════════════════

def fig6_bh_heatmap(bh):
    import matplotlib.patches as mpatches

    ANN = {"1w": 52, "1m": 12, "3m": 4, "6m": 2, "1y": 1}
    horizon_order = ["1w", "1m", "3m", "6m", "1y"]
    horizon_labels = ["1W", "1M", "3M", "6M", "1Y"]
    vix_order = ["low", "moderate", "high"]
    seg_order = ["Monologue/Top of Show", "Lightning Round", "Mid-Show/Interview"]
    seg_labels = {"Monologue/Top of Show": "Monologue",
                  "Lightning Round": "Lightning Round",
                  "Mid-Show/Interview": "Mid-Show"}

    rows = [(v, s) for v in vix_order for s in seg_order]
    n_rows = len(rows)
    n_cols = len(horizon_order)

    alpha_mat = np.zeros((n_rows, n_cols))
    reject_mat = np.zeros((n_rows, n_cols), dtype=bool)
    for i, (v, s) in enumerate(rows):
        for j, h in enumerate(horizon_order):
            row = bh[(bh["vix"] == v) & (bh["segment"] == s) & (bh["horizon"] == h)]
            if len(row) == 1:
                alpha_mat[i, j] = row["mean_alpha"].iloc[0] * ANN[h] * 100
                reject_mat[i, j] = bool(row["reject_tp"].iloc[0])

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    vmax = np.ceil(np.max(np.abs(alpha_mat)) / 5) * 5
    im = ax.imshow(alpha_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = alpha_mat[i, j]
            survivor = reject_mat[i, j]
            color = "white" if abs(val) > vmax * 0.55 else "black"
            weight = "bold" if survivor else "normal"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight=weight)
            if survivor:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=3.0))

    for i in [3, 6]:
        ax.axhline(i - 0.5, color="black", linewidth=1.5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(horizon_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    # Segment names only; VIX is drawn separately in the left gutter. Placing
    # VIX at the block *center row* (y=1,4,7) sat on the same index as
    # "Lightning Round" and collided. Use the vertical midway between
    # Monologue and Lightning (0.5, 3.5, 6.5) so labels clear all three rows.
    ax.set_yticklabels(
        [seg_labels[s] for v, s in rows],
        fontsize=9,
    )

    vix_brief = {
        "low": "Low VIX (<20)",
        "moderate": "Mod VIX (20–30)",
        "high": "High VIX (≥30)",
    }
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for vix, yc in zip(vix_order, [0.5, 3.5, 6.5]):
        ax.text(
            -0.20,
            yc,
            vix_brief[vix],
            transform=trans,
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xlabel("Forward horizon", fontsize=10, labelpad=6)

    cbar = plt.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("Ann. α (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.34, right=0.92, top=0.98, bottom=0.1)
    fig.savefig(
        FIGS / "fig6_bh_heatmap.png",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()
    print("  Saved fig6_bh_heatmap.png")


def main():
    print("Loading data...")
    df, seq, sens = load()
    print(f"  Signals: {len(df)}, Sequences: {len(seq)}, Sensitivity: {len(sens)} rows")

    print("\nGenerating figures...")
    fig1_core_finding(df, seq)
    fig2_sensitivity(sens)
    fig4_vix_hold(df, seq)
    fig5_horizons(df, seq)

    bh_path = BASE / "bh_grid.csv"
    if bh_path.exists():
        bh = pd.read_csv(bh_path)
        fig6_bh_heatmap(bh)
    else:
        raise SystemExit(
            f"Missing {bh_path.name}; run scripts/replicate_headline_tables.py "
            "before scripts/visualize_sequence.py."
        )

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
