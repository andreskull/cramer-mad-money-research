"""
Generate figures/fig_funnel.png — the four-feature funnel that opens Section 6.

Starting from all 4,964 first recommendations at the 1-year horizon, three
signal-time filters compound into the strongest in-sample cell:

    1. Engagement tier = casual buy
    2. Market cap < $2B (small-cap)
    3. VIX < 30 at signal time

Each row reports N, mean SPY-relative return, and a t-test p-value vs zero.
The bottom row is annotated with the equivalent market-neutral pair-trade P&L.
Box width scales as sqrt(N) relative to the top row (with a small floor) so
the funnel is visible. Explanatory text lives in the paper, not the figure.

Reads:
    data/cramer_sequenced.csv
    data/sequence_summary.csv
    data/market_caps.csv

Writes:
    figures/fig_funnel.png
"""

import sys
from pathlib import Path

try:
    import matplotlib
except ModuleNotFoundError:  # pragma: no cover
    print(
        "error: matplotlib is not installed. From the repository root run:\n"
        "  pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from None

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGS = BASE / "figures"

# Publication-friendly sequential palette: light → deep blue-slate (no harsh reds)
# Step text: dark on light steps; white on the two deepest steps
_BOX_FACE = ("#ebeef2", "#d3dde8", "#5a7a9a", "#2f4058")
_TEXT_PRIMARY = ("#1c2330", "#1c2330", "#f8fafc", "#f8fafc")
_TEXT_MUTED = ("#4a5568", "#4a5568", "#e8edf4", "#e8edf4")

# Pair-trade footer: warm neutral strip (reads as “economic takeaway,” not a 5th step)
_FOOTER_FACE = "#f4f0e8"
_FOOTER_EDGE = "#c4b8a4"


def load_first_recs_1y():
    df = pd.read_csv(DATA / "cramer_sequenced.csv")
    sumr = pd.read_csv(DATA / "sequence_summary.csv")
    mc = pd.read_csv(DATA / "market_caps.csv")

    df = df[(df["is_sequence_start"]) & (df["requested_horizon"] == "1y") & (df["kept"])].copy()
    df = df.merge(
        sumr[["sequence_id", "has_cramer_owns", "has_hold_recommendation"]],
        on="sequence_id",
        how="left",
    )

    def tier(row):
        if row["has_cramer_owns"]:
            return "portfolio"
        if row["has_hold_recommendation"]:
            return "hold_rec"
        return "casual"

    df["tier"] = df.apply(tier, axis=1)

    mc_lookup = mc.set_index("ticker")["market_cap"].to_dict()
    df["market_cap"] = df["ticker_symbol"].map(mc_lookup)
    return df


def cell_stats(values):
    n = len(values)
    if n < 2:
        return n, np.nan, np.nan
    mean = values.mean()
    t, p = stats.ttest_1samp(values, 0.0)
    return n, mean, p


def fmt_p(p):
    if np.isnan(p):
        return "p = n/a"
    if p < 0.0001:
        return "p < 0.0001"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _box_width(n: int, n0: int, w_max: float = 7.0, w_min: float = 3.15) -> float:
    """Funnel width ∝ sqrt(N/N0) so steps stay readable; smallest cells share a floor width."""
    return float(w_min + (w_max - w_min) * np.sqrt(n / n0))


def render(rows, output_path):
    nrows = len(rows)
    n0 = int(rows[0][1])
    widths = [_box_width(int(r[1]), n0) for r in rows]

    fig_w, fig_h = 9.2, 5.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cx = 5.0  # horizontal center of funnel
    ax.set_xlim(0, 10)
    ax.axis("off")

    box_h = 0.72
    row_pitch = 0.88
    y_top = 3.0
    y_centers = [y_top - i * row_pitch for i in range(nrows)]

    # In-axes title: sits just above the top box (tight; no suptitle whitespace)
    title_y = y_centers[0] + box_h / 2 + 0.22
    ax.text(
        cx,
        title_y,
        "From all first recommendations to the strongest in-sample cell",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="600",
        color="#1a1f2e",
    )
    y_hi = title_y + 0.36  # room for title line above va=bottom anchor (tight_layout in y)

    for i, (label, n, mean, p) in enumerate(rows):
        y = y_centers[i]
        box_w = widths[i]
        x_left = cx - box_w / 2
        face = _BOX_FACE[i % len(_BOX_FACE)]
        tc = _TEXT_PRIMARY[i % len(_TEXT_PRIMARY)]
        tmu = _TEXT_MUTED[i % len(_TEXT_MUTED)]

        rect = mpatches.FancyBboxPatch(
            (x_left, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0.6,
            edgecolor=("#b8c0cc" if i < 2 else "#1a2535"),
            facecolor=face,
        )
        ax.add_patch(rect)

        p_str = "  ·  " + fmt_p(p) if i > 0 else ""
        body = f"N = {n:,}  ·  SPY-rel α = {mean * 100:+.2f}%{p_str}"
        # Slightly smaller type when the box is narrow
        fs_t, fs_b = (9.2, 8.3) if box_w < 4.0 else (10.0, 9.0)

        ax.text(
            cx,
            y + 0.08,
            label,
            ha="center",
            va="center",
            color=tc,
            fontsize=fs_t,
            fontweight="600",
        )
        ax.text(
            cx,
            y - 0.16,
            body,
            ha="center",
            va="center",
            color=tmu,
            fontsize=fs_b,
        )

    # Arrows: connect centered stack (same x) from bottom of box i to top of box i+1
    for i in range(nrows - 1):
        y_top_edge = y_centers[i] - box_h / 2
        y_next_bottom = y_centers[i + 1] + box_h / 2
        ax.annotate(
            "",
            xy=(cx, y_next_bottom + 0.01),
            xytext=(cx, y_top_edge - 0.01),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#5c6570",
                lw=1.2,
                mutation_scale=8,
            ),
        )

    # Pair-trade strip: same width as the bottom (narrowest) filter box, wrapped to fit
    bottom_n, bottom_mean, _ = rows[-1][1], rows[-1][2], rows[-1][3]
    pair_pnl = -bottom_mean * 100
    y_last = y_centers[-1]
    foot_w = widths[-1]
    foot_x = cx - foot_w / 2
    gap = 0.1
    fs1 = 8.4 if foot_w < 4.0 else 8.8
    fs2 = 8.0 if foot_w < 4.0 else 8.3
    # Fixed two-line title + two-line detail (no aggressive textwrap)
    title_lines = [
        "Equivalent market-neutral pair trade",
        "· short stock, long SPY",
    ]
    detail_lines = [
        f"+{pair_pnl:.1f}% mean P&L per position · N = {bottom_n:,}",
        "(same cell as row above)",
    ]
    n1, n2 = 2, 2
    line_h = 0.095
    inner_gap = 0.06
    vpad = 0.16
    foot_h = max(0.72, vpad * 2 + n1 * line_h + inner_gap + n2 * line_h)
    foot_y = y_last - box_h / 2 - gap - foot_h
    foot = mpatches.FancyBboxPatch(
        (foot_x, foot_y),
        foot_w,
        foot_h,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        linewidth=0.7,
        edgecolor=_FOOTER_EDGE,
        facecolor=_FOOTER_FACE,
    )
    ax.add_patch(foot)
    y_line = foot_y + foot_h - vpad
    for line in title_lines:
        ax.text(
            cx,
            y_line,
            line,
            ha="center",
            va="top",
            fontsize=fs1,
            color="#3d3830",
            fontweight="600",
        )
        y_line -= line_h
    y_line -= inner_gap
    for line in detail_lines:
        ax.text(
            cx,
            y_line,
            line,
            ha="center",
            va="top",
            fontsize=fs2,
            color="#5c5348",
        )
        y_line -= line_h

    y_lo = foot_y - 0.12
    ax.set_ylim(y_lo, y_hi)

    # Tight frame: no footnote in figure (moved to paper body)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


def main():
    df = load_first_recs_1y()

    base_n, base_mean, base_p = cell_stats(df["spy_relative_return"])
    casual = df[df["tier"] == "casual"]
    casual_n, casual_mean, casual_p = cell_stats(casual["spy_relative_return"])
    small = casual[casual["market_cap"].notna() & (casual["market_cap"] < 2e9)]
    small_n, small_mean, small_p = cell_stats(small["spy_relative_return"])
    lowvix = small[small["vix_at_signal"] < 30]
    lowvix_n, lowvix_mean, lowvix_p = cell_stats(lowvix["spy_relative_return"])

    rows = [
        ("All first recommendations (1Y horizon)", base_n, base_mean, base_p),
        ("Filter: engagement tier = casual buy", casual_n, casual_mean, casual_p),
        ("Filter: market cap < $2B (small-cap)", small_n, small_mean, small_p),
        ("Filter: VIX < 30 at signal time", lowvix_n, lowvix_mean, lowvix_p),
    ]

    out = FIGS / "fig_funnel.png"
    out.parent.mkdir(exist_ok=True)
    render(rows, out)

    print("Funnel cells (current data):")
    for label, n, mean, p in rows:
        print(f"  {label}")
        print(f"      N={n:,}  alpha={mean*100:+.2f}%  {fmt_p(p)}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
