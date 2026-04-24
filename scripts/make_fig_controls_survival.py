"""
Generate figures/fig_controls_survival.png — the §6.3 robustness panel.

Shows the portfolio-pick alpha minus casual-buy alpha (in percentage points,
1Y horizon) inside every stratum reported in §6.3:

    SIZE BUCKET       — Mega-cap, Large-cap, Mid-cap, Small-cap
    PRIOR 90-DAY      — ≤−15%, −15% to 0%, 0% to +15%, ≥+15% (SPY-relative)
    GICS SECTOR       — Technology, Consumer Cyclical, Healthcare, Industrials

Strata where either the portfolio cell or the casual-buy cell has fewer than
20 observations are rendered as "(small N)" / "(limited N)" placeholders so
the chart honestly flags where the control gap can't be reliably measured.

Reads:
    data/cramer_sequenced.csv
    data/sequence_summary.csv
    data/market_caps.csv
    data/momentum_signals.csv

Writes:
    figures/fig_controls_survival.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGS = BASE / "figures"

MIN_N = 20

SIZE_ORDER = ["Mega-cap", "Large-cap", "Mid-cap", "Small-cap"]
SIZE_LABELS = {
    "Mega-cap": "Mega-cap (>$200B)",
    "Large-cap": "Large-cap ($10\u2013200B)",
    "Mid-cap": "Mid-cap ($2\u201310B)",
    "Small-cap": "Small-cap (<$2B)",
}

MOMENTUM_BUCKETS = [
    ("\u2264 \u221215%",   lambda x: x <= -0.15),
    ("\u221215% to 0%",    lambda x: (x > -0.15) & (x <= 0)),
    ("0% to +15%",         lambda x: (x > 0) & (x < 0.15)),
    ("\u2265 +15%",        lambda x: x >= 0.15),
]

SECTORS = ["Technology", "Consumer Cyclical", "Healthcare", "Industrials"]


def load_first_recs_1y():
    df = pd.read_csv(DATA / "cramer_sequenced.csv")
    sumr = pd.read_csv(DATA / "sequence_summary.csv")
    mc = pd.read_csv(DATA / "market_caps.csv")
    mom = pd.read_csv(DATA / "momentum_signals.csv")

    df = df[(df["is_sequence_start"]) & (df["requested_horizon"] == "1y") & (df["kept"])].copy()
    df = df.merge(
        sumr[["sequence_id", "has_cramer_owns", "has_hold_recommendation"]],
        on="sequence_id", how="left",
    )

    def tier(row):
        if row["has_cramer_owns"]:
            return "portfolio"
        if row["has_hold_recommendation"]:
            return "hold_rec"
        return "casual"

    df["tier"] = df.apply(tier, axis=1)

    df = df.merge(mc[["ticker", "bucket"]], left_on="ticker_symbol", right_on="ticker", how="left")
    df = df.merge(mom[["sequence_id", "lookback_alpha_90d"]], on="sequence_id", how="left")

    return df


def gap(df, mask):
    sub = df[mask]
    p = sub[sub["tier"] == "portfolio"]["spy_relative_return"]
    c = sub[sub["tier"] == "casual"]["spy_relative_return"]
    if len(p) < MIN_N or len(c) < MIN_N:
        return None, len(p), len(c)
    return (p.mean() - c.mean()) * 100.0, len(p), len(c)


def collect_rows(df):
    rows = []

    rows.append(("section", "SIZE BUCKET"))
    for b in SIZE_ORDER:
        g, np_, nc_ = gap(df, df["bucket"] == b)
        rows.append(("bar", SIZE_LABELS[b], g, np_, nc_))

    rows.append(("section", "PRIOR 90-DAY vs SPY"))
    for label, fn in MOMENTUM_BUCKETS:
        mask = df["lookback_alpha_90d"].notna() & fn(df["lookback_alpha_90d"])
        g, np_, nc_ = gap(df, mask)
        rows.append(("bar", label, g, np_, nc_))

    rows.append(("section", "GICS SECTOR"))
    for s in SECTORS:
        g, np_, nc_ = gap(df, df["gics_sector"] == s)
        rows.append(("bar", s, g, np_, nc_))

    return rows


def render(rows, output_path):
    bar_rows = [r for r in rows if r[0] == "bar"]
    n_bars = len(bar_rows)
    section_rows = [(i, r[1]) for i, r in enumerate(rows) if r[0] == "section"]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    y_positions = []
    cur_y = 0
    label_for_y = {}
    bar_value_for_y = {}
    n_for_y = {}

    is_first_section = True
    for r in rows:
        if r[0] == "section":
            if not is_first_section:
                cur_y -= 0.5
            is_first_section = False
            label_for_y[cur_y] = ("SECTION", r[1])
            cur_y -= 1.1
        else:
            _, label, value, np_, nc_ = r
            y_positions.append(cur_y)
            label_for_y[cur_y] = ("BAR", label)
            bar_value_for_y[cur_y] = value
            n_for_y[cur_y] = (np_, nc_)
            cur_y -= 1.0

    section_band_color = "#f6f6f6"
    section_y_ranges = []
    last_section_top = None
    last_section_label = None
    for y, kind_label in sorted(label_for_y.items(), reverse=True):
        kind, label = kind_label
        if kind == "SECTION":
            if last_section_top is not None:
                section_y_ranges.append((last_section_label, last_section_top, y))
            last_section_top = y
            last_section_label = label
    if last_section_top is not None:
        section_y_ranges.append((last_section_label, last_section_top, cur_y - 0.5))

    for label, top, bottom in section_y_ranges:
        ax.axhspan(bottom, top, facecolor=section_band_color, edgecolor="none", zorder=0)

    bar_height = 0.55
    for y in y_positions:
        v = bar_value_for_y[y]
        if v is None:
            continue
        color = "#2f78b9" if v >= 0 else "#c9302c"
        ax.barh(y, v, height=bar_height, color=color, zorder=2)
        offset = 0.5 if v >= 0 else -0.5
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, y, f"{v:+.1f} pp", va="center", ha=ha,
                fontsize=10, fontweight="bold", color="#222", zorder=3)

    for y in y_positions:
        v = bar_value_for_y[y]
        if v is None:
            np_, nc_ = n_for_y[y]
            tag = "(small N)" if min(np_, nc_) < 5 else "(limited N)"
            ax.text(0.5, y, tag, va="center", ha="left",
                    fontsize=10, color="#999", style="italic", zorder=3)

    label_x = -30
    for y, kind_label in label_for_y.items():
        kind, label = kind_label
        if kind == "SECTION":
            ax.text(label_x, y, label, va="center", ha="left",
                    fontsize=11, fontweight="bold", color="#222")
        else:
            ax.text(label_x + 1.5, y, label, va="center", ha="left",
                    fontsize=10, color="#222")

    ax.axvline(0, color="#222", lw=1, zorder=1)
    ax.set_xlim(label_x, 30)
    ax.set_ylim(cur_y - 0.3, 0.6)

    ax.set_xticks([-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25])
    ax.set_xticklabels(["-25", "-20", "-15", "-10", "-5", "0", "+5", "+10", "+15", "+20", "+25"])
    ax.set_yticks([])
    ax.set_xlabel("Portfolio-pick alpha \u2212 casual-buy alpha  (percentage points, 1Y)",
                  fontsize=10.5)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.tick_params(axis="x", colors="#444")

    # No suptitle or footer: interpretation lives in the paper Figure 6 caption, not the PNG
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)


def main():
    df = load_first_recs_1y()
    rows = collect_rows(df)

    out = FIGS / "fig_controls_survival.png"
    render(rows, out)

    print("Controls-survival cells (current data, 1Y horizon, first recs only):")
    for r in rows:
        if r[0] == "section":
            print(f"\n[{r[1]}]")
        else:
            _, label, value, np_, nc_ = r
            if value is None:
                print(f"  {label:<28}  N_port={np_}  N_casual={nc_}  (insufficient)")
            else:
                print(f"  {label:<28}  gap={value:+.2f} pp   (N_port={np_}, N_casual={nc_})")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
