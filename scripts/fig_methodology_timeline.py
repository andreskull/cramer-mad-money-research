"""
Methodology visualization: Signal Sequence Model illustrated with Visa (V).
Shows original signals, sequence grouping, hold-state propagation, and pruning.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

# Colors
C_START = "#636363"       # start_long (gray)
C_HOLD = "#2166ac"        # hold_long (blue)
C_HOLDSTATE = "#92c5de"   # in hold state background
C_KEPT = "#000000"        # kept signal border
C_PRUNED = "#d9d9d9"      # pruned signal (light gray)
C_SEQ_BG = ["#fee8c8", "#e0ecf4", "#f0f0f0", "#e5f5e0", "#fde0dd",
            "#efedf5", "#fff7bc", "#deebf7", "#fcbba1", "#d9f0a3"]


def load_v_data():
    df = pd.read_csv(DATA / "cramer_sequenced.csv")
    d1y = df[df["requested_horizon"] == "1y"].copy()
    v = d1y[d1y["ticker_symbol"] == "V"].sort_values("signal_date").copy()
    v["date"] = pd.to_datetime(v["signal_date"])
    return v


def main():
    v = load_v_data()

    fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                              gridspec_kw={"height_ratios": [2, 2, 2], "hspace": 0.35})

    date_min = v["date"].min() - timedelta(days=30)
    date_max = v["date"].max() + timedelta(days=60)

    sequences = sorted(v["sequence_id"].unique())
    seq_colors = {s: C_SEQ_BG[i % len(C_SEQ_BG)] for i, s in enumerate(sequences)}

    # ═══════════════════════════════════════════════════════════════════
    # Panel A: Raw signals with original type
    # ═══════════════════════════════════════════════════════════════════
    ax = axes[0]
    ax.set_title("A. Raw Signals — Original LLM Classification", fontweight="bold", loc="left")

    for _, row in v.iterrows():
        color = C_HOLD if row["signal_type"] == "hold_long" else C_START
        marker = "D" if row["signal_type"] == "hold_long" else "o"
        ax.plot(row["date"], row["start_price"], marker=marker, color=color,
                markersize=8, markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    # Connect with price line
    ax.plot(v["date"], v["start_price"], color="#bdbdbd", linewidth=0.8, zorder=2)

    ax.set_xlim(date_min, date_max)
    ax.set_ylabel("Entry Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_START,
               markeredgecolor="black", markersize=8, label="start_long"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_HOLD,
               markeredgecolor="black", markersize=8, label="hold_long"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    # Annotation: count
    ax.text(0.99, 0.95, f"Visa (V): {len(v)} signals, 2018–2024",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ═══════════════════════════════════════════════════════════════════
    # Panel B: Sequences with hold-state propagation
    # ═══════════════════════════════════════════════════════════════════
    ax = axes[1]
    ax.set_title("B. Signal Sequences — Hold-State Propagation", fontweight="bold", loc="left")

    # Draw sequence background spans
    for seq_id in sequences:
        seq_data = v[v["sequence_id"] == seq_id].sort_values("date")
        if len(seq_data) == 0:
            continue
        d_start = seq_data["date"].iloc[0] - timedelta(days=5)
        d_end = seq_data["date"].iloc[-1] + timedelta(days=5)
        has_hold = seq_data["in_hold_state"].any()

        # Background bar
        ax.axvspan(d_start, d_end, alpha=0.15,
                   color=C_HOLDSTATE if has_hold else "#f0f0f0",
                   zorder=1)

        # Sequence label
        mid_date = d_start + (d_end - d_start) / 2
        y_top = seq_data["start_price"].max() + 6
        label = f"Seq {sequences.index(seq_id)+1}"
        if has_hold:
            label += " [HOLD]"
        ax.text(mid_date, y_top, label, ha="center", va="bottom",
                fontsize=7, fontweight="bold" if has_hold else "normal",
                color=C_HOLD if has_hold else "#636363")

    # Draw signals
    for _, row in v.iterrows():
        in_hold = row["in_hold_state"] == True
        is_fh = row["is_first_hold"] == True
        is_start = row["is_sequence_start"] == True

        if is_fh:
            marker = "*"
            size = 14
            color = C_HOLD
        elif is_start:
            marker = "s"
            size = 9
            color = "#e6550d" if not in_hold else C_HOLD
        elif in_hold:
            marker = "D"
            size = 7
            color = C_HOLD
        else:
            marker = "o"
            size = 7
            color = C_START

        ax.plot(row["date"], row["start_price"], marker=marker, color=color,
                markersize=size, markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    ax.plot(v["date"], v["start_price"], color="#bdbdbd", linewidth=0.8, zorder=2)

    ax.set_xlim(date_min, date_max)
    ax.set_ylabel("Entry Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e6550d",
               markeredgecolor="black", markersize=9, label="Sequence start (entry signal)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_HOLD,
               markeredgecolor="black", markersize=14, label="First hold (state transition)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_HOLD,
               markeredgecolor="black", markersize=7, label="In hold state"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_START,
               markeredgecolor="black", markersize=7, label="Not in hold state"),
        mpatches.Patch(facecolor=C_HOLDSTATE, alpha=0.3, label="Hold-confirmed sequence"),
        mpatches.Patch(facecolor="#f0f0f0", alpha=0.5, label="Non-hold sequence"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ═══════════════════════════════════════════════════════════════════
    # Panel C: After pruning — kept vs discarded
    # ═══════════════════════════════════════════════════════════════════
    ax = axes[2]
    ax.set_title("C. After Pruning — Kept Signals for Statistical Testing", fontweight="bold", loc="left")

    # Draw sequence backgrounds (same as B)
    for seq_id in sequences:
        seq_data = v[v["sequence_id"] == seq_id].sort_values("date")
        if len(seq_data) == 0:
            continue
        d_start = seq_data["date"].iloc[0] - timedelta(days=5)
        d_end = seq_data["date"].iloc[-1] + timedelta(days=5)
        has_hold = seq_data["in_hold_state"].any()
        ax.axvspan(d_start, d_end, alpha=0.10,
                   color=C_HOLDSTATE if has_hold else "#f0f0f0", zorder=1)

    # Draw ALL signals (pruned as faint, kept as bold)
    for _, row in v.iterrows():
        is_kept = row["kept"] == True
        is_fh = row["is_first_hold"] == True
        is_start = row["is_sequence_start"] == True
        in_hold = row["in_hold_state"] == True

        if not is_kept:
            # Pruned: faint X
            ax.plot(row["date"], row["start_price"], marker="x", color=C_PRUNED,
                    markersize=6, markeredgewidth=1.5, zorder=3, alpha=0.6)
        else:
            # Kept: bold with type-specific marker
            if is_fh:
                marker, size, color = "*", 15, C_HOLD
            elif is_start:
                marker, size, color = "s", 10, "#e6550d" if not in_hold else C_HOLD
            elif in_hold:
                marker, size, color = "D", 8, C_HOLD
            else:
                marker, size, color = "o", 8, C_START

            ax.plot(row["date"], row["start_price"], marker=marker, color=color,
                    markersize=size, markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    # Price line through kept signals only
    kept = v[v["kept"] == True]
    ax.plot(kept["date"], kept["start_price"], color="#bdbdbd", linewidth=0.8,
            zorder=2, linestyle="--")

    ax.set_xlim(date_min, date_max)
    ax.set_ylabel("Entry Price ($)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    n_kept = v["kept"].sum()
    n_total = len(v)
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e6550d",
               markeredgecolor="black", markersize=10, label="Kept: sequence start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_HOLD,
               markeredgecolor="black", markersize=14, label="Kept: first hold"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_HOLD,
               markeredgecolor="black", markersize=8, label="Kept: hold-state signal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_START,
               markeredgecolor="black", markersize=8, label="Kept: non-hold signal"),
        Line2D([0], [0], marker="x", color=C_PRUNED, markersize=7,
               markeredgewidth=2, linestyle="None", label=f"Pruned ({n_total - n_kept} of {n_total})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9, ncol=2)

    # Summary annotation
    ax.text(0.99, 0.95,
            f"Kept: {n_kept}/{n_total} signals ({n_kept/n_total*100:.0f}%)\n"
            f"Pruned: time <60d & price <10% from last kept",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Title ──────────────────────────────────────────────────────────
    fig.suptitle("Signal Sequence Model: Methodology Illustrated with Visa (V)",
                 fontsize=14, fontweight="bold", y=0.98)

    out = FIGS / "fig_methodology_timeline.png"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")
    print(f"  V: {n_total} signals → {n_kept} kept, {v['sequence_id'].nunique()} sequences")


if __name__ == "__main__":
    main()
