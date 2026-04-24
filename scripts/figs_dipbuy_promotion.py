"""
Figures for §6.4a (promotion effect) and §6.4b (dip-buy gradient).

Outputs:
  figures/fig_promotion_effect.png   visualizes Table 12
  figures/fig_dip_buy_gradient.png   visualizes Table 13

Both figures show 95% intervals derived under the same cluster-aware
inference the paper reports in the corresponding tables: the promotion
panel uses the ticker-block bootstrap CI from stats_helpers; the dip-buy
panel uses ticker-clustered standard errors on the per-position alpha
within each (tier, drawdown threshold) cell.

Run after study_b_momentum.py (which produces the price matrix and
sequenced data this script reads from).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from stats_helpers import (  # noqa: E402
    block_bootstrap_diff_p,
    cluster_robust_mean_p,
    panel_fe_diff_p,
)

BASE = os.path.dirname(SCRIPT_DIR)
DATA = os.path.join(BASE, "data")
FIGS = os.path.join(BASE, "figures")
os.makedirs(FIGS, exist_ok=True)

SPY_FIGI = "BBG000BDTF76"
LOOKBACK_DAYS = 90

C_PORT = "#2166ac"
C_NOHOLD = "#b2182b"
C_MENTION = "#1a9850"
C_SILENCE = "#bdbdbd"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


# ── Promotion effect (Table 12) ───────────────────────────────────────


def build_promotion_panel():
    """Return the active-window portfolio stock-day timing panel.

    For each stock Cramer has disclosed as owned, include trading days from
    the first through the last `cramer_owns` mention date. This keeps the
    "silent" comparison inside the observed ownership-disclosure window.
    """
    df = pd.read_csv(os.path.join(DATA, "cramer_sequenced.csv"))
    pm = pd.read_parquet(os.path.join(DATA, "daily_price_matrix.parquet"))
    pm["trade_date"] = pd.to_datetime(pm["trade_date"])

    figi_map = (
        df[["figi_id", "ticker_symbol"]].dropna().drop_duplicates()
    )
    figi_ticker = figi_map.set_index("figi_id")["ticker_symbol"].to_dict()

    port_signals = df[df["hold_subtype"] == "cramer_owns"].copy()
    port_signals["signal_date_dt"] = pd.to_datetime(port_signals["signal_date"])
    port_tickers = set(port_signals["ticker_symbol"].unique())

    ticker_signal_dates = {}
    for _, row in port_signals.iterrows():
        ticker_signal_dates.setdefault(row["figi_id"], set()).add(
            row["signal_date_dt"].date()
        )
    port_figis = [
        f for f in ticker_signal_dates
        if figi_ticker.get(f) in port_tickers
    ]

    figi_long, ret_long, treat_long = [], [], []
    for figi in port_figis:
        prices = (
            pm[pm["figi_id"] == figi]
            .set_index("trade_date")["adj_close"]
            .sort_index()
        )
        if len(prices) < 35:
            continue
        sig_dates = ticker_signal_dates.get(figi, set())
        if not sig_dates:
            continue
        first_sig = pd.Timestamp(min(sig_dates))
        last_sig = pd.Timestamp(max(sig_dates))
        for i in range(30, len(prices)):
            trade_date = prices.index[i]
            if not (first_sig <= trade_date <= last_sig):
                continue
            p_now = float(prices.iloc[i])
            p_then = float(prices.iloc[i - 30])
            if not (np.isfinite(p_now) and np.isfinite(p_then) and p_then > 0):
                continue
            figi_long.append(figi)
            ret_long.append(p_now / p_then - 1.0)
            treat_long.append(1 if trade_date.date() in sig_dates else 0)

    return (
        np.asarray(figi_long),
        np.asarray(ret_long, dtype=float),
        np.asarray(treat_long, dtype=int),
    )


def fig_promotion_effect():
    section("FIG: PROMOTION EFFECT (Table 12)")
    figi_arr, ret_arr, treat_arr = build_promotion_panel()
    n_panel = len(ret_arr)
    print(f"  Panel size: {n_panel} stock-day observations across "
          f"{len(set(figi_arr))} portfolio tickers")

    rec_mean = float(ret_arr[treat_arr == 1].mean())
    sil_mean = float(ret_arr[treat_arr == 0].mean())
    n_rec = int((treat_arr == 1).sum())
    n_sil = int((treat_arr == 0).sum())

    n_pan, g_pan, beta_pan, se_pan, p_pan = panel_fe_diff_p(
        ret_arr, treat_arr, figi_arr
    )
    n_bb, g_bb, diff_bb, p_bb, ci_low, ci_high = block_bootstrap_diff_p(
        ret_arr, treat_arr, figi_arr, b_iters=1000
    )
    half_width = (ci_high - ci_low) / 2.0
    print(f"  Mention mean = {rec_mean*100:+.2f}% (N={n_rec})")
    print(f"  Silence mean = {sil_mean*100:+.2f}% (N={n_sil})")
    print(f"  Panel FE + ticker-clustered SE: beta={beta_pan*100:+.2f}pp  "
          f"se={se_pan*100:.2f}pp  p={p_pan:.4f}  (N={n_pan}, G={g_pan})")
    print(f"  Diff = {diff_bb*100:+.2f}pp  bootstrap 95% CI "
          f"[{ci_low*100:+.2f}, {ci_high*100:+.2f}] pp  p={p_bb:.4f}")

    # The bootstrap CI is on the *difference*. We center half the width on
    # each bar so the visual distance between the two whiskers matches the
    # bootstrap-derived uncertainty in the gap.
    bar_se = half_width / 2.0

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    means = [rec_mean * 100, sil_mean * 100]
    colors = [C_MENTION, C_SILENCE]
    labels = [
        f"Cramer mentions\nN={n_rec:,}",
        f"Cramer silent\nN={n_sil:,}",
    ]

    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=1.5, width=0.55)
    ax.errorbar(x, means, yerr=[bar_se * 100, bar_se * 100], fmt="none",
                ecolor="black", capsize=5, linewidth=1.2, capthick=1.2)

    for xi, m in zip(x, means):
        ax.text(xi, m + 0.25, f"{m:+.2f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Trailing 30-day return on portfolio stock (%)")
    ax.set_title("Trailing 30-day return: mention days vs. silent days", pad=8)
    ax.set_ylim(0, max(means) * 1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(FIGS, "fig_promotion_effect.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


# ── Dip-buy gradient (Table 13) ───────────────────────────────────────


def seq_category(row):
    if row.get("has_cramer_owns", False):
        return "portfolio"
    if row.get("has_hold_recommendation", False):
        return "hold_rec"
    return "no_hold"


def compute_lookback_for_signals(signals_df, prices_by_figi, spy_series):
    """90-day absolute return (signal_date relative to signal_date − 90d)
    for each row in signals_df, computed from the daily price matrix so
    that repeat-mention signals (which momentum_signals.csv does not
    cover) are included."""
    out_abs = []
    for _, row in signals_df.iterrows():
        fid = row["figi_id"]
        sd = row["signal_date"]
        if pd.isna(fid) or fid not in prices_by_figi:
            out_abs.append(np.nan)
            continue
        s = prices_by_figi[fid]
        s_le = s.loc[:sd]
        if s_le.empty:
            out_abs.append(np.nan)
            continue
        end_price = s_le.iloc[-1]
        cutoff = sd - pd.Timedelta(days=LOOKBACK_DAYS)
        s_lb = s.loc[:cutoff]
        if s_lb.empty:
            out_abs.append(np.nan)
            continue
        start_price = s_lb.iloc[-1]
        if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
            out_abs.append(np.nan)
            continue
        out_abs.append(end_price / start_price - 1.0)
    return pd.Series(out_abs, index=signals_df.index, name="lookback_ret_90d")


def cell_stats(alphas, tickers):
    """Per-cell mean alpha and ±1.96·SE on the ticker-clustered SE."""
    alphas = pd.Series(alphas).dropna()
    if len(alphas) < 5:
        return None
    n, mean, p_iid, p_cl, n_g, t_cl = cluster_robust_mean_p(
        alphas.values, tickers.loc[alphas.index].values
    )
    if not np.isfinite(t_cl) or t_cl == 0:
        se_cl = float("nan")
    else:
        se_cl = abs(mean / t_cl)
    return {
        "n": int(n),
        "mean": float(mean),
        "se_cluster": float(se_cl),
        "p_cluster": float(p_cl),
        "n_clusters": int(n_g),
    }


def fig_dip_buy_gradient():
    section("FIG: DIP-BUY GRADIENT (Table 13)")

    df = pd.read_csv(os.path.join(DATA, "cramer_sequenced.csv"))
    seq = pd.read_csv(os.path.join(DATA, "sequence_summary.csv"))
    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    df_1y = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    df_1y = df_1y.merge(
        seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left"
    )
    df_1y["signal_date"] = pd.to_datetime(df_1y["signal_date"])

    portfolio_all = df_1y[df_1y["seq_cat"] == "portfolio"].copy()
    nohold_all = df_1y[df_1y["seq_cat"] == "no_hold"].copy()

    prices = pd.read_parquet(os.path.join(DATA, "daily_price_matrix.parquet"))
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices_by_figi = {
        fid: g.set_index("trade_date")["adj_close"].astype(float).sort_index()
        for fid, g in prices.groupby("figi_id")
    }
    spy_series = prices_by_figi.get(SPY_FIGI)

    portfolio_all["lookback_ret_90d"] = compute_lookback_for_signals(
        portfolio_all, prices_by_figi, spy_series
    )
    nohold_all["lookback_ret_90d"] = compute_lookback_for_signals(
        nohold_all, prices_by_figi, spy_series
    )

    thresholds = [
        ("Aggregate", None),
        ("≤ −10%", -0.10),
        ("≤ −15%", -0.15),
        ("≤ −20%", -0.20),
        ("≤ −25%", -0.25),
    ]

    rows = []
    for label, thr in thresholds:
        for tier_label, sub in [("Portfolio", portfolio_all),
                                ("Casual buy", nohold_all)]:
            if thr is None:
                cell = sub
            else:
                cell = sub[sub["lookback_ret_90d"] <= thr]
            cell = cell.dropna(subset=["spy_relative_return", "ticker_symbol"])
            if len(cell) < 5:
                rows.append({"label": label, "tier": tier_label,
                             "mean": np.nan, "se": np.nan, "n": len(cell),
                             "p": np.nan})
                continue
            stats_dict = cell_stats(
                cell["spy_relative_return"], cell["ticker_symbol"]
            )
            rows.append({
                "label": label,
                "tier": tier_label,
                "mean": stats_dict["mean"],
                "se": stats_dict["se_cluster"],
                "n": stats_dict["n"],
                "p": stats_dict["p_cluster"],
            })

    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(thresholds))
    bar_width = 0.35

    for i, (tier_label, color) in enumerate([("Portfolio", C_PORT),
                                             ("Casual buy", C_NOHOLD)]):
        sub = grid[grid["tier"] == tier_label].set_index("label").reindex(
            [t[0] for t in thresholds]
        )
        means = sub["mean"].fillna(0).values * 100
        ses = sub["se"].fillna(0).values * 100
        ns = sub["n"].fillna(0).astype(int).values
        present = sub["mean"].notna().values

        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, color=color,
                      label=tier_label, alpha=0.85, edgecolor="white",
                      linewidth=1.2)
        whisker_yerr = [1.96 * s if ok else 0 for s, ok in zip(ses, present)]
        ax.errorbar(x + offset, means, yerr=whisker_yerr, fmt="none",
                    ecolor="black", capsize=4, linewidth=0.9, capthick=0.9)

        for bar, m, n, ok in zip(bars, means, ns, present):
            if not ok or n < 5:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.5,
                        f"N={n} (thin)", ha="center", va="bottom",
                        fontsize=8, color="grey", style="italic")
                bar.set_facecolor("none")
                bar.set_edgecolor(color)
                bar.set_hatch("///")
                bar.set_height(0)
                continue
            yoff = 0.6 if m >= 0 else -0.6
            va = "bottom" if m >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, m + yoff,
                    f"{m:+.1f}%\n(N={n})", ha="center", va=va, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t[0] for t in thresholds])
    ax.set_xlabel("Prior 90-day absolute return at signal time")
    ax.set_ylabel("Forward 1Y SPY-relative alpha (%)")
    ax.set_title("Forward 1Y alpha by prior 90d return (Portfolio vs. Casual buy)",
                 pad=8)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(FIGS, "fig_dip_buy_gradient.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Saved: {out}")


def main():
    fig_promotion_effect()
    fig_dip_buy_gradient()


if __name__ == "__main__":
    main()
