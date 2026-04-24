"""
Replicates the headline tables in the paper that rely on
cluster-robust standard errors and / or the QQQ benchmark, neither of
which the older `study_*.py` scripts implement.

Outputs printed to stdout:
    - Table 4   Engagement-tier first-recommendation alpha (1Y), §6.1
                (ticker-clustered + two-way ticker x signal-month for
                the casual-buy cell)
    - Table 6   Casual-buy first-recommendation alpha by market-cap
                bucket (1Y), §6.2
    - Table 7   Small-cap casual-buy pair trade and simple short by
                VIX regime (1Y), §6.2
    - Table 8   Small-cap pair trade vs SPY and QQQ benchmarks
                (combined VIX<30, 1Y), §6.2
    - Table 8a  5 x 4 specification grid of small-cap casual-buy pair
                trade across (cap cutoff, VIX cutoff), §6.2

Files written to data/:
    - bh_grid.csv          5 horizons x 3 VIX regimes x 3 show segments
                           cluster-robust pvals + BH-corrected decisions
                           consumed by visualize_sequence.fig6_bh_heatmap
                           (Figure 11)
    - headline_tables.csv  long-form copy of every cell printed above so
                           downstream tools can diff against the paper
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve()
BASE = THIS.parent.parent
sys.path.insert(0, str(THIS.parent))

from stats_helpers import (
    bh_correct,
    cluster_robust_mean_p,
    two_way_cluster_mean_p,
)

DATA = BASE / "data"


def load_universe():
    df = pd.read_csv(DATA / "cramer_sequenced.csv")
    seq = pd.read_csv(DATA / "sequence_summary.csv")
    mc = pd.read_csv(DATA / "market_caps.csv")

    def _seq_cat(r):
        if r.get("has_cramer_owns", False):
            return "portfolio"
        if r.get("has_hold_recommendation", False):
            return "hold_rec"
        return "no_hold"

    seq = seq.copy()
    seq["seq_cat"] = seq.apply(_seq_cat, axis=1)
    df = df.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
    df = df.merge(
        mc[["ticker", "market_cap", "bucket"]],
        left_on="ticker_symbol",
        right_on="ticker",
        how="left",
    )
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["signal_month"] = df["signal_date"].dt.to_period("M").astype(str)
    return df


def fmt_p(p):
    if pd.isna(p):
        return "  n/a"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def fmt_pct(x, sign=False):
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:+.2f}%" if sign else f"{x * 100:.2f}%"


def table4(df_first):
    print("\n=== Table 4: Engagement-tier alpha, first-recommendation, 1Y, pruned ===")
    rows = []
    for cat in ("portfolio", "hold_rec", "no_hold"):
        sub = df_first[df_first["seq_cat"] == cat]
        n, m, p_iid, p_cl, n_g, _ = cluster_robust_mean_p(
            sub["spy_relative_return"], sub["ticker_symbol"]
        )
        rows.append(
            {
                "table": 4,
                "cell": f"{cat}",
                "N": n,
                "G": n_g,
                "mean_alpha": m,
                "p_iid": p_iid,
                "p_cluster_ticker": p_cl,
            }
        )
        print(
            f"  {cat:>11s}  N={n:5d}  G={n_g:5d}  alpha={fmt_pct(m, sign=True):>8s}  "
            f"p_iid={fmt_p(p_iid):>9s}  p_cluster={fmt_p(p_cl):>9s}"
        )

    sub_nh = df_first[df_first["seq_cat"] == "no_hold"]
    n, m, p2, n_a, n_b = two_way_cluster_mean_p(
        sub_nh["spy_relative_return"],
        sub_nh["ticker_symbol"],
        sub_nh["signal_month"],
    )
    print(
        f"  no_hold (2-way ticker x signal-month)  N={n}  G_ticker={n_a}  "
        f"G_month={n_b}  p_two_way={fmt_p(p2)}"
    )
    rows.append(
        {
            "table": 4,
            "cell": "no_hold__two_way_ticker_x_month",
            "N": n,
            "G": n_a,
            "mean_alpha": m,
            "p_iid": float("nan"),
            "p_cluster_ticker": float("nan"),
            "p_cluster_two_way": p2,
            "G_b": n_b,
        }
    )
    return rows


def table6(df_first):
    print("\n=== Table 6: Casual-buy first-rec alpha by market-cap bucket, 1Y ===")
    casual = df_first[df_first["seq_cat"] == "no_hold"]
    rows = []
    for bucket in ("Mega-cap", "Large-cap", "Mid-cap", "Small-cap"):
        sub = casual[casual["bucket"] == bucket]
        n, m, p_iid, p_cl, n_g, _ = cluster_robust_mean_p(
            sub["spy_relative_return"], sub["ticker_symbol"]
        )
        abs_ret = sub["absolute_return"].mean() if n else float("nan")
        pct_neg_spy = (
            (sub["spy_relative_return"] < 0).mean() if n else float("nan")
        )
        rows.append(
            {
                "table": 6,
                "cell": bucket,
                "N": n,
                "G": n_g,
                "mean_alpha": m,
                "abs_return": abs_ret,
                "pct_neg_spy": pct_neg_spy,
                "p_cluster_ticker": p_cl,
            }
        )
        print(
            f"  {bucket:>10s}  N={n:5d}  G={n_g:5d}  alpha={fmt_pct(m, sign=True):>8s}  "
            f"abs={fmt_pct(abs_ret, sign=True):>8s}  pct_neg={pct_neg_spy * 100:5.1f}%  "
            f"p_cluster={fmt_p(p_cl):>9s}"
        )
    return rows


def table7(df_first):
    print("\n=== Table 7: Small-cap casual-buy pair trade & simple short by VIX, 1Y ===")
    casual = df_first[df_first["seq_cat"] == "no_hold"]
    small = casual[(casual["bucket"] == "Small-cap") & casual["vix_at_signal"].notna()]
    bins = [
        ("Low (<20)", small[small["vix_at_signal"] < 20]),
        (
            "Moderate (20-30)",
            small[(small["vix_at_signal"] >= 20) & (small["vix_at_signal"] < 30)],
        ),
        ("High (>=30)", small[small["vix_at_signal"] >= 30]),
        ("Combined <30", small[small["vix_at_signal"] < 30]),
    ]
    rows = []
    for label, sub in bins:
        if not len(sub):
            continue
        pair = -sub["spy_relative_return"]
        short = -sub["absolute_return"] - sub["spy_return"]
        n, m_p, _, p_p, n_g, _ = cluster_robust_mean_p(pair, sub["ticker_symbol"])
        _, m_s, _, p_s, _, _ = cluster_robust_mean_p(short, sub["ticker_symbol"])
        rows.append(
            {
                "table": 7,
                "cell": label,
                "N": n,
                "G": n_g,
                "pair_pnl": m_p,
                "p_pair_cluster": p_p,
                "short_pnl": m_s,
                "p_short_cluster": p_s,
            }
        )
        print(
            f"  {label:>20s}  N={n:5d}  G={n_g:5d}  "
            f"pair={fmt_pct(m_p, sign=True):>8s} (p={fmt_p(p_p)})  "
            f"short={fmt_pct(m_s, sign=True):>8s} (p={fmt_p(p_s)})"
        )
    return rows


def table8(df_first):
    print("\n=== Table 8: Small-cap pair trade SPY vs QQQ benchmarks (VIX<30, 1Y) ===")
    print("    (also reports t-stats and two-way ticker x signal-month p-values")
    print("     used in the §6.2 cluster-robust verification block)")
    casual = df_first[df_first["seq_cat"] == "no_hold"]
    small = casual[
        (casual["bucket"] == "Small-cap")
        & casual["vix_at_signal"].notna()
        & (casual["vix_at_signal"] < 30)
        & casual["qqq_relative_return"].notna()
    ]
    rows = []
    for bench, col in (("SPY", "spy_relative_return"), ("QQQ", "qqq_relative_return")):
        pair = -small[col]
        n, m, _, p_cl, n_g, t_stat = cluster_robust_mean_p(pair, small["ticker_symbol"])
        n2, m2, p_two, n_a, n_b = two_way_cluster_mean_p(
            pair, small["ticker_symbol"], small["signal_month"]
        )
        rows.append(
            {
                "table": 8,
                "cell": bench,
                "N": n,
                "G": n_g,
                "pair_pnl": m,
                "t_cluster_ticker": t_stat,
                "p_cluster_ticker": p_cl,
                "p_cluster_two_way": p_two,
            }
        )
        print(
            f"  Long {bench}, short stock  N={n:5d}  G={n_g:5d}  "
            f"pair={fmt_pct(m, sign=True):>8s}  t_cluster={t_stat:5.2f}  "
            f"p_cluster={fmt_p(p_cl):>9s}  p_two_way={p_two:.2e}"
        )
    return rows


def table8a(df_first):
    print("\n=== Table 8a: 5x4 spec grid for casual-buy pair trade vs SPY (1Y) ===")
    casual = df_first[df_first["seq_cat"] == "no_hold"]
    cap_cuts = [1e9, 2e9, 3e9, 5e9, 10e9]
    cap_labels = ["<$1B", "<$2B", "<$3B", "<$5B", "<$10B"]
    vix_cuts = [25, 30, 35, None]
    vix_labels = ["VIX<25", "VIX<30", "VIX<35", "no VIX filter"]
    rows = []
    header = f"{'cap':>8s}  " + "  ".join(f"{vl:>20s}" for vl in vix_labels)
    print(header)
    for cap_cut, cap_lab in zip(cap_cuts, cap_labels):
        cells = []
        for vix_cut, vix_lab in zip(vix_cuts, vix_labels):
            sub = casual[casual["market_cap"] < cap_cut]
            if vix_cut is not None:
                sub = sub[
                    sub["vix_at_signal"].notna() & (sub["vix_at_signal"] < vix_cut)
                ]
            pair = -sub["spy_relative_return"]
            n, m, _, p_cl, n_g, _ = cluster_robust_mean_p(pair, sub["ticker_symbol"])
            rows.append(
                {
                    "table": "8a",
                    "cell": f"{cap_lab} x {vix_lab}",
                    "N": n,
                    "G": n_g,
                    "pair_pnl": m,
                    "p_cluster_ticker": p_cl,
                }
            )
            cells.append(f"{fmt_pct(m, sign=True):>8s} N={n:4d} p={fmt_p(p_cl):>7s}")
        print(f"{cap_lab:>8s}  " + "  ".join(f"{c:>20s}" for c in cells))
    return rows


def build_bh_grid(df):
    """5 horizons x 3 VIX regimes x 3 show segments cluster-robust grid
    consumed by visualize_sequence.fig6_bh_heatmap (Figure 11)."""

    def _vix_bin(v):
        if pd.isna(v):
            return None
        if v < 20:
            return "low"
        if v < 30:
            return "moderate"
        return "high"

    seg_keep = ("Monologue/Top of Show", "Lightning Round", "Mid-Show/Interview")
    horizons = ("1w", "1m", "3m", "6m", "1y")

    base = df[df["kept"] & df["is_sequence_start"]].copy()
    base["vix_bin"] = base["vix_at_signal"].apply(_vix_bin)
    base = base[base["vix_bin"].notna() & base["show_segment_heuristic"].isin(seg_keep)]

    rows = []
    for h in horizons:
        sub_h = base[base["requested_horizon"] == h]
        for v in ("low", "moderate", "high"):
            for s in seg_keep:
                cell = sub_h[(sub_h["vix_bin"] == v) & (sub_h["show_segment_heuristic"] == s)]
                n, m, _, p_cl, n_g, _ = cluster_robust_mean_p(
                    cell["spy_relative_return"], cell["ticker_symbol"]
                )
                rows.append(
                    {
                        "horizon": h,
                        "vix": v,
                        "segment": s,
                        "n": n,
                        "n_clusters": n_g,
                        "mean_alpha": m,
                        "p_cluster_ticker": p_cl,
                    }
                )
    out = pd.DataFrame(rows)
    reject_tp, p_adj = bh_correct(out["p_cluster_ticker"].to_numpy(), alpha=0.05)
    out["reject_tp"] = reject_tp
    out["p_bh_adjusted"] = p_adj
    out.to_csv(DATA / "bh_grid.csv", index=False)
    n_surv = int(out["reject_tp"].sum())
    print(
        f"\n=== Figure 11 BH grid: {n_surv}/{len(out)} cells survive cluster-robust "
        f"BH at FDR 0.05 (saved data/bh_grid.csv) ==="
    )
    return out


def main():
    df = load_universe()
    df_first = df[
        (df["requested_horizon"] == "1y")
        & (df["kept"])
        & (df["is_sequence_start"])
    ].copy()
    print(
        f"Loaded {len(df):,} signal observations; first-recommendation 1Y "
        f"pruned universe N = {len(df_first):,}"
    )

    rows = []
    rows += table4(df_first)
    rows += table6(df_first)
    rows += table7(df_first)
    rows += table8(df_first)
    rows += table8a(df_first)
    pd.DataFrame(rows).to_csv(DATA / "headline_tables.csv", index=False)
    print(f"\nWrote data/headline_tables.csv ({len(rows)} rows)")

    build_bh_grid(df)


if __name__ == "__main__":
    main()
