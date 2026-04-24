"""
Validates: paper §6.4b, Table 13 (dip-buy gradient — axis-choice sensitivity).

Sensitivity study: Does the Pillar 3 dip-buy result survive if the lookback is
measured as absolute stock return rather than SPY-relative return?

Forward measure (1Y alpha) remains SPY-relative in both cases, so the only
thing changing is the lookback axis.

Outputs, for portfolio-tier first recommendations:
  - dip-buy sweep using lookback_ret_90d (absolute)
  - dip-buy sweep using lookback_alpha_90d (SPY-relative) — reprint for comparison
  - correlation between the two lookback measures on the analysed cell
  - headline-threshold portfolio vs casual-buy contrast on absolute axis
  - short-term (30d) version of the sweep on absolute axis
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")


def seq_category(row):
    if row.get("has_cramer_owns", False):
        return "portfolio"
    elif row.get("has_hold_recommendation", False):
        return "hold_rec"
    return "no_hold"


def section(title):
    print(f"\n{'='*74}")
    print(f"  {title}")
    print(f"{'='*74}")


def load_data():
    df = pd.read_csv(os.path.join(DATA, "cramer_sequenced.csv"))
    seq = pd.read_csv(os.path.join(DATA, "sequence_summary.csv"))
    mcaps = pd.read_csv(os.path.join(DATA, "market_caps.csv"))
    lb = pd.read_csv(os.path.join(DATA, "momentum_signals.csv"))

    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    df_1y = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    df_1y = df_1y.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
    df_1y = df_1y.merge(mcaps[["ticker", "bucket"]], left_on="ticker_symbol",
                        right_on="ticker", how="left")

    for col in ["lookback_ret_30d", "lookback_alpha_30d",
                "lookback_ret_90d", "lookback_alpha_90d",
                "lookback_ret_180d", "lookback_alpha_180d"]:
        lb[col] = pd.to_numeric(lb[col], errors="coerce")

    lb_cols = ["signal_id",
               "lookback_ret_30d", "lookback_alpha_30d",
               "lookback_ret_90d", "lookback_alpha_90d",
               "lookback_ret_180d", "lookback_alpha_180d"]
    df_1y = df_1y.merge(lb[lb_cols], on="signal_id", how="left")

    return df_1y


def sweep(df, col, thresholds, label):
    print(f"\n  {label}")
    print(f"  {'threshold':<16} {'N':>6} {'mean α':>10} {'median α':>10} "
          f"{'% win':>7} {'p':>8}")
    print(f"  {'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")
    for thresh in thresholds:
        sub = df[df[col] <= thresh]
        n = len(sub)
        if n < 10:
            print(f"  ≤ {thresh*100:>+6.1f}%       {n:>6}  (insufficient)")
            continue
        a = sub["spy_relative_return"]
        t, p = stats.ttest_1samp(a, 0)
        print(f"  ≤ {thresh*100:>+6.1f}%       {n:>6} "
              f"{a.mean()*100:>+9.2f}% {a.median()*100:>+9.2f}% "
              f"{(a > 0).mean()*100:>6.1f}% {p:>8.4f}")


def main():
    df_1y = load_data()

    portfolio_first = df_1y[(df_1y["is_sequence_start"] == True) &
                            (df_1y["seq_cat"] == "portfolio")].copy()
    nohold_first = df_1y[(df_1y["is_sequence_start"] == True) &
                         (df_1y["seq_cat"] == "no_hold")].copy()

    # ── 1. Correlation between the two lookback measures ─────────────
    section("1. HOW DIFFERENT ARE THE TWO LOOKBACK MEASURES?")
    both = portfolio_first.dropna(subset=["lookback_ret_90d", "lookback_alpha_90d"])
    n = len(both)
    r = both["lookback_ret_90d"].corr(both["lookback_alpha_90d"])
    print(f"  Portfolio first-recs with both 90d measures: N={n}")
    print(f"  Pearson correlation(absolute 90d ret, SPY-rel 90d alpha): {r:.3f}")
    print()
    print(f"  Summary stats on the portfolio cell:")
    print(f"  {'':16} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10}")
    for col, lab in [("lookback_ret_90d", "absolute 90d"),
                     ("lookback_alpha_90d", "SPY-rel 90d")]:
        s = both[col]
        print(f"  {lab:<16} {s.mean()*100:>+9.2f}% {s.median()*100:>+9.2f}% "
              f"{s.std()*100:>9.2f}% {s.min()*100:>+9.2f}% {s.max()*100:>+9.2f}%")

    # ── 2. Portfolio dip-buy sweep: absolute vs SPY-relative ─────────
    section("2. PORTFOLIO DIP-BUY SWEEP — ABSOLUTE 90D RETURN")
    print(f"  Lookback axis: stock's ABSOLUTE 90-day return (not market-adjusted)")
    print(f"  Forward axis:  SPY-relative 1Y alpha")
    print(f"  N_total portfolio first-recs with 90d absolute return: "
          f"{portfolio_first['lookback_ret_90d'].notna().sum()}")

    thresholds_abs = [None, 0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    pf_abs = portfolio_first.dropna(subset=["lookback_ret_90d"])
    print(f"\n  Baseline (any, no lookback filter)")
    a = pf_abs["spy_relative_return"]
    t, p = stats.ttest_1samp(a, 0)
    print(f"  {'any':<16} {len(a):>6} {a.mean()*100:>+9.2f}% {a.median()*100:>+9.2f}% "
          f"{(a > 0).mean()*100:>6.1f}% {p:>8.4f}")
    sweep(pf_abs, "lookback_ret_90d", thresholds_abs[1:],
          "Threshold sweep on absolute 90d return:")

    section("3. PORTFOLIO DIP-BUY SWEEP — SPY-RELATIVE 90D ALPHA (REPRINT)")
    print(f"  Lookback axis: stock's 90-day SPY-RELATIVE return")
    print(f"  Forward axis:  SPY-relative 1Y alpha")
    pf_rel = portfolio_first.dropna(subset=["lookback_alpha_90d"])
    print(f"  N_total portfolio first-recs with 90d SPY-rel alpha: {len(pf_rel)}")
    thresholds_rel = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    sweep(pf_rel, "lookback_alpha_90d", thresholds_rel,
          "Threshold sweep on SPY-relative 90d alpha:")

    # ── 3. Headline threshold: portfolio vs casual-buy contrast ──────
    section("4. HEADLINE-THRESHOLD CONTRAST (absolute axis)")
    print(f"  Deep-drawdown cell: absolute 90d return ≤ -15%")
    print(f"  Compare portfolio first-recs vs casual-buy first-recs.\n")

    for label, sub in [
        ("Portfolio  (deep dip, 90d abs ≤ -15%)",
         portfolio_first[portfolio_first["lookback_ret_90d"] <= -0.15]),
        ("Casual buy (deep dip, 90d abs ≤ -15%)",
         nohold_first[nohold_first["lookback_ret_90d"] <= -0.15]),
        ("Portfolio  (deep dip, 90d abs ≤ -20%)",
         portfolio_first[portfolio_first["lookback_ret_90d"] <= -0.20]),
        ("Casual buy (deep dip, 90d abs ≤ -20%)",
         nohold_first[nohold_first["lookback_ret_90d"] <= -0.20]),
    ]:
        a = sub["spy_relative_return"]
        n = len(a)
        if n < 10:
            print(f"  {label:<42} N={n:>4} (insufficient)")
            continue
        t, p = stats.ttest_1samp(a, 0)
        print(f"  {label:<42} N={n:>4}  mean α = {a.mean()*100:>+7.2f}%  "
              f"median = {a.median()*100:>+7.2f}%  p = {p:.4f}")

    # ── 4. Short-term (30d) version of the sweep ─────────────────────
    section("5. SHORT-TERM DIP — ABSOLUTE 30D RETURN (portfolio)")
    pf_abs_30 = portfolio_first.dropna(subset=["lookback_ret_30d"])
    print(f"  Portfolio first-recs with 30d abs return: {len(pf_abs_30)}")
    thresholds_30 = [0.0, -0.05, -0.10, -0.15, -0.20]
    sweep(pf_abs_30, "lookback_ret_30d", thresholds_30,
          "Threshold sweep on absolute 30d return:")

    # ── 5. Quick sanity check: do "dips" agree? ──────────────────────
    section("6. OVERLAP CHECK")
    print("  How often does a 90d SPY-rel ≤ -15% stock ALSO have abs ≤ -15%?")
    both = portfolio_first.dropna(subset=["lookback_ret_90d", "lookback_alpha_90d"])
    alpha_dip = both["lookback_alpha_90d"] <= -0.15
    abs_dip   = both["lookback_ret_90d"]   <= -0.15
    n_alpha = alpha_dip.sum()
    n_abs   = abs_dip.sum()
    n_both  = (alpha_dip & abs_dip).sum()
    n_alpha_only = (alpha_dip & ~abs_dip).sum()
    n_abs_only   = (~alpha_dip & abs_dip).sum()
    print(f"    SPY-rel dip (α ≤ -15%):              N = {n_alpha}")
    print(f"    Absolute dip (ret ≤ -15%):           N = {n_abs}")
    print(f"    Intersection (both):                 N = {n_both}")
    print(f"    SPY-rel dip only (not absolute dip): N = {n_alpha_only}")
    print(f"    Absolute dip only (not SPY-rel dip): N = {n_abs_only}")
    if n_alpha > 0:
        print(f"    Share of SPY-rel dips that are also absolute dips: "
              f"{n_both/n_alpha*100:.1f}%")
    if n_abs > 0:
        print(f"    Share of absolute dips that are also SPY-rel dips: "
              f"{n_both/n_abs*100:.1f}%")


if __name__ == "__main__":
    main()
