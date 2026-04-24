"""
Actionable quantification studies:

Q1: Inverse strategy on small-cap casual picks — standalone profitability
Q2: Portfolio dip-buy signal — what drawdown threshold maximizes forward alpha?
Q3: VIX × conviction — how should volatility regime weight the signals?
Q4: Combined feature — portfolio dip-buy × VIX regime
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


def vix_regime(v):
    if pd.isna(v): return None
    if v < 20: return "Low"
    if v < 30: return "Moderate"
    return "High"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


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
    df_1y["vix_regime"] = df_1y["vix_at_signal"].apply(vix_regime)

    # Merge lookback data
    lb["lookback_alpha_90d"] = pd.to_numeric(lb["lookback_alpha_90d"], errors="coerce")
    lb["lookback_alpha_30d"] = pd.to_numeric(lb["lookback_alpha_30d"], errors="coerce")
    lb["lookback_alpha_180d"] = pd.to_numeric(lb["lookback_alpha_180d"], errors="coerce")
    lb_cols = ["signal_id", "lookback_alpha_30d", "lookback_alpha_90d", "lookback_alpha_180d"]
    df_1y = df_1y.merge(lb[lb_cols], on="signal_id", how="left")

    return df_1y


# ── Q1: Inverse small-cap casual picks ────────────────────────────────

def inverse_small_cap(df_1y):
    section("Q1: INVERSE STRATEGY ON SMALL-CAP CASUAL PICKS")
    print("  If small-cap casual picks average -24.6% alpha, can we short them?")
    print("  Forward 1Y alpha of the SHORT position = -1 × long alpha")

    first_recs = df_1y[df_1y["is_sequence_start"] == True].copy()

    # Test: short all small-cap casual picks
    target = first_recs[(first_recs["bucket"] == "Small-cap") &
                        (first_recs["seq_cat"] == "no_hold")]
    long_alpha = target["spy_relative_return"]
    long_abs = target["absolute_return"]
    n = len(long_alpha)
    t, p = stats.ttest_1samp(long_alpha, 0)

    print(f"\n  Small-cap no-hold first recommendations (N={n}):")
    print(f"    Long 1Y SPY-relative alpha: {long_alpha.mean()*100:+.2f}% (t={t:.2f}, p={p:.4f})")
    print(f"    Long 1Y absolute return:    {long_abs.mean()*100:+.2f}%")
    print(f"    → SHORT position alpha:     {-long_alpha.mean()*100:+.2f}%")
    print(f"    → SHORT absolute return:    {-long_abs.mean()*100:+.2f}%")

    # Hit rate analysis
    losers = (long_alpha < 0).sum()
    abs_losers = (long_abs < 0).sum()
    print(f"\n    Negative SPY-rel alpha: {losers}/{n} ({losers/n*100:.1f}%)")
    print(f"    Negative absolute return: {abs_losers}/{n} ({abs_losers/n*100:.1f}%)")

    # Refine by VIX
    print(f"\n  Further split by VIX regime at signal:")
    for vix in ["Low", "Moderate", "High"]:
        sub = target[target["vix_regime"] == vix]
        if len(sub) < 20:
            print(f"    {vix} VIX: N={len(sub)} (insufficient)")
            continue
        t, p = stats.ttest_1samp(sub["spy_relative_return"], 0)
        print(f"    {vix:>8} VIX: N={len(sub):>4}, long α={sub['spy_relative_return'].mean()*100:+.2f}%, "
              f"t={t:.2f}, p={p:.4f}, → SHORT α={-sub['spy_relative_return'].mean()*100:+.2f}%")

    # Compare: small-cap + small-cap mid-cap combined casual
    section("Q1b: SHORTING NON-LARGE CASUAL PICKS")
    print("  What if we expand to include mid-cap? (larger universe)")
    target2 = first_recs[(first_recs["bucket"].isin(["Mid-cap", "Small-cap"])) &
                         (first_recs["seq_cat"] == "no_hold")]
    long_alpha2 = target2["spy_relative_return"]
    n2 = len(long_alpha2)
    t2, p2 = stats.ttest_1samp(long_alpha2, 0)
    print(f"\n  Mid+Small-cap no-hold first recs (N={n2}):")
    print(f"    Long α: {long_alpha2.mean()*100:+.2f}%, t={t2:.2f}, p={p2:.4f}")
    print(f"    → SHORT α: {-long_alpha2.mean()*100:+.2f}%")


# ── Q2: Portfolio dip-buy — drawdown threshold sweep ──────────────────

def dip_buy_thresholds(df_1y):
    section("Q2: PORTFOLIO DIP-BUY — OPTIMAL DRAWDOWN THRESHOLD")
    print("  For portfolio-tier first recommendations, split by prior 90d return.")
    print("  Does forward alpha systematically rise as we focus on deeper dips?")

    first_recs = df_1y[(df_1y["is_sequence_start"] == True) &
                       (df_1y["seq_cat"] == "portfolio") &
                       (df_1y["lookback_alpha_90d"].notna())].copy()
    print(f"\n  Portfolio first recs with 90d lookback: {len(first_recs)}")

    # Absolute drawdown thresholds (not SPY-relative)
    print(f"\n  By 90d SPY-relative lookback return threshold:")
    print(f"  {'Threshold':<15} {'N':>5} {'Fwd Alpha':>12} {'Med':>10} {'p':>8}")
    for thresh_pct in [0, -5, -10, -15, -20, -25]:
        sub = first_recs[first_recs["lookback_alpha_90d"] <= thresh_pct / 100]
        n = len(sub)
        if n < 20:
            print(f"  lookback ≤ {thresh_pct}%  N={n:>5} (insufficient)")
            continue
        a = sub["spy_relative_return"]
        t, p = stats.ttest_1samp(a, 0)
        print(f"  lookback ≤ {thresh_pct:>3}%  N={n:>5}  {a.mean()*100:>+10.2f}%  "
              f"{a.median()*100:>+8.2f}%  {p:>7.4f}")

    # Compare against portfolio-tier baseline
    baseline = first_recs["spy_relative_return"].mean()
    print(f"\n  Portfolio baseline (no filter): α = {baseline*100:+.2f}%")

    # Try 30d lookback too
    print(f"\n  By 30d SPY-relative lookback return threshold (short-term dip):")
    print(f"  {'Threshold':<15} {'N':>5} {'Fwd Alpha':>12} {'Med':>10} {'p':>8}")
    for thresh_pct in [0, -3, -5, -10, -15]:
        sub = first_recs[(first_recs["lookback_alpha_30d"].notna()) &
                         (first_recs["lookback_alpha_30d"] <= thresh_pct / 100)]
        n = len(sub)
        if n < 20:
            print(f"  lookback ≤ {thresh_pct}%  N={n:>5} (insufficient)")
            continue
        a = sub["spy_relative_return"]
        t, p = stats.ttest_1samp(a, 0)
        print(f"  lookback ≤ {thresh_pct:>3}%  N={n:>5}  {a.mean()*100:>+10.2f}%  "
              f"{a.median()*100:>+8.2f}%  {p:>7.4f}")


# ── Q3: VIX × conviction tier ──────────────────────────────────────────

def vix_conviction(df_1y):
    section("Q3: VIX × CONVICTION TIER — WHEN TO FOLLOW CRAMER?")

    first_recs = df_1y[(df_1y["is_sequence_start"] == True) &
                       (df_1y["vix_regime"].notna())].copy()

    print(f"\n  Forward 1Y alpha by VIX regime × conviction tier:")
    print(f"  {'VIX':<10} {'Tier':<12} {'N':>5} {'Alpha':>10} {'Med':>10} {'p':>8}")
    for vix in ["Low", "Moderate", "High"]:
        for tier in ["portfolio", "hold_rec", "no_hold"]:
            sub = first_recs[(first_recs["vix_regime"] == vix) &
                             (first_recs["seq_cat"] == tier)]["spy_relative_return"]
            n = len(sub)
            if n < 10:
                print(f"  {vix:<10} {tier:<12} N={n} (insufficient)")
                continue
            t, p = stats.ttest_1samp(sub, 0)
            print(f"  {vix:<10} {tier:<12} {n:>5} {sub.mean()*100:>+8.2f}% "
                  f"{sub.median()*100:>+8.2f}% {p:>7.4f}")

    # Specific VIX thresholds
    print(f"\n  Portfolio tier by exact VIX level at signal:")
    port = first_recs[first_recs["seq_cat"] == "portfolio"].copy()
    port = port[port["vix_at_signal"].notna()]
    for vix_thresh in [15, 18, 20, 25, 30]:
        below = port[port["vix_at_signal"] < vix_thresh]["spy_relative_return"]
        above = port[port["vix_at_signal"] >= vix_thresh]["spy_relative_return"]
        if len(below) >= 20 and len(above) >= 20:
            tb, pb = stats.ttest_1samp(below, 0)
            ta, pa = stats.ttest_1samp(above, 0)
            print(f"    VIX < {vix_thresh}: N={len(below):>3}, α={below.mean()*100:+.2f}% (p={pb:.4f}) "
                  f"| VIX ≥ {vix_thresh}: N={len(above):>3}, α={above.mean()*100:+.2f}% (p={pa:.4f})")


# ── Q4: Combined feature — portfolio + dip + VIX ──────────────────────

def combined_signal(df_1y):
    section("Q4: COMBINED SIGNAL — PORTFOLIO + PRIOR DRAWDOWN + VIX")
    print("  Best Cramer trade: portfolio stock, recent decline, elevated VIX")

    first_recs = df_1y[(df_1y["is_sequence_start"] == True) &
                       (df_1y["seq_cat"] == "portfolio") &
                       (df_1y["lookback_alpha_90d"].notna()) &
                       (df_1y["vix_at_signal"].notna())].copy()

    # Split by dip (90d SPY-rel lookback < 0) vs no dip
    # And by VIX high/low
    print(f"\n  Portfolio first recs: N={len(first_recs)}")
    print(f"  {'Condition':<45} {'N':>5} {'Alpha':>10} {'Med':>10} {'p':>8}")

    conditions = [
        ("Any portfolio pick (baseline)", first_recs),
        ("Portfolio + dip (90d α ≤ 0)",
         first_recs[first_recs["lookback_alpha_90d"] <= 0]),
        ("Portfolio + deep dip (90d α ≤ -10%)",
         first_recs[first_recs["lookback_alpha_90d"] <= -0.10]),
        ("Portfolio + VIX ≥ 20 (elevated)",
         first_recs[first_recs["vix_at_signal"] >= 20]),
        ("Portfolio + VIX ≥ 25 (high)",
         first_recs[first_recs["vix_at_signal"] >= 25]),
        ("Portfolio + dip + VIX ≥ 20",
         first_recs[(first_recs["lookback_alpha_90d"] <= 0) &
                    (first_recs["vix_at_signal"] >= 20)]),
        ("Portfolio + deep dip + VIX ≥ 20",
         first_recs[(first_recs["lookback_alpha_90d"] <= -0.10) &
                    (first_recs["vix_at_signal"] >= 20)]),
    ]

    for label, sub in conditions:
        a = sub["spy_relative_return"]
        n = len(a)
        if n < 10:
            print(f"  {label:<45} {n:>5} (insufficient)")
            continue
        t, p = stats.ttest_1samp(a, 0)
        print(f"  {label:<45} {n:>5} {a.mean()*100:>+8.2f}% {a.median()*100:>+8.2f}% {p:>7.4f}")


def main():
    df_1y = load_data()
    inverse_small_cap(df_1y)
    dip_buy_thresholds(df_1y)
    vix_conviction(df_1y)
    combined_signal(df_1y)
    section("DONE")


if __name__ == "__main__":
    main()
