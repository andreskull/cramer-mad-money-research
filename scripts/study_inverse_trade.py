"""
Correct accounting of the inverse small-cap casual trade.

Distinguish:
  (A) Simple short — P&L = -stock_return (absolute)
  (B) Long-short pair — short stock, long SPY = SPY_return - stock_return (market-neutral)

The SPY-relative alpha of the LONG (-24.62%) only equals the P&L of a
market-neutral pair. A standalone short earns the negative of the absolute
return, which must then be benchmarked against SPY to judge the alternative.
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


def trade_stats(sub, label):
    """Compute long, simple short, and pair-trade statistics for one subset."""
    if len(sub) < 10:
        print(f"  {label:<45} N={len(sub)} (insufficient)")
        return

    # Long: absolute return of stock
    long_abs = sub["absolute_return"]
    # SPY during same window: absolute - SPY_relative
    spy_ret = sub["absolute_return"] - sub["spy_relative_return"]
    # Simple short: -stock return
    short_abs = -long_abs
    # Short alpha vs SPY = short_abs - spy_ret = -stock - spy
    short_vs_spy = short_abs - spy_ret
    # Pair trade: short stock + long SPY = spy_ret - stock_ret = -(stock - spy) = -SPY_relative
    pair = -sub["spy_relative_return"]

    n = len(sub)
    _, p_long_alpha = stats.ttest_1samp(sub["spy_relative_return"], 0)
    _, p_short_abs = stats.ttest_1samp(short_abs, 0)
    _, p_short_vs_spy = stats.ttest_1samp(short_vs_spy, 0)
    _, p_pair = stats.ttest_1samp(pair, 0)

    print(f"\n  {label}  (N={n})")
    print(f"    Long absolute return:         {long_abs.mean()*100:>+7.2f}%")
    print(f"    SPY during same windows:      {spy_ret.mean()*100:>+7.2f}%")
    print(f"    Long SPY-relative alpha:      {sub['spy_relative_return'].mean()*100:>+7.2f}%   (p={p_long_alpha:.4f})")
    print(f"    Simple short absolute P&L:    {short_abs.mean()*100:>+7.2f}%   (p={p_short_abs:.4f})")
    print(f"    Simple short vs SPY (α):      {short_vs_spy.mean()*100:>+7.2f}%   (p={p_short_vs_spy:.4f})")
    print(f"    Pair trade (short + long SPY):{pair.mean()*100:>+7.2f}%   (p={p_pair:.4f})")


def main():
    df = pd.read_csv(os.path.join(DATA, "cramer_sequenced.csv"))
    seq = pd.read_csv(os.path.join(DATA, "sequence_summary.csv"))
    mcaps = pd.read_csv(os.path.join(DATA, "market_caps.csv"))

    seq["seq_cat"] = seq.apply(seq_category, axis=1)
    df_1y = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    df_1y = df_1y.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
    df_1y = df_1y.merge(mcaps[["ticker", "bucket"]], left_on="ticker_symbol",
                        right_on="ticker", how="left")
    df_1y["vix_regime"] = df_1y["vix_at_signal"].apply(vix_regime)

    first_recs = df_1y[df_1y["is_sequence_start"] == True].copy()
    small_casual = first_recs[(first_recs["bucket"] == "Small-cap") &
                              (first_recs["seq_cat"] == "no_hold")]

    print("="*70)
    print("  INVERSE TRADE ON SMALL-CAP CASUAL PICKS — CORRECT ACCOUNTING")
    print("  Pruned signals, first-recommendation entries, 1Y horizon")
    print("="*70)

    trade_stats(small_casual, "All small-cap casual")

    for vix in ["Low", "Moderate", "High"]:
        sub = small_casual[small_casual["vix_regime"] == vix]
        trade_stats(sub, f"Small-cap casual + {vix} VIX")

    # VIX < 30 (profitable regime for inverse)
    sub = small_casual[small_casual["vix_at_signal"] < 30]
    trade_stats(sub, "Small-cap casual + VIX < 30")

    # Mid+small
    ms_casual = first_recs[(first_recs["bucket"].isin(["Mid-cap", "Small-cap"])) &
                           (first_recs["seq_cat"] == "no_hold")]
    trade_stats(ms_casual, "Mid+Small-cap casual (all VIX)")

    sub = ms_casual[ms_casual["vix_at_signal"] < 30]
    trade_stats(sub, "Mid+Small-cap casual + VIX < 30")


if __name__ == "__main__":
    main()
