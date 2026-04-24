"""
Validates: paper §6.4b, Table 13 (dip-buy gradient — all-kept-signals robustness)
           and the +24.2% / N=156 / p=0.005 repeat-mention sub-cell cited in the
           "not driven by the first recommendation" follow-up paragraph.

Sensitivity check (Pillar 3, Section 6.4):

Does the dip-buy result on portfolio names hold when we use *all* kept signals
on portfolio stocks, rather than only the first recommendation in each
sequence?

The first-recommendation-only filter (is_sequence_start = True) was chosen so
that each sequence is counted once. But every kept signal is, by construction,
an independent decision moment for the viewer (the sequence model has already
pruned redundant mentions). A repeat mention of a Trust holding several months
later, after further drawdown, is a fresh dip-buy decision and arguably should
count.

Lookback returns in `momentum_signals.csv` are populated only for first-rec
signals, so we compute the 90-day absolute and SPY-relative lookback for
*all* kept portfolio signals from `daily_price_matrix.parquet`, then re-run
the dip-buy sweep on the broader sample.

Outputs:
  - sample sizes and aggregate alpha (first-rec only vs all kept signals)
  - dip-buy sweep on SPY-relative 90d lookback (both samples)
  - dip-buy sweep on absolute 90d return (both samples)
  - headline contrast at the −15% threshold
  - sub-cell on repeat-mention dip-buys only
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

SPY_FIGI = "BBG000BDTF76"
LOOKBACK_DAYS = 90


def seq_category(row):
    if row.get("has_cramer_owns", False):
        return "portfolio"
    elif row.get("has_hold_recommendation", False):
        return "hold_rec"
    return "no_hold"


def section(title):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")


def load_data():
    df = pd.read_csv(os.path.join(DATA, "cramer_sequenced.csv"))
    seq = pd.read_csv(os.path.join(DATA, "sequence_summary.csv"))
    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    df_1y = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    df_1y = df_1y.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
    df_1y["signal_date"] = pd.to_datetime(df_1y["signal_date"])
    return df_1y


def compute_lookback_for_signals(signals_df, prices, spy_series):
    """For each row in signals_df, compute 90d absolute and SPY-relative
    lookback returns from the daily price matrix.

    Returns a Series indexed like signals_df with two columns added.
    """
    out_abs = []
    out_alpha = []
    prices = prices.copy()
    prices["adj_close"] = prices["adj_close"].astype(float)
    by_figi = {fid: g.set_index("trade_date")["adj_close"].sort_index()
               for fid, g in prices.groupby("figi_id")}
    spy_series = spy_series.astype(float)

    for _, row in signals_df.iterrows():
        fid = row["figi_id"]
        sd = row["signal_date"]
        if pd.isna(fid) or fid not in by_figi:
            out_abs.append(np.nan); out_alpha.append(np.nan); continue
        s = by_figi[fid]
        # last close on or before signal_date
        s_le = s.loc[:sd]
        if s_le.empty:
            out_abs.append(np.nan); out_alpha.append(np.nan); continue
        end_price = s_le.iloc[-1]
        # last close on or before (signal_date − 90d)
        cutoff = sd - pd.Timedelta(days=LOOKBACK_DAYS)
        s_lb = s.loc[:cutoff]
        if s_lb.empty:
            out_abs.append(np.nan); out_alpha.append(np.nan); continue
        start_price = s_lb.iloc[-1]
        if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
            out_abs.append(np.nan); out_alpha.append(np.nan); continue
        ret_abs = end_price / start_price - 1.0

        # SPY return over the same window
        spy_le = spy_series.loc[:sd]
        spy_lb = spy_series.loc[:cutoff]
        if spy_le.empty or spy_lb.empty:
            out_abs.append(ret_abs); out_alpha.append(np.nan); continue
        spy_ret = spy_le.iloc[-1] / spy_lb.iloc[-1] - 1.0
        out_abs.append(ret_abs)
        out_alpha.append(ret_abs - spy_ret)

    return pd.Series(out_abs, index=signals_df.index, name="lookback_ret_90d"), \
           pd.Series(out_alpha, index=signals_df.index, name="lookback_alpha_90d")


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
        a = sub["spy_relative_return"].dropna()
        if len(a) < 5:
            print(f"  ≤ {thresh*100:>+6.1f}%       {n:>6}  (alpha NA)")
            continue
        t, p = stats.ttest_1samp(a, 0)
        win = (a > 0).mean() * 100
        print(f"  ≤ {thresh*100:>+6.1f}%       {n:>6} "
              f"{a.mean()*100:>+9.2f}% {a.median()*100:>+9.2f}% "
              f"{win:>6.1f}% {p:>8.4f}")


def stat_line(a):
    if len(a) < 5:
        return f"N={len(a)} (insufficient)"
    t, p = stats.ttest_1samp(a, 0)
    return f"N={len(a):>4}  alpha={a.mean()*100:>+7.2f}%  median={a.median()*100:>+7.2f}%  p={p:.4f}"


def main():
    df_1y = load_data()

    portfolio_all = df_1y[df_1y["seq_cat"] == "portfolio"].copy()
    portfolio_first = portfolio_all[portfolio_all["is_sequence_start"] == True].copy()
    nohold_first = df_1y[
        (df_1y["seq_cat"] == "no_hold") & (df_1y["is_sequence_start"] == True)
    ].copy()

    section("0. Sample sizes")
    print(f"  Portfolio kept signals (all):              N = {len(portfolio_all):>6}")
    print(f"  Portfolio kept signals (first rec only):   N = {len(portfolio_first):>6}")
    print(f"  Portfolio kept signals (repeat mention):   N = {len(portfolio_all)-len(portfolio_first):>6}")
    print(f"  Casual-buy kept signals (first rec only):  N = {len(nohold_first):>6}")

    section("1. Aggregate 1Y alpha — first-rec only vs all kept signals")
    for label, sub in [
        ("Portfolio first-rec only", portfolio_first),
        ("Portfolio all kept signals", portfolio_all),
        ("Casual-buy first-rec only", nohold_first),
    ]:
        a = sub["spy_relative_return"].dropna()
        print(f"  {label:<32} {stat_line(a)}")

    print("\n[loading daily price matrix to compute lookback for repeat-mention signals]")
    prices = pd.read_parquet(os.path.join(DATA, "daily_price_matrix.parquet"))
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    spy_series = (
        prices[prices["figi_id"] == SPY_FIGI]
        .set_index("trade_date")["adj_close"]
        .sort_index()
    )
    print(f"  SPY price points: {len(spy_series)}  range {spy_series.index.min().date()} → {spy_series.index.max().date()}")

    print("  computing lookback for ALL kept portfolio signals...")
    abs_s, alpha_s = compute_lookback_for_signals(portfolio_all, prices, spy_series)
    portfolio_all["lookback_ret_90d"] = abs_s
    portfolio_all["lookback_alpha_90d"] = alpha_s
    print(f"  computed lookback on {alpha_s.notna().sum()} / {len(portfolio_all)} signals")

    print("  computing lookback for casual-buy first-rec signals (same method, for apples-to-apples)...")
    abs_n, alpha_n = compute_lookback_for_signals(nohold_first, prices, spy_series)
    nohold_first["lookback_ret_90d"] = abs_n
    nohold_first["lookback_alpha_90d"] = alpha_n

    section("2. Dip-buy sweep — SPY-RELATIVE 90d lookback")
    spy_thresholds = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    print("\n  PORTFOLIO — first-recommendation only (paper Section 6.4 baseline)")
    pf_first_lb = portfolio_all[portfolio_all["is_sequence_start"] == True].dropna(subset=["lookback_alpha_90d"])
    sweep(pf_first_lb, "lookback_alpha_90d", spy_thresholds, "")
    print("\n  PORTFOLIO — all kept signals (sensitivity)")
    sweep(portfolio_all.dropna(subset=["lookback_alpha_90d"]),
          "lookback_alpha_90d", spy_thresholds, "")
    print("\n  CASUAL-BUY — first-recommendation only (paper baseline)")
    sweep(nohold_first.dropna(subset=["lookback_alpha_90d"]),
          "lookback_alpha_90d", spy_thresholds, "")

    section("3. Dip-buy sweep — ABSOLUTE 90d return")
    abs_thresholds = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    print("\n  PORTFOLIO — first-recommendation only")
    sweep(pf_first_lb, "lookback_ret_90d", abs_thresholds, "")
    print("\n  PORTFOLIO — all kept signals")
    sweep(portfolio_all.dropna(subset=["lookback_ret_90d"]),
          "lookback_ret_90d", abs_thresholds, "")

    section("4. Headline contrast at −15% SPY-relative threshold")
    th = -0.15
    pf_first_dip = pf_first_lb[pf_first_lb["lookback_alpha_90d"] <= th]["spy_relative_return"].dropna()
    pf_all_dip = portfolio_all[portfolio_all["lookback_alpha_90d"] <= th]["spy_relative_return"].dropna()
    nh_first_dip = nohold_first[nohold_first["lookback_alpha_90d"] <= th]["spy_relative_return"].dropna()
    print(f"  Portfolio  first-rec only  : {stat_line(pf_first_dip)}")
    print(f"  Portfolio  all kept signals: {stat_line(pf_all_dip)}")
    print(f"  Casual-buy first-rec only  : {stat_line(nh_first_dip)}")
    if len(pf_all_dip) >= 5 and len(nh_first_dip) >= 5:
        print(f"  Portfolio−CasualBuy spread (all kept signals vs first-rec casual): "
              f"{(pf_all_dip.mean()-nh_first_dip.mean())*100:+.2f} pp")
    if len(pf_first_dip) >= 5 and len(nh_first_dip) >= 5:
        print(f"  Portfolio−CasualBuy spread (first-rec only): "
              f"{(pf_first_dip.mean()-nh_first_dip.mean())*100:+.2f} pp")

    section("5b. VIX × dip-buy (all kept portfolio signals)")
    pa = portfolio_all.dropna(subset=["lookback_alpha_90d", "vix_at_signal"]).copy()
    pa["vix_at_signal"] = pd.to_numeric(pa["vix_at_signal"], errors="coerce")
    print(f"  Portfolio + VIX<20 (any lookback)            : {stat_line(pa[pa['vix_at_signal']<20]['spy_relative_return'].dropna())}")
    print(f"  Portfolio + VIX<20 + lookback<=-10% SPY-rel  : {stat_line(pa[(pa['vix_at_signal']<20)&(pa['lookback_alpha_90d']<=-0.10)]['spy_relative_return'].dropna())}")
    print(f"  Portfolio + VIX<20 + lookback<=-15% SPY-rel  : {stat_line(pa[(pa['vix_at_signal']<20)&(pa['lookback_alpha_90d']<=-0.15)]['spy_relative_return'].dropna())}")
    print(f"  Portfolio + VIX<20 + lookback<=-10% absolute : {stat_line(pa[(pa['vix_at_signal']<20)&(pa['lookback_ret_90d']<=-0.10)]['spy_relative_return'].dropna())}")
    print(f"  Portfolio + VIX<20 + lookback<=-15% absolute : {stat_line(pa[(pa['vix_at_signal']<20)&(pa['lookback_ret_90d']<=-0.15)]['spy_relative_return'].dropna())}")
    print()
    print(f"  Portfolio + VIX<30 + lookback<=-15% SPY-rel  : {stat_line(pa[(pa['vix_at_signal']<30)&(pa['lookback_alpha_90d']<=-0.15)]['spy_relative_return'].dropna())}")
    print(f"  Portfolio + VIX<30 + lookback<=-15% absolute : {stat_line(pa[(pa['vix_at_signal']<30)&(pa['lookback_ret_90d']<=-0.15)]['spy_relative_return'].dropna())}")
    print(f"  Portfolio + VIX>=30 + lookback<=-15% SPY-rel : {stat_line(pa[(pa['vix_at_signal']>=30)&(pa['lookback_alpha_90d']<=-0.15)]['spy_relative_return'].dropna())}")

    section("5c. Casual-buy at -15% drawdowns, on the same axes (all kept)")
    nh_all = df_1y[df_1y["seq_cat"] == "no_hold"].copy()
    abs_n2, alpha_n2 = compute_lookback_for_signals(nh_all, prices, spy_series)
    nh_all["lookback_ret_90d"] = abs_n2
    nh_all["lookback_alpha_90d"] = alpha_n2
    print(f"  Casual-buy all kept signals (aggregate)            : {stat_line(nh_all['spy_relative_return'].dropna())}")
    print(f"  Casual-buy all kept signals, ≤ -15% SPY-rel        : {stat_line(nh_all[nh_all['lookback_alpha_90d']<=-0.15]['spy_relative_return'].dropna())}")
    print(f"  Casual-buy all kept signals, ≤ -15% absolute       : {stat_line(nh_all[nh_all['lookback_ret_90d']<=-0.15]['spy_relative_return'].dropna())}")
    print(f"  Casual-buy all kept signals, ≤ -10% SPY-rel        : {stat_line(nh_all[nh_all['lookback_alpha_90d']<=-0.10]['spy_relative_return'].dropna())}")
    print(f"  Casual-buy all kept signals, ≤ -10% absolute       : {stat_line(nh_all[nh_all['lookback_ret_90d']<=-0.10]['spy_relative_return'].dropna())}")
    print(f"  Casual-buy all kept signals, ≤  0% SPY-rel         : {stat_line(nh_all[nh_all['lookback_alpha_90d']<=0.0]['spy_relative_return'].dropna())}")
    print(f"  Casual-buy all kept signals, ≤  0% absolute        : {stat_line(nh_all[nh_all['lookback_ret_90d']<=0.0]['spy_relative_return'].dropna())}")

    section("5. Repeat-mention sub-cell")
    repeat = portfolio_all[portfolio_all["is_sequence_start"] == False]
    repeat_lb = repeat.dropna(subset=["lookback_alpha_90d"])
    print(f"  All repeat-mention portfolio signals w/ lookback: N={len(repeat_lb)}")
    print(f"  Aggregate alpha on repeat mentions: {stat_line(repeat_lb['spy_relative_return'].dropna())}")
    print()
    print("  Dip-buy sweep on repeat mentions only (SPY-relative 90d):")
    sweep(repeat_lb, "lookback_alpha_90d", spy_thresholds, "")
    print()
    print("  Dip-buy sweep on repeat mentions only (absolute 90d):")
    sweep(repeat_lb, "lookback_ret_90d", abs_thresholds, "")


if __name__ == "__main__":
    main()
