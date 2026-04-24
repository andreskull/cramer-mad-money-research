"""
Study B: Momentum, Reversal, and Cramer's Selection Patterns
=============================================================
Tests whether the conviction gradient is explained by momentum selection —
i.e. Cramer recommends stocks that have already been running up.

Reads: data/cramer_sequenced.csv, data/sequence_summary.csv,
       data/daily_price_matrix.parquet
Writes: data/momentum_signals.csv, data/momentum_analysis_results.csv
"""

import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

LOOKBACK_WINDOWS = [30, 60, 90, 180]
TIER_ORDER = ["portfolio", "hold_rec", "no_hold"]
TIER_LABELS = {"portfolio": "Portfolio", "hold_rec": "Hold-rec", "no_hold": "No-hold"}


def seq_category(row):
    if row.get("has_cramer_owns", False):
        return "portfolio"
    elif row.get("has_hold_recommendation", False):
        return "hold_rec"
    return "no_hold"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── Step 1: Compute pre-recommendation returns ───────────────────────

def compute_lookback_returns(df_1y, price_matrix, figi_map):
    """Join signal dates to price history and compute lookback returns."""
    section("STEP 1: COMPUTE PRE-RECOMMENDATION RETURNS")

    first_recs = df_1y[df_1y["is_sequence_start"] == True].copy()
    print(f"  First-recommendation entries (1Y, kept): {len(first_recs)}")

    # Build price lookup: figi_id → date → adj_close
    pm = price_matrix.copy()
    pm["trade_date"] = pd.to_datetime(pm["trade_date"])
    pm = pm.sort_values(["figi_id", "trade_date"])

    # SPY figi from figi_map
    spy_figi = figi_map[figi_map["ticker_symbol"] == "SPY"]["figi_id"].values
    if len(spy_figi) == 0:
        print("  WARNING: SPY not found in figi_map. Using absolute returns only.")
        spy_prices = None
    else:
        spy_figi = spy_figi[0]
        spy_prices = pm[pm["figi_id"] == spy_figi][["trade_date", "adj_close"]].copy()
        spy_prices = spy_prices.set_index("trade_date")["adj_close"].sort_index()
        print(f"  SPY price series: {len(spy_prices)} trading days")

    lookback_rows = []
    first_recs["signal_date_dt"] = pd.to_datetime(first_recs["signal_date"])
    unique_figis = first_recs["figi_id"].unique()

    # Pre-group price data for speed
    price_groups = {figi: grp.set_index("trade_date")["adj_close"].sort_index()
                    for figi, grp in pm[pm["figi_id"].isin(unique_figis)].groupby("figi_id")}

    total = len(first_recs)
    for idx, (_, row) in enumerate(first_recs.iterrows()):
        if idx % 500 == 0:
            print(f"\r  Processing signal {idx+1}/{total}...", end="", flush=True)

        figi = row["figi_id"]
        sig_date = row["signal_date_dt"]

        if figi not in price_groups:
            continue

        prices = price_groups[figi]
        # Find the closest trading day on or before signal date
        valid_dates = prices.index[prices.index <= sig_date]
        if len(valid_dates) == 0:
            continue
        sig_price_date = valid_dates[-1]
        sig_price = prices[sig_price_date]

        rec = {
            "signal_id": row["signal_id"],
            "sequence_id": row["sequence_id"],
            "ticker_symbol": row["ticker_symbol"],
            "figi_id": figi,
            "signal_date": row["signal_date"],
            "seq_cat": row["seq_cat"],
            "spy_relative_return": row["spy_relative_return"],
            "gics_sector": row["gics_sector"],
        }

        for window in LOOKBACK_WINDOWS:
            lookback_dates = prices.index[prices.index <= sig_price_date]
            if len(lookback_dates) <= window:
                rec[f"lookback_ret_{window}d"] = None
                rec[f"lookback_alpha_{window}d"] = None
                continue

            lb_price = lookback_dates[-window - 1]
            stock_ret = sig_price / prices[lb_price] - 1
            rec[f"lookback_ret_{window}d"] = stock_ret

            if spy_prices is not None:
                spy_valid = spy_prices.index[spy_prices.index <= sig_price_date]
                spy_lb = spy_prices.index[spy_prices.index <= lb_price]
                if len(spy_valid) > 0 and len(spy_lb) > 0:
                    spy_ret = spy_prices[spy_valid[-1]] / spy_prices[spy_lb[-1]] - 1
                    rec[f"lookback_alpha_{window}d"] = stock_ret - spy_ret
                else:
                    rec[f"lookback_alpha_{window}d"] = None
            else:
                rec[f"lookback_alpha_{window}d"] = None

        lookback_rows.append(rec)

    print(f"\r  Processed {total} signals, {len(lookback_rows)} with price data")

    lb_df = pd.DataFrame(lookback_rows)

    # Ensure numeric dtypes for all computed columns
    for w in LOOKBACK_WINDOWS:
        lb_df[f"lookback_ret_{w}d"] = pd.to_numeric(lb_df[f"lookback_ret_{w}d"], errors="coerce")
        lb_df[f"lookback_alpha_{w}d"] = pd.to_numeric(lb_df[f"lookback_alpha_{w}d"], errors="coerce")
    lb_df["spy_relative_return"] = pd.to_numeric(lb_df["spy_relative_return"], errors="coerce")

    print(f"\n  Coverage rates:")
    for w in LOOKBACK_WINDOWS:
        col = f"lookback_ret_{w}d"
        n = lb_df[col].notna().sum()
        print(f"    {w}d: {n}/{len(lb_df)} ({n/len(lb_df)*100:.1f}%)")

    out_path = os.path.join(DATA, "momentum_signals.csv")
    lb_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return lb_df


# ── Step 2: Prior-return profiles by tier ─────────────────────────────

def prior_return_profiles(lb_df):
    section("STEP 2: PRIOR-RETURN PROFILES BY TIER (B-H1, B-H2)")

    primary_col = "lookback_alpha_90d"
    valid = lb_df[lb_df[primary_col].notna()].copy()
    print(f"  Signals with 90d lookback alpha: {len(valid)}")

    print("\n  Mean lookback alpha by tier and window:")
    for w in LOOKBACK_WINDOWS:
        col = f"lookback_alpha_{w}d"
        sub = lb_df[lb_df[col].notna()]
        print(f"\n    {w}d window:")
        for tier in TIER_ORDER:
            tdata = sub[sub["seq_cat"] == tier][col]
            if len(tdata) < 10:
                continue
            t, p = stats.ttest_1samp(tdata, 0)
            print(f"      {TIER_LABELS[tier]:>10}: N={len(tdata):>4}, "
                  f"mean={tdata.mean()*100:+.2f}%, med={tdata.median()*100:+.2f}%, "
                  f"t={t:.2f}, p={p:.4f}")

    # KS tests between tiers
    print(f"\n  Tier comparison (90d lookback alpha, KS test):")
    pairs = [("portfolio", "no_hold"), ("portfolio", "hold_rec"), ("hold_rec", "no_hold")]
    for a, b in pairs:
        a_data = valid[valid["seq_cat"] == a][primary_col]
        b_data = valid[valid["seq_cat"] == b][primary_col]
        if len(a_data) >= 10 and len(b_data) >= 10:
            ks, p = stats.ks_2samp(a_data, b_data)
            rs, p_rs = stats.mannwhitneyu(a_data, b_data, alternative="two-sided")
            print(f"    {TIER_LABELS[a]} vs {TIER_LABELS[b]}: KS={ks:.3f} p={p:.4f}, "
                  f"rank-sum p={p_rs:.4f}")

    port_mean = valid[valid["seq_cat"] == "portfolio"][primary_col].mean()
    nohold_mean = valid[valid["seq_cat"] == "no_hold"][primary_col].mean()
    if port_mean > nohold_mean:
        print(f"\n  → B-H1 evidence: portfolio picks DO show higher prior momentum "
              f"({port_mean*100:+.2f}% vs {nohold_mean*100:+.2f}%)")
    else:
        print(f"\n  → B-H1 NOT supported: portfolio picks do not show higher prior momentum")


# ── Step 3: Momentum control ─────────────────────────────────────────

def momentum_control(lb_df):
    section("STEP 3: MOMENTUM CONTROL — DOES PRIOR RETURN EXPLAIN FORWARD ALPHA? (B-H3)")

    primary_col = "lookback_alpha_90d"
    valid = lb_df[lb_df[primary_col].notna() & lb_df["spy_relative_return"].notna()].copy()
    print(f"  Signals with both 90d lookback and 1Y forward alpha: {len(valid)}")

    valid["lb_quintile"] = pd.qcut(valid[primary_col], 5, labels=["Q1\n(losers)", "Q2", "Q3", "Q4", "Q5\n(winners)"],
                                   duplicates="drop")

    print("\n  Forward 1Y alpha by lookback quintile × conviction tier:")
    print(f"  {'Quintile':<12}", end="")
    for tier in TIER_ORDER:
        print(f"  {TIER_LABELS[tier]:>14}", end="")
    print(f"  {'P>NH diff':>10}")

    heatmap_data = {}

    for q in valid["lb_quintile"].cat.categories:
        qdata = valid[valid["lb_quintile"] == q]
        q_label = str(q).replace("\n", " ")
        print(f"  {q_label:<12}", end="")
        row_data = {}
        for tier in TIER_ORDER:
            tdata = qdata[qdata["seq_cat"] == tier]["spy_relative_return"]
            if len(tdata) >= 10:
                mean_a = tdata.mean()
                print(f"  {mean_a*100:>+7.2f}% ({len(tdata):>3})", end="")
                row_data[tier] = mean_a
            else:
                print(f"  {'n/a':>14}", end="")
                row_data[tier] = None

        # Test portfolio vs no-hold within quintile
        port = qdata[qdata["seq_cat"] == "portfolio"]["spy_relative_return"]
        nh = qdata[qdata["seq_cat"] == "no_hold"]["spy_relative_return"]
        if len(port) >= 10 and len(nh) >= 10:
            _, p = stats.ttest_ind(port, nh)
            print(f"  p={p:.3f}", end="")
        print()
        heatmap_data[str(q).replace("\n", " ")] = row_data

    # Summary: does gradient survive?
    gradient_survives = 0
    gradient_tested = 0
    for q in valid["lb_quintile"].cat.categories:
        qdata = valid[valid["lb_quintile"] == q]
        port = qdata[qdata["seq_cat"] == "portfolio"]["spy_relative_return"]
        nh = qdata[qdata["seq_cat"] == "no_hold"]["spy_relative_return"]
        if len(port) >= 10 and len(nh) >= 10:
            gradient_tested += 1
            if port.mean() > nh.mean():
                gradient_survives += 1

    print(f"\n  Gradient direction (portfolio > no-hold): {gradient_survives}/{gradient_tested} quintiles")
    if gradient_survives >= gradient_tested * 0.6:
        print("  → B-H3 REJECTED: conviction gradient SURVIVES after momentum control")
        print("    The gradient is not reducible to momentum selection.")
    else:
        print("  → B-H3 SUPPORTED: conviction gradient COLLAPSES within momentum quintiles")
        print("    The gradient is substantially a momentum effect — major paper revision needed.")

    return valid, heatmap_data


# ── Step 4: Recommendation timing (B-H4) ─────────────────────────────

def recommendation_timing(df_full, price_matrix, figi_map):
    section("STEP 4: RECOMMENDATION TIMING — DOES CRAMER PROMOTE WHEN WINNING? (B-H4)")

    # Portfolio tickers with cramer_owns
    port_signals = df_full[(df_full["hold_subtype"] == "cramer_owns")].copy()
    port_tickers = port_signals["ticker_symbol"].unique()
    print(f"  Portfolio tickers with cramer_owns: {len(port_tickers)}")

    port_signals["signal_date_dt"] = pd.to_datetime(port_signals["signal_date"])
    port_signal_dates = set()
    ticker_signal_dates = {}
    for _, row in port_signals.iterrows():
        key = (row["figi_id"], row["signal_date_dt"].date())
        port_signal_dates.add(key)
        if row["figi_id"] not in ticker_signal_dates:
            ticker_signal_dates[row["figi_id"]] = set()
        ticker_signal_dates[row["figi_id"]].add(row["signal_date_dt"].date())

    pm = price_matrix.copy()
    pm["trade_date"] = pd.to_datetime(pm["trade_date"])
    pm = pm.sort_values(["figi_id", "trade_date"])

    figi_ticker = figi_map.set_index("figi_id")["ticker_symbol"].to_dict()
    port_figis = [f for f in ticker_signal_dates.keys()
                  if figi_ticker.get(f) in port_tickers]

    rec_trailing = []
    silence_trailing = []

    for figi in port_figis:
        prices = pm[pm["figi_id"] == figi].set_index("trade_date")["adj_close"].sort_index()
        if len(prices) < 35:
            continue

        signal_dates_for_figi = ticker_signal_dates.get(figi, set())
        if not signal_dates_for_figi:
            continue
        first_sig = pd.Timestamp(min(signal_dates_for_figi))
        last_sig = pd.Timestamp(max(signal_dates_for_figi))

        for i in range(30, len(prices)):
            trade_date = prices.index[i]
            if not (first_sig <= trade_date <= last_sig):
                continue
            date = trade_date.date()
            trailing_30d_ret = prices.iloc[i] / prices.iloc[i - 30] - 1

            if date in signal_dates_for_figi:
                rec_trailing.append(trailing_30d_ret)
            else:
                silence_trailing.append(trailing_30d_ret)

    rec_trailing = np.array(rec_trailing, dtype=np.float64)
    silence_trailing = np.array(silence_trailing, dtype=np.float64)
    rec_trailing = rec_trailing[np.isfinite(rec_trailing)]
    silence_trailing = silence_trailing[np.isfinite(silence_trailing)]

    print(f"  Recommendation-day observations: {len(rec_trailing)}")
    print(f"  Silence-day observations: {len(silence_trailing)}")

    if len(rec_trailing) >= 30 and len(silence_trailing) >= 30:
        rec_mean = np.mean(rec_trailing)
        sil_mean = np.mean(silence_trailing)
        t, p_iid = stats.ttest_ind(rec_trailing, silence_trailing)
        rs, p_rs = stats.mannwhitneyu(rec_trailing, silence_trailing, alternative="greater")

        # Defensible inference: trailing 30d returns are massively
        # autocorrelated within ticker (29/30 of each return shared with
        # the next observation) and clustered across tickers. Treating
        # stock-days as iid grossly understates the SE. We report a panel
        # FE regression with cluster-robust SEs at the ticker level, plus
        # a non-parametric ticker-block bootstrap, both of which absorb
        # the autocorrelation.
        from stats_helpers import (
            block_bootstrap_diff_p,
            panel_fe_diff_p,
        )

        figi_long = []
        ret_long = []
        treat_long = []
        for figi in port_figis:
            prices = pm[pm["figi_id"] == figi].set_index("trade_date")["adj_close"].sort_index()
            if len(prices) < 35:
                continue
            signal_dates_for_figi = ticker_signal_dates.get(figi, set())
            if not signal_dates_for_figi:
                continue
            first_sig = pd.Timestamp(min(signal_dates_for_figi))
            last_sig = pd.Timestamp(max(signal_dates_for_figi))
            for i in range(30, len(prices)):
                trade_date = prices.index[i]
                if not (first_sig <= trade_date <= last_sig):
                    continue
                date = trade_date.date()
                p_now = float(prices.iloc[i])
                p_then = float(prices.iloc[i - 30])
                if not (np.isfinite(p_now) and np.isfinite(p_then) and p_then > 0):
                    continue
                trailing = p_now / p_then - 1.0
                figi_long.append(figi)
                ret_long.append(trailing)
                treat_long.append(1 if date in signal_dates_for_figi else 0)
        figi_arr = np.asarray(figi_long)
        ret_arr = np.asarray(ret_long, dtype=float)
        treat_arr = np.asarray(treat_long, dtype=int)

        n_pan, g_pan, beta_pan, se_pan, p_pan = panel_fe_diff_p(
            ret_arr, treat_arr, figi_arr
        )
        n_bb, g_bb, diff_bb, p_bb, ci_low, ci_high = block_bootstrap_diff_p(
            ret_arr, treat_arr, figi_arr, b_iters=1000
        )

        print(f"\n  Trailing 30d return on recommendation days: {rec_mean*100:+.2f}%")
        print(f"  Trailing 30d return on silence days:         {sil_mean*100:+.2f}%")
        print(f"  Difference: {(rec_mean - sil_mean)*100:+.2f}pp")
        print(f"  iid t-test (NOT VALID, autocorr): t={t:.2f}, p={p_iid:.4f}")
        print(f"  Rank-sum (iid, NOT VALID): p={p_rs:.4f}")
        print(
            f"  Panel FE + ticker-clustered SE: beta={beta_pan*100:+.2f}pp  "
            f"se={se_pan*100:.2f}pp  p={p_pan:.4f}  (N={n_pan}, G={g_pan})"
        )
        print(
            f"  Ticker-block bootstrap (B=1000): diff={diff_bb*100:+.2f}pp  "
            f"95% CI=[{ci_low*100:+.2f}pp, {ci_high*100:+.2f}pp]  p={p_bb:.4f}"
        )

        if p_pan < 0.05 and beta_pan > 0:
            print(
                "\n  → B-H4 SUPPORTED: Cramer recommends portfolio stocks more often "
                "after positive runs (panel FE + cluster-robust SE)"
            )
        else:
            print("\n  → B-H4 NOT SUPPORTED: recommendation timing not linked to recent performance")

    # Event study: average trailing return in 60 days leading up to each rec
    print("\n  Computing event study (60-day leadup to recommendations)...")
    event_data = {"portfolio": [], "no_hold": []}

    for tier, subset in [("portfolio", df_full[df_full["hold_subtype"] == "cramer_owns"]),
                         ("no_hold", df_full[(df_full["hold_subtype"].isna()) &
                                             (df_full["is_sequence_start"] == True) &
                                             (df_full["kept"] == True)])]:
        subset = subset.copy()
        subset["signal_date_dt"] = pd.to_datetime(subset["signal_date"])
        for _, row in subset.head(500).iterrows():
            figi = row["figi_id"]
            if figi not in price_groups_global:
                continue
            prices = price_groups_global[figi]
            sig_date = row["signal_date_dt"]
            valid_dates = prices.index[prices.index <= sig_date]
            if len(valid_dates) < 65:
                continue

            window_prices = prices.iloc[prices.index.get_indexer(valid_dates[-61:], method="nearest")]
            if len(window_prices) < 61:
                continue

            base = float(window_prices.iloc[0])
            trailing = [float(window_prices.iloc[j]) / base - 1.0 for j in range(61)]
            event_data[tier].append(trailing)

    return event_data, rec_trailing, silence_trailing


# ── Step 5: Reversal detection ────────────────────────────────────────

def reversal_detection(lb_df):
    section("STEP 5: REVERSAL DETECTION (CONTRARIAN PICKS)")

    primary_col = "lookback_alpha_90d"
    valid = lb_df[lb_df[primary_col].notna() & lb_df["spy_relative_return"].notna()].copy()

    q20 = valid[primary_col].quantile(0.20)
    dip_buys = valid[valid[primary_col] <= q20].copy()
    print(f"  Bottom quintile lookback threshold (90d alpha): {q20*100:.2f}%")
    print(f"  'Dip buy' signals: {len(dip_buys)}")

    print(f"\n  Forward 1Y alpha for 'dip buy' signals by tier:")
    for tier in TIER_ORDER:
        tdata = dip_buys[dip_buys["seq_cat"] == tier]["spy_relative_return"]
        if len(tdata) >= 10:
            t, p = stats.ttest_1samp(tdata, 0)
            print(f"    {TIER_LABELS[tier]:>10}: N={len(tdata):>4}, α={tdata.mean()*100:+.2f}%, "
                  f"med={tdata.median()*100:+.2f}%, t={t:.2f}, p={p:.4f}")
        else:
            print(f"    {TIER_LABELS[tier]:>10}: N={len(tdata):>4} (insufficient)")

    # Compare dip-buy performance to full sample
    for tier in TIER_ORDER:
        dip = dip_buys[dip_buys["seq_cat"] == tier]["spy_relative_return"]
        full = valid[valid["seq_cat"] == tier]["spy_relative_return"]
        if len(dip) >= 10 and len(full) >= 20:
            diff = dip.mean() - full.mean()
            print(f"\n    {TIER_LABELS[tier]}: dip-buy α - full α = {diff*100:+.2f}pp")


# ── Step 6: Joint model ──────────────────────────────────────────────

def joint_model(lb_df):
    section("STEP 6: JOINT MODEL (REGRESSION)")

    primary_col = "lookback_alpha_90d"
    valid = lb_df[lb_df[primary_col].notna() & lb_df["spy_relative_return"].notna()].copy()

    valid["is_portfolio"] = (valid["seq_cat"] == "portfolio").astype(int)
    valid["is_hold_rec"] = (valid["seq_cat"] == "hold_rec").astype(int)

    print("  Model: forward_alpha_1Y = β₀ + β₁·is_portfolio + β₂·is_hold_rec + β₃·lookback_90d + ε")
    X = valid[["is_portfolio", "is_hold_rec", primary_col]]
    X = sm.add_constant(X)
    y = valid["spy_relative_return"]

    model = sm.OLS(y, X).fit()
    print(f"\n{model.summary()}")

    # With sector fixed effects
    if "gics_sector" in valid.columns and valid["gics_sector"].notna().sum() > 100:
        print("\n\n  Model with sector fixed effects:")
        sector_dummies = pd.get_dummies(valid["gics_sector"].fillna("Unknown"),
                                        prefix="sector", drop_first=True, dtype=float)
        X_fe = pd.concat([X.reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
        y_fe = y.reset_index(drop=True)
        model_fe = sm.OLS(y_fe, X_fe).fit()

        # Only print key coefficients
        key_vars = ["const", "is_portfolio", "is_hold_rec", primary_col]
        print(f"\n  Key coefficients (sector FEs included but not shown):")
        print(f"  {'Variable':<25} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8}")
        for var in key_vars:
            if var in model_fe.params.index:
                print(f"  {var:<25} {model_fe.params[var]:>10.4f} {model_fe.bse[var]:>10.4f} "
                      f"{model_fe.tvalues[var]:>8.2f} {model_fe.pvalues[var]:>8.4f}")
        print(f"\n  R² (no FE): {model.rsquared:.4f}, R² (with sector FE): {model_fe.rsquared:.4f}")

    return model


# ── Main ──────────────────────────────────────────────────────────────

price_groups_global = {}

def main():
    global price_groups_global

    print("Study B: Momentum, Reversal, and Cramer's Selection Patterns")
    print("=" * 70)

    df = pd.read_csv(os.path.join(DATA, "cramer_sequenced.csv"))
    seq = pd.read_csv(os.path.join(DATA, "sequence_summary.csv"))
    pm = pd.read_parquet(os.path.join(DATA, "daily_price_matrix.parquet"))

    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    df_1y = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    df_1y = df_1y.merge(seq[["sequence_id", "has_cramer_owns", "has_hold_recommendation", "seq_cat"]],
                        on="sequence_id", how="left")

    figi_map = df[["figi_id", "ticker_symbol"]].drop_duplicates()

    # Pre-build price lookup for event study
    pm_dt = pm.copy()
    pm_dt["trade_date"] = pd.to_datetime(pm_dt["trade_date"])
    price_groups_global.update({
        figi: grp.set_index("trade_date")["adj_close"].sort_index()
        for figi, grp in pm_dt.groupby("figi_id")
    })

    lb_df = compute_lookback_returns(df_1y, pm, figi_map)
    prior_return_profiles(lb_df)
    valid, heatmap_data = momentum_control(lb_df)
    event_data, _, _ = recommendation_timing(df, pm, figi_map)
    reversal_detection(lb_df)
    model = joint_model(lb_df)

    # Save summary results
    results = []
    for w in LOOKBACK_WINDOWS:
        col = f"lookback_alpha_{w}d"
        for tier in TIER_ORDER:
            tdata = lb_df[lb_df["seq_cat"] == tier][col].dropna()
            if len(tdata) >= 10:
                t, p = stats.ttest_1samp(tdata, 0)
                results.append({"test": "prior_return_profile", "window": f"{w}d",
                                "tier": tier, "n": len(tdata), "mean": tdata.mean(),
                                "median": tdata.median(), "t_stat": t, "p_value": p})
    pd.DataFrame(results).to_csv(os.path.join(DATA, "momentum_analysis_results.csv"), index=False)
    print(f"  Saved: data/momentum_analysis_results.csv")

    section("SUMMARY")
    print("  Study B complete. Key files:")
    print(f"    data/momentum_signals.csv")
    print(f"    data/momentum_analysis_results.csv")


if __name__ == "__main__":
    main()
