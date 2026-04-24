"""
Sensitivity analysis: sweep hold_window, non_hold_window, prune parameters.
Verify key findings are stable across parameter choices.

Updated for reclassified hold signals:
  - Only cramer_owns triggers hold state (not hold_recommendation)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data"


def load_raw():
    df = pd.read_csv(BASE / "cramer_snapshot_full.csv")
    df = df[df["signal_type"] != "start_short"].copy()
    horizon_rank = {"1y": 0, "6m": 1, "3m": 2, "1m": 3, "1w": 4}
    df["_hz_rank"] = df["requested_horizon"].map(horizon_rank)
    unique = (
        df.sort_values(["ticker_symbol", "signal_date", "_hz_rank"])
        .drop_duplicates(subset=["signal_date", "ticker_symbol", "signal_type"], keep="first")
        .sort_values(["ticker_symbol", "signal_date"])
        .copy()
    )
    df.drop(columns=["_hz_rank"], inplace=True)
    unique.drop(columns=["_hz_rank"], inplace=True)
    return df, unique


def build_and_prune(unique, hold_window, non_hold_window, prune_gap, prune_price):
    """Run full sequence model with given parameters. Returns annotated unique signals."""
    seq_id = 0
    ann = []

    for ticker, grp in unique.groupby("ticker_symbol"):
        grp = grp.sort_values("signal_date")
        has_seen_hold = False
        hold_state = False
        prev_date = None
        pos = 0

        for idx in grp.index:
            row = unique.loc[idx]
            cur_date = pd.to_datetime(row["signal_date"])
            start_new = False

            if prev_date is None:
                start_new = True
            else:
                active_window = hold_window if has_seen_hold else non_hold_window
                gap = (cur_date - prev_date).days
                if gap > active_window:
                    start_new = True

            if start_new:
                seq_id += 1
                hold_state = False
                has_seen_hold = False
                pos = 0

            pos += 1
            is_first_hold = False
            if row.get("hold_subtype") == "cramer_owns" and not has_seen_hold:
                is_first_hold = True
                hold_state = True
                has_seen_hold = True

            ann.append({
                "idx": idx,
                "sequence_id": seq_id,
                "is_sequence_start": (pos == 1),
                "is_first_hold": is_first_hold,
                "in_hold_state": hold_state,
            })
            prev_date = cur_date

    adf = pd.DataFrame(ann).set_index("idx")

    # Pruning
    adf["kept"] = False
    for seq, grp_idx in adf.groupby("sequence_id"):
        ref_date = None
        ref_price = None
        for idx in grp_idx.index:
            row = unique.loc[idx]
            cur_date = pd.to_datetime(row["signal_date"])
            cur_price = row["start_price"]
            vp = pd.notna(cur_price) and cur_price > 0

            if adf.loc[idx, "is_sequence_start"] or adf.loc[idx, "is_first_hold"]:
                adf.loc[idx, "kept"] = True
                ref_date = cur_date
                ref_price = cur_price if vp else ref_price
            else:
                if ref_price and ref_price > 0 and vp:
                    dg = (cur_date - ref_date).days
                    pd_diff = abs(cur_price - ref_price) / ref_price
                    if dg <= prune_gap and pd_diff <= prune_price:
                        adf.loc[idx, "kept"] = False
                    else:
                        adf.loc[idx, "kept"] = True
                        ref_date = cur_date
                        ref_price = cur_price
                else:
                    adf.loc[idx, "kept"] = True
                    ref_date = cur_date
                    ref_price = cur_price if vp else ref_price

    # Build sequence summary
    merged = unique.join(adf)
    seq_has_hold = merged.groupby("sequence_id")["is_first_hold"].any()
    seq_total = merged.groupby("sequence_id").size()

    return adf, seq_has_hold, seq_total


def evaluate(unique, adf, seq_has_hold, seq_total):
    """Compute key metrics for one parameter combination."""
    merged = unique.join(adf)
    # Filter to 1Y horizon signals only (for performance)
    m1y = merged[merged["requested_horizon"] == "1y"].copy()
    kept = m1y[m1y["kept"] == True].copy()
    kept["has_cramer_owns"] = kept["sequence_id"].map(seq_has_hold)
    kept["total_signals"] = kept["sequence_id"].map(seq_total)

    n_seq = adf["sequence_id"].nunique()
    n_kept = int(adf["kept"].sum())

    # Entry signals
    entries = kept[kept["is_sequence_start"]]
    hold_e = entries[entries["has_cramer_owns"] == True]["spy_relative_return"].dropna()
    nohold_e = entries[entries["has_cramer_owns"] != True]["spy_relative_return"].dropna()

    hold_alpha = hold_e.mean() if len(hold_e) > 0 else np.nan
    nohold_alpha = nohold_e.mean() if len(nohold_e) > 0 else np.nan
    _, diff_p = stats.ttest_ind(hold_e, nohold_e) if len(hold_e) > 5 and len(nohold_e) > 5 else (np.nan, np.nan)
    _, nohold_p = stats.ttest_1samp(nohold_e, 0) if len(nohold_e) > 5 else (np.nan, np.nan)

    # Hold state
    hs = kept[kept["in_hold_state"] == True]["spy_relative_return"].dropna()
    nhs = kept[kept["in_hold_state"] != True]["spy_relative_return"].dropna()
    hs_alpha = hs.mean() if len(hs) > 0 else np.nan
    _, hs_p = stats.ttest_1samp(hs, 0) if len(hs) > 5 else (np.nan, np.nan)

    # Conviction: 16+ vs singleton
    sing = entries[entries["total_signals"] == 1]["spy_relative_return"].dropna()
    big = entries[entries["total_signals"] >= 16]["spy_relative_return"].dropna()
    sing_alpha = sing.mean() if len(sing) > 0 else np.nan
    big_alpha = big.mean() if len(big) > 0 else np.nan

    return {
        "n_seq": n_seq,
        "n_kept": n_kept,
        "n_hold_e": len(hold_e),
        "n_nohold_e": len(nohold_e),
        "hold_entry_alpha": hold_alpha * 100 if not np.isnan(hold_alpha) else np.nan,
        "nohold_entry_alpha": nohold_alpha * 100 if not np.isnan(nohold_alpha) else np.nan,
        "diff_p": diff_p,
        "nohold_p": nohold_p,
        "holdstate_alpha": hs_alpha * 100 if not np.isnan(hs_alpha) else np.nan,
        "holdstate_p": hs_p,
        "singleton_alpha": sing_alpha * 100 if not np.isnan(sing_alpha) else np.nan,
        "big16_alpha": big_alpha * 100 if not np.isnan(big_alpha) else np.nan,
    }


def main():
    print("Loading data...")
    df, unique = load_raw()

    results = []

    # Sweep hold_window
    print("\n── Sweep: hold_window (non_hold=60, prune=60d/10%) ──")
    for hw in [90, 120, 150, 180]:
        adf, shh, st = build_and_prune(unique, hw, 60, 60, 0.10)
        r = evaluate(unique, adf, shh, st)
        r["hold_window"] = hw
        r["non_hold_window"] = 60
        r["prune_gap"] = 60
        r["prune_price"] = 0.10
        results.append(r)
        print(f"  hw={hw:>3}: seq={r['n_seq']:>5}, hold_e={r['hold_entry_alpha']:+.2f}%, "
              f"nohold_e={r['nohold_entry_alpha']:+.2f}% (p={r['nohold_p']:.4f}), "
              f"diff_p={r['diff_p']:.4f}, hs={r['holdstate_alpha']:+.2f}% (p={r['holdstate_p']:.4f})")

    # Sweep non_hold_window
    print("\n── Sweep: non_hold_window (hold=120, prune=60d/10%) ──")
    for nhw in [45, 60, 90, 120]:
        adf, shh, st = build_and_prune(unique, 120, nhw, 60, 0.10)
        r = evaluate(unique, adf, shh, st)
        r["hold_window"] = 120
        r["non_hold_window"] = nhw
        r["prune_gap"] = 60
        r["prune_price"] = 0.10
        results.append(r)
        print(f"  nhw={nhw:>3}: seq={r['n_seq']:>5}, nohold_e={r['nohold_entry_alpha']:+.2f}% "
              f"(p={r['nohold_p']:.4f}), singleton={r['singleton_alpha']:+.2f}%, "
              f"big16={r['big16_alpha']:+.2f}%")

    # Sweep prune parameters
    print("\n── Sweep: prune_gap × prune_price (hold=120, non_hold=60) ──")
    for pg in [30, 60, 90]:
        for pp in [0.10, 0.15, 0.20]:
            adf, shh, st = build_and_prune(unique, 120, 60, pg, pp)
            r = evaluate(unique, adf, shh, st)
            r["hold_window"] = 120
            r["non_hold_window"] = 60
            r["prune_gap"] = pg
            r["prune_price"] = pp
            results.append(r)
            print(f"  {pg}d/{pp*100:.0f}%: kept={r['n_kept']:>5}, "
                  f"nohold_e={r['nohold_entry_alpha']:+.2f}% (p={r['nohold_p']:.4f}), "
                  f"diff_p={r['diff_p']:.4f}, hs={r['holdstate_alpha']:+.2f}% (p={r['holdstate_p']:.4f})")

    # Save
    rdf = pd.DataFrame(results)
    rdf.to_csv(BASE / "sensitivity_results.csv", index=False)
    print(f"\nSaved sensitivity_results.csv ({len(rdf)} rows)")

    # Summary
    print("\n── STABILITY CHECK ──")
    print(f"Non-hold entry alpha negative in {sum(1 for r in results if r['nohold_entry_alpha'] < 0)}/{len(results)} combinations")
    print(f"Non-hold entry p<0.05 in {sum(1 for r in results if r['nohold_p'] < 0.05)}/{len(results)} combinations")
    print(f"Hold/non-hold diff p<0.05 in {sum(1 for r in results if r['diff_p'] < 0.05)}/{len(results)} combinations")
    print(f"Hold-state alpha positive in {sum(1 for r in results if r['holdstate_alpha'] > 0)}/{len(results)} combinations")


if __name__ == "__main__":
    main()
