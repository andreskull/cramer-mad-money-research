"""
Signal Sequence Model — Implementation
See SPEC_sequence_model.md for full specification.

Reads cramer_snapshot_full.csv, builds sequences, reclassifies hold state,
prunes duplicates, and writes:
  - cramer_sequenced.csv   (signal-level, all horizons, with annotations)
  - sequence_summary.csv   (one row per sequence)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data"

# ── Parameters ──────────────────────────────────────────────────────────
HOLD_WINDOW = 120        # days — max gap for hold-confirmed sequences
NON_HOLD_WINDOW = 60     # days — max gap for sequences without hold
PRUNE_GAP = 60           # days — min gap between kept signals
PRUNE_PRICE = 0.10       # 10% — min price divergence between kept signals


def load_and_prepare():
    """Load data, drop short signals, deduplicate to unique signals."""
    df = pd.read_csv(BASE / "cramer_snapshot_full.csv")

    # Remove the 2 start_short signals (PEP 2024-01-02) — long-only study
    df = df[df["signal_type"] != "start_short"].copy()

    # Canonical unique signals: one row per (signal_date, ticker, signal_type)
    # Deduplicate across ALL horizons — some signals exist at short horizons
    # but not 1Y (position boundary filtering removes long horizons early).
    # Keep the longest available horizon row for price/return data.
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


def build_sequences(unique):
    """
    Step 1+2: Detect sequences and assign hold state.
    Returns unique signals DataFrame with sequence annotations.
    """
    seq_id = 0
    annotations = []

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
            end_reason = None

            if prev_date is None:
                start_new = True
            else:
                active_window = HOLD_WINDOW if has_seen_hold else NON_HOLD_WINDOW
                gap = (cur_date - prev_date).days

                if gap > active_window:
                    end_reason = "gap_expired"
                    start_new = True
                # No reversal check needed — short signals already removed

            # Mark end reason on previous sequence's last signal
            if end_reason and len(annotations) > 0:
                annotations[-1]["end_reason"] = end_reason

            if start_new:
                seq_id += 1
                hold_state = False
                has_seen_hold = False
                pos = 0

            pos += 1

            is_first_hold = False
            # ONLY cramer_owns triggers the hold state transition
            if row["signal_type"] == "hold_long" and row["hold_subtype"] == "cramer_owns" and not has_seen_hold:
                is_first_hold = True
                hold_state = True
                has_seen_hold = True

            annotations.append({
                "orig_index": idx,
                "sequence_id": seq_id,
                "position_in_sequence": pos,
                "is_sequence_start": (pos == 1),
                "is_first_hold": is_first_hold,
                "in_hold_state": hold_state,
                "end_reason": None,
            })

            prev_date = cur_date

    ann = pd.DataFrame(annotations).set_index("orig_index")
    return ann


def prune_signals(unique, ann):
    """
    Step 3: Prune duplicates within sequences.
    - Sequence starts: always kept
    - First holds: always kept
    - Others: kept if >PRUNE_GAP days OR >PRUNE_PRICE price diff from last kept
    """
    ann["kept"] = False

    for seq, grp_idx in ann.groupby("sequence_id"):
        ref_date = None
        ref_price = None

        for idx in grp_idx.index:
            row = unique.loc[idx]
            cur_date = pd.to_datetime(row["signal_date"])
            cur_price = row["start_price"]
            valid_price = pd.notna(cur_price) and cur_price > 0

            if ann.loc[idx, "is_sequence_start"]:
                ann.loc[idx, "kept"] = True
                ref_date = cur_date
                ref_price = cur_price if valid_price else ref_price

            elif ann.loc[idx, "is_first_hold"]:
                ann.loc[idx, "kept"] = True
                ref_date = cur_date
                ref_price = cur_price if valid_price else ref_price

            else:
                if ref_price and ref_price > 0 and valid_price:
                    days_gap = (cur_date - ref_date).days
                    price_diff = abs(cur_price - ref_price) / ref_price
                    if days_gap <= PRUNE_GAP and price_diff <= PRUNE_PRICE:
                        ann.loc[idx, "kept"] = False
                    else:
                        ann.loc[idx, "kept"] = True
                        ref_date = cur_date
                        ref_price = cur_price
                else:
                    ann.loc[idx, "kept"] = True
                    ref_date = cur_date
                    ref_price = cur_price if valid_price else ref_price

    return ann


def build_sequence_summary(unique, ann):
    """Build one row per sequence with summary statistics."""
    merged = unique.join(ann.drop(columns=["hold_subtype"], errors="ignore"))

    summaries = []
    for seq_id, grp in merged.groupby("sequence_id"):
        first = grp.iloc[0]
        last = grp.iloc[-1]
        first_hold_rows = grp[grp["is_first_hold"]]

        summaries.append({
            "sequence_id": seq_id,
            "ticker": first["ticker_symbol"],
            "start_date": first["signal_date"],
            "end_date": last["signal_date"],
            "total_signals": len(grp),
            "kept_signals": int(grp["kept"].sum()),
            "has_cramer_owns": (grp["hold_subtype"] == "cramer_owns").any(),
            "has_hold_recommendation": (grp["hold_subtype"] == "hold_recommendation").any(),
            "first_hold_date": (
                first_hold_rows.iloc[0]["signal_date"]
                if len(first_hold_rows) > 0 else None
            ),
            "end_reason": last["end_reason"] if pd.notna(last.get("end_reason")) else "gap_expired",
            "entry_alpha": first["spy_relative_return"],
            "entry_price": first["start_price"],
            "gics_sector": first["gics_sector"],
            "vix_at_entry": first["vix_at_signal"],
        })

    return pd.DataFrame(summaries)


def annotate_all_horizons(df, unique, ann):
    """
    Join sequence annotations back to all horizon rows.
    Match on (signal_date, ticker_symbol, signal_type).
    """
    # Build lookup from unique signals
    unique_ann = unique[["signal_date", "ticker_symbol", "signal_type"]].join(ann)

    # Merge into full dataset
    merge_keys = ["signal_date", "ticker_symbol", "signal_type"]
    ann_cols = ["sequence_id", "position_in_sequence", "is_sequence_start",
                "is_first_hold", "in_hold_state", "kept", "end_reason"]

    lookup = unique_ann[merge_keys + ann_cols].drop_duplicates(subset=merge_keys)

    result = df.merge(lookup, on=merge_keys, how="left")

    # Signals not matched (shouldn't happen, but safety)
    unmatched = result["sequence_id"].isna().sum()
    if unmatched > 0:
        print(f"  WARNING: {unmatched} rows unmatched ({unmatched/len(result)*100:.1f}%)")

    return result


def main():
    print("Loading data...")
    df, unique = load_and_prepare()
    print(f"  Full dataset: {len(df)} rows ({df['signal_type'].value_counts().to_dict()})")
    print(f"  Unique signals (1Y): {len(unique)}")
    print(f"  Unique tickers: {unique['ticker_symbol'].nunique()}")

    print("\nBuilding sequences...")
    ann = build_sequences(unique)
    n_seq = ann["sequence_id"].nunique()
    n_owns_seq = ann.groupby("sequence_id")["is_first_hold"].any().sum()
    print(f"  Sequences: {n_seq}")
    print(f"  Portfolio sequences (cramer_owns): {n_owns_seq} ({n_owns_seq/n_seq*100:.1f}%)")
    print(f"  Other sequences: {n_seq - n_owns_seq} ({(n_seq-n_owns_seq)/n_seq*100:.1f}%)")

    print("\nPruning duplicates...")
    ann = prune_signals(unique, ann)
    n_kept = ann["kept"].sum()
    print(f"  Kept: {n_kept} ({n_kept/len(ann)*100:.1f}%)")
    print(f"  Pruned: {len(ann)-n_kept} ({(len(ann)-n_kept)/len(ann)*100:.1f}%)")

    print("\nBuilding sequence summary...")
    seq_summary = build_sequence_summary(unique, ann)

    print("\nAnnotating all horizons...")
    result = annotate_all_horizons(df, unique, ann)

    # Save
    out_seq = BASE / "cramer_sequenced.csv"
    out_summary = BASE / "sequence_summary.csv"
    result.to_csv(out_seq, index=False)
    seq_summary.to_csv(out_summary, index=False)
    print(f"\nSaved: {out_seq} ({len(result)} rows)")
    print(f"Saved: {out_summary} ({len(seq_summary)} rows)")

    # Quick validation
    print("\n── Validation ──")
    unmatched = result["sequence_id"].isna().sum()
    print(f"Unmatched rows: {unmatched} ({unmatched/len(result)*100:.1f}%)")
    kept = result[(result["requested_horizon"] == "1y") & (result["kept"] == True)]
    print(f"Kept 1Y signals: {len(kept)}")
    print(f"  Sequence starts: {int(kept['is_sequence_start'].sum())}")
    print(f"  First holds: {int(kept['is_first_hold'].sum())}")
    print(f"  In hold state: {int(kept['in_hold_state'].fillna(False).sum())}")
    print(f"  Not in hold state: {int((~kept['in_hold_state'].fillna(True)).sum())}")

    # Trace NVDA
    nvda = result[(result["ticker_symbol"] == "NVDA") &
                  (result["requested_horizon"] == "1y")].sort_values("signal_date").head(15)
    print("\nNVDA trace (first 15, 1Y):")
    for _, r in nvda.iterrows():
        k = "KEPT" if r["kept"] else "    "
        h = "HOLD" if r["in_hold_state"] else "    "
        fh = "1stH" if r["is_first_hold"] else "    "
        ss = "STRT" if r["is_sequence_start"] else "    "
        ht = str(r["hold_subtype"])[:4] if pd.notna(r["hold_subtype"]) else "    "
        print(f"  seq={int(r['sequence_id']):>4} p={int(r['position_in_sequence']):>2}"
              f"  {r['signal_date']}  {r['signal_type']:>10} ({ht})"
              f"  ${r['start_price']:>7.2f}  {k} {h} {ss} {fh}")


if __name__ == "__main__":
    main()
