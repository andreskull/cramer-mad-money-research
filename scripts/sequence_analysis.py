"""
Sequence Analysis — All analysis layers from the spec.
Reads cramer_sequenced.csv and sequence_summary.csv.

Updated for reclassified hold signals:
  - has_cramer_owns: Cramer personally holds in Trust/portfolio
  - has_hold_recommendation: advisory "hold" for viewers (NOT personal ownership)
  - Three-way split: portfolio / hold-rec / no-hold
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data"
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)


def load():
    df = pd.read_csv(BASE / "cramer_sequenced.csv")
    seq = pd.read_csv(BASE / "sequence_summary.csv")
    return df, seq


def vix_regime(v):
    if pd.isna(v):
        return None
    if v < 20:
        return "Low"
    elif v < 30:
        return "Moderate"
    return "High"


def ttest_alpha(series):
    """T-test that mean != 0.  Returns (mean, p, n)."""
    s = series.dropna()
    if len(s) < 5:
        return (np.nan, np.nan, len(s))
    t, p = stats.ttest_1samp(s, 0)
    return (s.mean(), p, len(s))


def section_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def seq_category(row):
    """Three-way sequence classification."""
    if row.get("has_cramer_owns", False):
        return "portfolio"
    elif row.get("has_hold_recommendation", False):
        return "hold_rec"
    else:
        return "no_hold"


# ── Layer 1: The Portfolio Signal ────────────────────────────────────────

def analyze_portfolio_signal(df, seq):
    section_header("LAYER 1: THE PORTFOLIO SIGNAL (3-WAY)")

    # Use 1Y horizon for main results
    d = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    d["vix_regime"] = d["vix_at_signal"].apply(vix_regime)

    # Merge sequence info
    seq["seq_cat"] = seq.apply(seq_category, axis=1)
    d = d.merge(seq[["sequence_id", "has_cramer_owns", "has_hold_recommendation",
                      "total_signals", "seq_cat"]], on="sequence_id", how="left")

    print("\n── 1.1 Three-way entry signal comparison ──")
    entries = d[d["is_sequence_start"]]
    for cat, label in [("portfolio", "Portfolio entries (cramer_owns)"),
                        ("hold_rec", "Hold-rec entries"),
                        ("no_hold", "No-hold entries")]:
        sub = entries[entries["seq_cat"] == cat]
        m, p, n = ttest_alpha(sub["spy_relative_return"])
        print(f"  {label:>38}: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}")

    # Pairwise differences
    print("\n── 1.1b Pairwise differences (entry signals) ──")
    for a_cat, b_cat in [("portfolio", "no_hold"), ("portfolio", "hold_rec"),
                          ("hold_rec", "no_hold")]:
        a = entries[entries["seq_cat"] == a_cat]["spy_relative_return"].dropna()
        b = entries[entries["seq_cat"] == b_cat]["spy_relative_return"].dropna()
        if len(a) >= 5 and len(b) >= 5:
            t, p = stats.ttest_ind(a, b)
            diff = a.mean() - b.mean()
            print(f"  {a_cat:>10} vs {b_cat:<10}: diff={diff*100:+.2f}pp, t={t:.3f}, p={p:.4f}")

    print("\n── 1.2 First-hold signal forward performance (cramer_owns only) ──")
    first_holds = d[d["is_first_hold"]]
    m, p, n = ttest_alpha(first_holds["spy_relative_return"])
    print(f"  First holds (1Y): N={n}, alpha={m*100:+.2f}%, p={p:.4f}")

    print("\n── 1.3 All hold-state vs non-hold-state kept signals ──")
    for label, mask in [("In hold state (cramer_owns)", d["in_hold_state"] == True),
                        ("Not in hold state", d["in_hold_state"] != True)]:
        m, p, n = ttest_alpha(d[mask]["spy_relative_return"])
        print(f"  {label:>38}: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}")

    print("\n── 1.4 Within-sector comparison (entry signals, 3-way) ──")
    for sector in ["Technology", "Consumer Cyclical", "Healthcare",
                    "Industrials", "Financial Services"]:
        for cat in ["portfolio", "hold_rec", "no_hold"]:
            sub = entries[(entries["gics_sector"] == sector) & (entries["seq_cat"] == cat)]
            m, p, n = ttest_alpha(sub["spy_relative_return"])
            label = f"{sector[:12]:>12} {cat[:8]:>8}"
            if n >= 15:
                print(f"  {label}: N={n:>4}, alpha={m*100:+.2f}%, p={p:.4f}")

    print("\n── 1.5 By horizon (3-way) ──")
    for hz in ["1w", "1m", "3m", "6m", "1y"]:
        dhz = df[(df["requested_horizon"] == hz) & (df["kept"] == True)].copy()
        dhz = dhz.merge(seq[["sequence_id", "seq_cat"]], on="sequence_id", how="left")
        entries_hz = dhz[dhz["is_sequence_start"]]
        for cat in ["portfolio", "hold_rec", "no_hold"]:
            sub = entries_hz[entries_hz["seq_cat"] == cat]
            m, p, n = ttest_alpha(sub["spy_relative_return"])
            label = f"{hz:>3}/{cat[:8]:>8}"
            if n >= 20:
                print(f"  {label}: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}")


# ── Layer 2: Mention Frequency ──────────────────────────────────────────

def analyze_mention_frequency(df, seq):
    section_header("LAYER 2: MENTION FREQUENCY")

    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    # Use ALL signals (kept + discarded) for frequency counting
    d1y = df[df["requested_horizon"] == "1y"].copy()
    d1y = d1y.merge(seq[["sequence_id", "seq_cat", "total_signals"]], on="sequence_id", how="left")

    print("\n── 2.1 Mention frequency by sequence category ──")
    ticker_stats = d1y.groupby("ticker_symbol").agg(
        total_mentions=("signal_type", "count"),
    )
    # Get the "best" category per ticker (portfolio > hold_rec > no_hold)
    ticker_cat = d1y.groupby("ticker_symbol")["seq_cat"].apply(
        lambda x: "portfolio" if "portfolio" in x.values else
                  "hold_rec" if "hold_rec" in x.values else "no_hold"
    )
    ticker_stats["best_cat"] = ticker_cat
    for cat in ["portfolio", "hold_rec", "no_hold"]:
        sub = ticker_stats[ticker_stats["best_cat"] == cat]
        print(f"  {cat:>12}: N={len(sub):>5} tickers, median mentions={sub['total_mentions'].median():.0f}, "
              f"mean={sub['total_mentions'].mean():.1f}")

    print("\n── 2.2 Within-sector mention frequency ──")
    first_sector = d1y.groupby("ticker_symbol")["gics_sector"].first()
    ticker_stats = ticker_stats.join(first_sector)
    for sector in ["Technology", "Consumer Cyclical", "Healthcare", "Industrials"]:
        for cat in ["portfolio", "hold_rec", "no_hold"]:
            sub = ticker_stats[(ticker_stats["gics_sector"] == sector) &
                               (ticker_stats["best_cat"] == cat)]
            if len(sub) >= 5:
                print(f"  {sector[:12]:>12} {cat[:8]:>8}: N={len(sub):>4}, "
                      f"median={sub['total_mentions'].median():.0f}, mean={sub['total_mentions'].mean():.1f}")

    print("\n── 2.3 Hold-state signal share (all signals, incl. discarded) ──")
    total = len(d1y)
    in_hold = d1y["in_hold_state"].fillna(False).sum()
    print(f"  Total 1Y signals: {total}")
    print(f"  In hold state (cramer_owns): {int(in_hold)} ({in_hold/total*100:.1f}%)")
    print(f"  Not in hold state: {total - int(in_hold)} ({(total-in_hold)/total*100:.1f}%)")


# ── Layer 3: Conditional Structure (B-H) ────────────────────────────────

def analyze_conditional(df, seq):
    section_header("LAYER 3: CONDITIONAL STRUCTURE (B-H)")

    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    results = []
    for hz in ["1w", "1m", "3m", "6m", "1y"]:
        dhz = df[(df["requested_horizon"] == hz) & (df["kept"] == True)].copy()
        dhz["vix_regime"] = dhz["vix_at_signal"].apply(vix_regime)
        for vix in ["Low", "Moderate", "High"]:
            for seg in ["Monologue/Top of Show", "Lightning Round", "Mid-Show/Interview"]:
                sub = dhz[(dhz["vix_regime"] == vix) &
                          (dhz["show_segment_heuristic"] == seg)]
                a = sub["spy_relative_return"].dropna()
                if len(a) >= 20:
                    t, p = stats.ttest_1samp(a, 0)
                    results.append({
                        "horizon": hz, "vix": vix,
                        "segment": seg.split("/")[0][:10],
                        "N": len(a), "alpha": a.mean() * 100, "p_raw": p
                    })

    rdf = pd.DataFrame(results).sort_values("p_raw")
    m = len(rdf)
    rdf["rank"] = range(1, m + 1)
    rdf["bh_threshold"] = rdf["rank"] / m * 0.05
    rdf["bh_reject"] = rdf["p_raw"] <= rdf["bh_threshold"]

    survivors = rdf[rdf["bh_reject"]]
    print(f"\n── 3.1 B-H survivors (all kept): {len(survivors)} of {m} ──")
    for _, r in survivors.iterrows():
        print(f"  {r['horizon']:>3}/{r['vix']:>8}/{r['segment']:>10}: "
              f"N={int(r['N']):>5}, alpha={r['alpha']:+.2f}%, p={r['p_raw']:.4f}")

    # B-H on entry signals only
    print(f"\n── 3.2 B-H on entry signals only ──")
    results_e = []
    for hz in ["1w", "1m", "3m", "6m", "1y"]:
        dhz = df[(df["requested_horizon"] == hz) & (df["kept"] == True) &
                 (df["is_sequence_start"] == True)].copy()
        dhz["vix_regime"] = dhz["vix_at_signal"].apply(vix_regime)
        for vix in ["Low", "Moderate", "High"]:
            for seg in ["Monologue/Top of Show", "Lightning Round", "Mid-Show/Interview"]:
                sub = dhz[(dhz["vix_regime"] == vix) &
                          (dhz["show_segment_heuristic"] == seg)]
                a = sub["spy_relative_return"].dropna()
                if len(a) >= 20:
                    t, p = stats.ttest_1samp(a, 0)
                    results_e.append({
                        "horizon": hz, "vix": vix,
                        "segment": seg.split("/")[0][:10],
                        "N": len(a), "alpha": a.mean() * 100, "p_raw": p
                    })

    rdf_e = pd.DataFrame(results_e).sort_values("p_raw")
    m_e = len(rdf_e)
    rdf_e["rank"] = range(1, m_e + 1)
    rdf_e["bh_threshold"] = rdf_e["rank"] / m_e * 0.05
    rdf_e["bh_reject"] = rdf_e["p_raw"] <= rdf_e["bh_threshold"]

    survivors_e = rdf_e[rdf_e["bh_reject"]]
    print(f"  Survivors: {len(survivors_e)} of {m_e}")
    for _, r in survivors_e.iterrows():
        print(f"  {r['horizon']:>3}/{r['vix']:>8}/{r['segment']:>10}: "
              f"N={int(r['N']):>5}, alpha={r['alpha']:+.2f}%, p={r['p_raw']:.4f}")


# ── Layer 4: Sequence Structure ─────────────────────────────────────────

def analyze_sequence_structure(df, seq):
    section_header("LAYER 4: SEQUENCE STRUCTURE")

    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    d = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    d = d.merge(seq[["sequence_id", "has_cramer_owns", "total_signals", "seq_cat"]],
                on="sequence_id", how="left")

    print("\n── 4.1 Sequence length distribution (3-way) ──")
    for cat in ["portfolio", "hold_rec", "no_hold"]:
        sub = seq[seq["seq_cat"] == cat]
        print(f"  {cat:>12}: N={len(sub):>5}, signals median={sub['total_signals'].median():.0f}, "
              f"mean={sub['total_signals'].mean():.1f}, max={sub['total_signals'].max()}")

    print("\n── 4.2 Performance by position (portfolio sequences only) ──")
    port_seqs = seq[seq["has_cramer_owns"]]["sequence_id"].values
    port_kept = d[d["sequence_id"].isin(port_seqs)].copy()

    port_kept = port_kept.sort_values(["sequence_id", "signal_date"])
    port_kept["kept_position"] = port_kept.groupby("sequence_id").cumcount() + 1

    for pos in [1, 2, 3, 4, 5]:
        sub = port_kept[port_kept["kept_position"] == pos]
        m, p, n = ttest_alpha(sub["spy_relative_return"])
        if n >= 10:
            print(f"  Position {pos}: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}")

    sub = port_kept[port_kept["kept_position"] >= 6]
    m, p, n = ttest_alpha(sub["spy_relative_return"])
    if n >= 5:
        print(f"  Position 6+: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}")

    print("\n── 4.3 Entry alpha by sequence total_signals (3-way) ──")
    entries = d[d["is_sequence_start"]]
    for lo, hi, label in [(1, 1, "1 (singleton)"), (2, 3, "2-3"), (4, 7, "4-7"),
                           (8, 15, "8-15"), (16, 50, "16-50"), (51, 9999, "51+")]:
        sub = entries[(entries["total_signals"] >= lo) & (entries["total_signals"] <= hi)]
        m, p, n = ttest_alpha(sub["spy_relative_return"])
        # Also show portfolio fraction
        port_frac = sub["has_cramer_owns"].fillna(False).mean() * 100 if len(sub) > 0 else 0
        if n >= 10:
            print(f"  {label:>15}: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}, portfolio={port_frac:.0f}%")


# ── Layer 5: Inverse Cramer ─────────────────────────────────────────────

def analyze_inverse(df, seq):
    section_header("LAYER 5: INVERSE CRAMER")

    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    d = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    d["vix_regime"] = d["vix_at_signal"].apply(vix_regime)
    d = d.merge(seq[["sequence_id", "has_cramer_owns", "seq_cat"]], on="sequence_id", how="left")

    print("\n── 5.1 Blanket inverse (all kept signals) ──")
    a = d["spy_relative_return"].dropna()
    inv = -a
    m, p, n = ttest_alpha(inv)
    print(f"  Short all: N={n}, alpha={m*100:+.2f}%, p={p:.4f}")

    print("\n── 5.2 Targeted inverse: non-portfolio entries in Low VIX ──")
    # Non-portfolio = both hold_rec and no_hold
    target = d[(d["is_sequence_start"]) & (d["seq_cat"] != "portfolio") & (d["vix_regime"] == "Low")]
    inv = -target["spy_relative_return"].dropna()
    m, p, n = ttest_alpha(inv)
    print(f"  Short non-portfolio/Low: N={n}, alpha={m*100:+.2f}%, p={p:.4f}")

    # Narrower: no_hold only in Low VIX
    target2 = d[(d["is_sequence_start"]) & (d["seq_cat"] == "no_hold") & (d["vix_regime"] == "Low")]
    inv2 = -target2["spy_relative_return"].dropna()
    m, p, n = ttest_alpha(inv2)
    print(f"  Short no-hold/Low:       N={n}, alpha={m*100:+.2f}%, p={p:.4f}")

    print("\n── 5.3 Targeted inverse by segment (non-portfolio, Low VIX, entries) ──")
    for seg in ["Monologue/Top of Show", "Lightning Round", "Mid-Show/Interview"]:
        sub = d[(d["is_sequence_start"]) & (d["seq_cat"] != "portfolio") &
                (d["vix_regime"] == "Low") & (d["show_segment_heuristic"] == seg)]
        inv = -sub["spy_relative_return"].dropna()
        m, p, n = ttest_alpha(inv)
        if n >= 20:
            s = seg.split("/")[0]
            print(f"  {s:>15}: N={n:>4}, alpha={m*100:+.2f}%, p={p:.4f}")


# ── Layer 6: 2020 Robustness ────────────────────────────────────────────

def analyze_robustness(df, seq):
    section_header("LAYER 6: 2020 ROBUSTNESS")

    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    d = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    d = d.merge(seq[["sequence_id", "has_cramer_owns", "seq_cat"]], on="sequence_id", how="left")
    d["year"] = pd.to_datetime(d["signal_date"]).dt.year

    print("\n── 6.1 Key results excluding 2020 (3-way entries) ──")
    d_no20 = d[d["year"] != 2020]
    entries = d_no20[d_no20["is_sequence_start"]]
    for cat, label in [("portfolio", "Portfolio entries"),
                        ("hold_rec", "Hold-rec entries"),
                        ("no_hold", "No-hold entries"),
                        ]:
        sub = entries[entries["seq_cat"] == cat]
        m, p, n = ttest_alpha(sub["spy_relative_return"])
        print(f"  {label:>25}: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}")

    # Hold-state vs not, excl 2020
    for label, mask in [
        ("In hold state (owns)", d_no20["in_hold_state"] == True),
        ("Not in hold state", d_no20["in_hold_state"] != True),
    ]:
        sub = d_no20[mask]
        m, p, n = ttest_alpha(sub["spy_relative_return"])
        print(f"  {label:>25}: N={n:>5}, alpha={m*100:+.2f}%, p={p:.4f}")

    print("\n── 6.2 Three-way by year (entries) ──")
    all_entries = d[d["is_sequence_start"]]
    for year in sorted(all_entries["year"].unique()):
        for cat in ["portfolio", "hold_rec", "no_hold"]:
            sub = all_entries[(all_entries["year"] == year) & (all_entries["seq_cat"] == cat)]
            m, p, n = ttest_alpha(sub["spy_relative_return"])
            if n >= 10:
                label = f"{year}/{cat[:8]:>8}"
                print(f"  {label}: N={n:>4}, alpha={m*100:+.2f}%, p={p:.4f}")


def main():
    print("Loading sequenced data...")
    df, seq = load()
    print(f"  Signals: {len(df)}, Sequences: {len(seq)}")
    print(f"  Portfolio sequences: {(seq['has_cramer_owns'] == True).sum()}")
    print(f"  Hold-rec sequences:  {((seq['has_cramer_owns'] != True) & (seq['has_hold_recommendation'] == True)).sum()}")
    print(f"  No-hold sequences:   {((seq['has_cramer_owns'] != True) & (seq['has_hold_recommendation'] != True)).sum()}")

    analyze_portfolio_signal(df, seq)
    analyze_mention_frequency(df, seq)
    analyze_conditional(df, seq)
    analyze_sequence_structure(df, seq)
    analyze_inverse(df, seq)
    analyze_robustness(df, seq)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
