"""
Study A: Market Capitalization and the Conviction Gradient
==========================================================
Tests whether the three-tier conviction gradient (portfolio > hold-rec > no-hold)
proxies for stock size rather than information quality.

Reads: data/cramer_sequenced.csv, data/sequence_summary.csv
Writes: data/market_caps.csv, data/size_analysis_results.csv,
        figures/fig_size_composition.png, figures/fig_size_gradient.png
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIGS = os.path.join(BASE, "figures")
os.makedirs(FIGS, exist_ok=True)

SIZE_BUCKETS = [
    ("Mega-cap",  200e9,  np.inf),
    ("Large-cap", 10e9,   200e9),
    ("Mid-cap",   2e9,    10e9),
    ("Small-cap", 0,      2e9),
]
BUCKET_ORDER = [b[0] for b in SIZE_BUCKETS]
TIER_ORDER = ["portfolio", "hold_rec", "no_hold"]
TIER_LABELS = {"portfolio": "Portfolio pick", "hold_rec": "Hold-rec", "no_hold": "Casual buy"}
TIER_COLORS = {"portfolio": "#2166ac", "hold_rec": "#f4a582", "no_hold": "#b2182b"}


def assign_bucket(cap):
    if pd.isna(cap):
        return None
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= cap < hi:
            return name
    return None


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


# ── Step 1: Market cap enrichment ──────────────────────────────────────

def fetch_market_caps(tickers):
    """Fetch current market caps from yfinance. Returns DataFrame."""
    section("STEP 1: MARKET CAP ENRICHMENT")

    cache_path = os.path.join(DATA, "market_caps.csv")
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        # Consider successfully resolved tickers (success or no_cap) as done
        resolved = set(cached[cached["lookup_status"].isin(["success", "no_cap"])]["ticker"])
        missing = [t for t in tickers if t not in resolved]
        if not missing:
            print(f"  Using cached market caps ({len(cached)} tickers, {len(resolved)} resolved)")
            return cached
        # Drop failed entries so we can retry them
        cached = cached[cached["ticker"].isin(resolved)]
        print(f"  Cache has {len(resolved)} resolved tickers, retrying {len(missing)} missing/failed...")
    else:
        cached = pd.DataFrame()
        missing = list(tickers)

    results = []
    batch_size = 20
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(missing) - 1) // batch_size + 1
        sys.stdout.write(f"\r  Fetching batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
        sys.stdout.flush()

        for t in batch:
            for attempt in range(3):
                try:
                    ticker_obj = yf.Ticker(t)
                    info = ticker_obj.info
                    cap = info.get("marketCap")
                    results.append({"ticker": t, "market_cap": cap,
                                    "lookup_status": "success" if cap else "no_cap"})
                    break
                except Exception as e:
                    if "Rate" in str(e) or "429" in str(e) or "Too Many" in str(e):
                        wait = 5 * (attempt + 1)
                        time.sleep(wait)
                    else:
                        results.append({"ticker": t, "market_cap": None, "lookup_status": "failed"})
                        break
            else:
                results.append({"ticker": t, "market_cap": None, "lookup_status": "failed"})
            time.sleep(0.2)

        time.sleep(2)

    print()
    new_df = pd.DataFrame(results)
    if len(cached) > 0:
        full = pd.concat([cached, new_df], ignore_index=True)
    else:
        full = new_df

    full["bucket"] = full["market_cap"].apply(assign_bucket)
    full.to_csv(cache_path, index=False)

    total = len(full)
    ok = (full["lookup_status"] == "success").sum()
    no_cap = (full["lookup_status"] == "no_cap").sum()
    failed = (full["lookup_status"] == "failed").sum()
    print(f"  Total tickers: {total}")
    print(f"  Successful with cap: {ok} ({ok/total*100:.1f}%)")
    print(f"  Lookup succeeded but no cap: {no_cap}")
    print(f"  Lookup failed: {failed} ({failed/total*100:.1f}%)")

    if failed > 0:
        fail_list = full[full["lookup_status"] == "failed"]["ticker"].tolist()
        print(f"  Failed tickers (first 20): {fail_list[:20]}")

    print(f"\n  Bucket distribution:")
    bucket_counts = full[full["bucket"].notna()]["bucket"].value_counts()
    for b in BUCKET_ORDER:
        n = bucket_counts.get(b, 0)
        print(f"    {b:>10}: {n:>5} ({n/ok*100:.1f}%)" if ok > 0 else f"    {b:>10}: {n:>5}")

    return full


# ── Step 2: Composition analysis ──────────────────────────────────────

def composition_analysis(df_1y, mcaps):
    section("STEP 2: COMPOSITION ANALYSIS (A-H1)")

    merged = df_1y.merge(mcaps[["ticker", "bucket"]], left_on="ticker_symbol", right_on="ticker", how="left")
    merged = merged[merged["bucket"].notna()].copy()

    ct = pd.crosstab(merged["seq_cat"], merged["bucket"])
    ct = ct.reindex(index=TIER_ORDER, columns=BUCKET_ORDER, fill_value=0)
    print("\n  Signal counts (tier × bucket):")
    print("    " + ct.to_string().replace("\n", "\n    "))

    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print("\n  Row percentages (% of tier within each bucket):")
    print("    " + ct_pct.round(1).to_string().replace("\n", "\n    "))

    chi2, p_chi, dof, expected = stats.chi2_contingency(ct.values)
    print(f"\n  Chi-squared test: χ²={chi2:.1f}, df={dof}, p={p_chi:.2e}")
    if p_chi < 0.01:
        print("  → A-H1 SUPPORTED: tier distributions differ significantly by size bucket")
    else:
        print("  → A-H1 NOT SUPPORTED: no significant difference in size composition")

    # Sequence-level composition
    seq_merged = df_1y.drop_duplicates("sequence_id").merge(
        mcaps[["ticker", "bucket"]], left_on="ticker_symbol", right_on="ticker", how="left"
    )
    seq_merged = seq_merged[seq_merged["bucket"].notna()]
    ct_seq = pd.crosstab(seq_merged["seq_cat"], seq_merged["bucket"])
    ct_seq = ct_seq.reindex(index=TIER_ORDER, columns=BUCKET_ORDER, fill_value=0)
    print("\n  Sequence counts (tier × bucket):")
    print("    " + ct_seq.to_string().replace("\n", "\n    "))

    return merged, ct, ct_pct


# ── Step 3: Within-bucket conviction gradient ─────────────────────────

def within_bucket_gradient(merged):
    section("STEP 3: WITHIN-BUCKET CONVICTION GRADIENT (A-H2)")

    results = []
    first_recs = merged[merged["is_sequence_start"] == True].copy()

    for bucket in BUCKET_ORDER:
        bdata = first_recs[first_recs["bucket"] == bucket]
        print(f"\n  {bucket}:")
        for tier in TIER_ORDER:
            tdata = bdata[bdata["seq_cat"] == tier]["spy_relative_return"]
            n = len(tdata)
            if n < 10:
                print(f"    {TIER_LABELS[tier]:>10}: N={n:>4} (insufficient)")
                results.append({"bucket": bucket, "tier": tier, "n": n,
                                "mean_alpha": None, "median_alpha": None,
                                "se_alpha": None,
                                "t_stat": None, "p_value": None, "sufficient": False})
                continue

            mean_a = tdata.mean()
            med_a = tdata.median()
            t, p = stats.ttest_1samp(tdata, 0)
            se_a = float(tdata.std(ddof=1)) / np.sqrt(n)
            print(f"    {TIER_LABELS[tier]:>10}: N={n:>4}, α={mean_a*100:+.2f}%, "
                  f"med={med_a*100:+.2f}%, t={t:.2f}, p={p:.4f}")
            results.append({"bucket": bucket, "tier": tier, "n": n,
                            "mean_alpha": mean_a, "median_alpha": med_a,
                            "se_alpha": se_a,
                            "t_stat": t, "p_value": p, "sufficient": True})

        # Within-bucket tier comparison
        port_a = bdata[bdata["seq_cat"] == "portfolio"]["spy_relative_return"]
        nohold_a = bdata[bdata["seq_cat"] == "no_hold"]["spy_relative_return"]
        if len(port_a) >= 10 and len(nohold_a) >= 10:
            t, p = stats.ttest_ind(port_a, nohold_a)
            diff = port_a.mean() - nohold_a.mean()
            print(f"    Portfolio vs No-hold: Δα={diff*100:+.2f}pp, t={t:.2f}, p={p:.4f}")
            if p < 0.05:
                print(f"    → Gradient SURVIVES within {bucket}")
            else:
                print(f"    → Gradient NOT significant within {bucket}")

    results_df = pd.DataFrame(results)

    mega_port = results_df[(results_df["bucket"] == "Mega-cap") & (results_df["tier"] == "portfolio")]
    mega_nohold = results_df[(results_df["bucket"] == "Mega-cap") & (results_df["tier"] == "no_hold")]
    if len(mega_port) and len(mega_nohold):
        mp = mega_port.iloc[0]
        mn = mega_nohold.iloc[0]
        if mp["sufficient"] and mn["sufficient"]:
            if (mp["mean_alpha"] or 0) > (mn["mean_alpha"] or 0):
                print("\n  ✓ A-H2: portfolio > no-hold ordering holds within Mega-cap")
            else:
                print("\n  ✗ A-H2: portfolio ≤ no-hold within Mega-cap — size may explain gradient")

    return results_df


# ── Step 4: Size × casual-pick interaction ────────────────────────────

def size_casual_interaction(merged):
    section("STEP 4: SIZE × CASUAL-PICK INTERACTION (A-H3)")

    first_recs = merged[(merged["is_sequence_start"] == True) & (merged["seq_cat"] == "no_hold")].copy()
    results = []

    for bucket in BUCKET_ORDER:
        bdata = first_recs[first_recs["bucket"] == bucket]["spy_relative_return"]
        n = len(bdata)
        if n < 10:
            print(f"  {bucket:>10}: N={n:>4} (insufficient)")
            results.append({"bucket": bucket, "n": n, "mean_alpha": None, "p_value": None})
            continue
        mean_a = bdata.mean()
        t, p = stats.ttest_1samp(bdata, 0)
        print(f"  {bucket:>10}: N={n:>4}, no-hold α={mean_a*100:+.2f}%, t={t:.2f}, p={p:.4f}")
        results.append({"bucket": bucket, "n": n, "mean_alpha": mean_a, "t_stat": t, "p_value": p})

    results_df = pd.DataFrame(results)

    viable = results_df[results_df["mean_alpha"].notna()]
    if len(viable) >= 2:
        large_alpha = viable[viable["bucket"].isin(["Mega-cap", "Large-cap"])]["mean_alpha"].mean()
        small_alpha = viable[viable["bucket"].isin(["Mid-cap", "Small-cap"])]["mean_alpha"].mean()
        if not pd.isna(large_alpha) and not pd.isna(small_alpha):
            if small_alpha < large_alpha:
                print(f"\n  ✓ A-H3: casual-pick underperformance stronger in smaller stocks "
                      f"({small_alpha*100:.2f}% vs {large_alpha*100:.2f}%)")
            else:
                print(f"\n  ✗ A-H3: casual-pick underperformance NOT stronger in smaller stocks")

    return results_df


# ── Step 5: Dual control (size + sector) ──────────────────────────────

def dual_control(merged):
    section("STEP 5: DUAL CONTROL (SIZE + SECTOR)")

    first_recs = merged[merged["is_sequence_start"] == True].copy()
    first_recs["cell"] = first_recs["gics_sector"].astype(str) + " | " + first_recs["bucket"].astype(str)

    print("\n  Cells with N≥30 per tier:")
    cells = first_recs.groupby("cell")
    reported = 0
    for cell_name, cell_data in cells:
        tier_counts = cell_data["seq_cat"].value_counts()
        if all(tier_counts.get(t, 0) >= 30 for t in TIER_ORDER):
            print(f"\n    {cell_name}:")
            for tier in TIER_ORDER:
                tdata = cell_data[cell_data["seq_cat"] == tier]["spy_relative_return"]
                print(f"      {TIER_LABELS[tier]:>10}: N={len(tdata):>3}, α={tdata.mean()*100:+.2f}%")
            reported += 1

    if reported == 0:
        min_threshold = 15
        print(f"  No cells with N≥30 per tier. Relaxing to N≥{min_threshold}:")
        for cell_name, cell_data in cells:
            tier_counts = cell_data["seq_cat"].value_counts()
            if sum(tier_counts.get(t, 0) >= min_threshold for t in TIER_ORDER) >= 2:
                print(f"\n    {cell_name}:")
                for tier in TIER_ORDER:
                    tdata = cell_data[cell_data["seq_cat"] == tier]["spy_relative_return"]
                    n = len(tdata)
                    if n >= min_threshold:
                        print(f"      {TIER_LABELS[tier]:>10}: N={n:>3}, α={tdata.mean()*100:+.2f}%")
                    else:
                        print(f"      {TIER_LABELS[tier]:>10}: N={n:>3} (insufficient)")
                reported += 1

    if reported == 0:
        print("  No cells with sufficient data for dual control.")


# ── Visualizations ────────────────────────────────────────────────────

def plot_composition(ct_pct):
    # Short height: fits above page break when embedded in the PDF
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(BUCKET_ORDER))
    width = 0.25

    for i, tier in enumerate(TIER_ORDER):
        vals = [ct_pct.loc[tier, b] if b in ct_pct.columns else 0 for b in BUCKET_ORDER]
        bars = ax.bar(x + i * width, vals, width, label=TIER_LABELS[tier],
                      color=TIER_COLORS[tier], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xlabel("Size Bucket", fontsize=9.5)
    ax.set_ylabel("% of Tier's Signals", fontsize=9.5)
    ax.set_title(
        "Engagement-tier composition by market-cap bucket",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(BUCKET_ORDER, fontsize=9)
    leg = ax.legend(frameon=True, fontsize=8, loc="upper right", labelspacing=0.3, handlelength=1.1)
    leg.get_frame().set_linewidth(0.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.2)
    path = os.path.join(FIGS, "fig_size_composition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close()
    print(f"  Saved: {path}")


def plot_gradient(results_df):
    viable = results_df[results_df["sufficient"] == True].copy()
    buckets_with_data = [b for b in BUCKET_ORDER
                         if len(viable[(viable["bucket"] == b)]) >= 2]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(buckets_with_data))
    width = 0.25

    for i, tier in enumerate(TIER_ORDER):
        vals = []
        ns = []
        ses = []
        sufficients = []
        for b in buckets_with_data:
            row = results_df[(results_df["bucket"] == b) & (results_df["tier"] == tier)]
            if len(row) and bool(row["sufficient"].values[0]):
                vals.append(float(row["mean_alpha"].values[0]) * 100)
                ses.append(float(row["se_alpha"].values[0]) * 100)
                sufficients.append(True)
            else:
                vals.append(0.0)
                ses.append(0.0)
                sufficients.append(False)
            ns.append(int(row["n"].values[0]) if len(row) else 0)

        bars = ax.bar(x + i * width, vals, width, label=TIER_LABELS[tier],
                      color=TIER_COLORS[tier], alpha=0.85, edgecolor="white", linewidth=0.5)
        # ±1.96·SE whiskers — only for sufficient cells (N≥10).
        whisker_yerr = [1.96 * s if ok else 0 for s, ok in zip(ses, sufficients)]
        ax.errorbar(x + i * width, vals, yerr=whisker_yerr, fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8, capthick=0.8)
        for bar, v, n, ok in zip(bars, vals, ns, sufficients):
            if ok:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3 if v >= 0 else bar.get_height() - 1.2,
                        f"{v:+.1f}%",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
            else:
                bar.set_height(0)
                bar.set_facecolor("none")
                bar.set_edgecolor(TIER_COLORS[tier])
                bar.set_hatch("///")
                bar.set_linewidth(0.8)
                ax.text(bar.get_x() + bar.get_width()/2, 0.4,
                        f"N={n}", ha="center", va="bottom",
                        fontsize=7.5, color="grey", style="italic")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Size Bucket", fontsize=11)
    ax.set_ylabel("First-Rec 1Y SPY-Relative Alpha (%)", fontsize=11)
    ax.set_title("Engagement-tier first-recommendation alpha within market-cap buckets", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(buckets_with_data)
    ax.legend(frameon=True, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(0.5, 0.005,
             "Hatched markers (N<10) indicate cells too thin to plot an alpha; the underlying point estimates are reported in Table 9.  "
             "Whiskers = ±1.96·SE.",
             ha="center", va="bottom", fontsize=8.5, style="italic", color="grey")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(FIGS, "fig_size_gradient.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("Study A: Market Capitalization and the Conviction Gradient")
    print("=" * 70)

    df = pd.read_csv(os.path.join(DATA, "cramer_sequenced.csv"))
    seq = pd.read_csv(os.path.join(DATA, "sequence_summary.csv"))
    seq["seq_cat"] = seq.apply(seq_category, axis=1)

    df_1y = df[(df["requested_horizon"] == "1y") & (df["kept"] == True)].copy()
    df_1y = df_1y.merge(seq[["sequence_id", "has_cramer_owns", "has_hold_recommendation", "seq_cat"]],
                        on="sequence_id", how="left")

    tickers = sorted(df_1y["ticker_symbol"].unique())
    print(f"\n  Unique tickers to look up: {len(tickers)}")

    mcaps = fetch_market_caps(tickers)

    merged, ct, ct_pct = composition_analysis(df_1y, mcaps)
    gradient_results = within_bucket_gradient(merged)
    casual_results = size_casual_interaction(merged)
    dual_control(merged)

    section("VISUALIZATIONS")
    plot_composition(ct_pct)
    plot_gradient(gradient_results)

    all_results = pd.concat([
        gradient_results.assign(test="within_bucket_gradient"),
        casual_results.assign(test="casual_interaction"),
    ], ignore_index=True)
    out_path = os.path.join(DATA, "size_analysis_results.csv")
    all_results.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    section("SUMMARY")
    print("  Study A complete. Key files:")
    print(f"    data/market_caps.csv")
    print(f"    data/size_analysis_results.csv")
    print(f"    figures/fig_size_composition.png")
    print(f"    figures/fig_size_gradient.png")


if __name__ == "__main__":
    main()
