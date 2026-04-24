"""
Hold Signal Reclassification — LLM Classification & Merge.

Input: hold_signals_to_classify.csv (provided in data/) — deduplicated hold_long
signals with proof_segments_text already filled in.

Commands:
  classify  Run LLM classification → hold_classifications.csv
  merge     Merge hold_classifications.csv back into cramer_snapshot_full.csv
  validate  Check classification quality against known Trust holdings

Typical run order:
  1. python reclassify_holds.py classify    ← calls grok-4-fast-non-reasoning for all ~4,900 signals
  2. python reclassify_holds.py merge       ← joins classifications back into the snapshot
  3. python reclassify_holds.py validate    ← sanity check

Note: The data preparation steps (enrich, extract) that fetch proof segment text
from our database are not included here. The provided CSV files already contain
all the data needed to run classification.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE = Path(__file__).resolve().parent.parent / "data"


# ── Commands ────────────────────────────────────────────────────────────────

def merge():
    """Merge LLM classifications back into the dataset."""
    print("Loading classifications...")

    cls_path = BASE / "hold_classifications.csv"
    if not cls_path.exists():
        print(f"ERROR: {cls_path} not found.")
        print("Run LLM classification first, then save results as hold_classifications.csv")
        print("Expected columns: signal_id, hold_subtype (cramer_owns | hold_recommendation)")
        return

    cls = pd.read_csv(cls_path)
    print(f"  Classifications: {len(cls):,} rows")
    print(f"  Distribution:\n{cls['hold_subtype'].value_counts().to_string()}")

    valid = {"cramer_owns", "hold_recommendation"}
    invalid = cls[~cls["hold_subtype"].isin(valid)]
    if len(invalid) > 0:
        print(f"\nWARNING: {len(invalid)} rows with unexpected values:")
        print(invalid["hold_subtype"].value_counts().to_string())

    print("\nUpdating cramer_snapshot_full.csv with hold_subtype...")
    df = pd.read_csv(BASE / "cramer_snapshot_full.csv")
    df = df.merge(cls[["signal_id", "hold_subtype"]], on="signal_id", how="left")
    df.loc[df["signal_type"] != "hold_long", "hold_subtype"] = np.nan

    print(f"  hold_subtype distribution (all rows):")
    print(df["hold_subtype"].value_counts(dropna=False).to_string())

    df.to_csv(BASE / "cramer_snapshot_full.csv", index=False)
    print("Updated cramer_snapshot_full.csv")


def validate():
    """Check classification quality against known Charitable Trust holdings."""
    cls_path = BASE / "hold_classifications.csv"
    if not cls_path.exists():
        cls_path = BASE / "hold_signals_to_classify.csv"
        if not cls_path.exists():
            print("No classification files found.")
            return
        cls = pd.read_csv(cls_path)
        print("(Using extract file — LLM classification not yet done)")
    else:
        cls = pd.read_csv(cls_path)

    if "ticker_symbol" not in cls.columns:
        extract_df = pd.read_csv(BASE / "hold_signals_to_classify.csv")
        cls = cls.merge(extract_df[["signal_id", "ticker_symbol"]], on="signal_id", how="left")

    known_trust = [
        "AAPL", "NVDA", "MSFT", "META", "AMZN", "GOOGL", "LLY", "COST",
        "TJX", "CRM", "CRWD", "WFC", "GS", "MS", "HON", "DIS", "SBUX",
        "PANW", "DHR", "ETN", "GEHC", "LIN", "MU", "F", "ABBV",
    ]

    print("=== Validation: Known Trust Holdings ===")
    for ticker in known_trust[:15]:
        sub = cls[cls["ticker_symbol"] == ticker]
        if len(sub) == 0:
            print(f"  {ticker:6s}: NO hold signals")
            continue
        if "hold_subtype" in sub.columns:
            n_owns = (sub["hold_subtype"] == "cramer_owns").sum()
            n_rec  = (sub["hold_subtype"] == "hold_recommendation").sum()
            n_null = sub["hold_subtype"].isna().sum()
            print(f"  {ticker:6s}: {len(sub):3d} holds — owns={n_owns}, rec={n_rec}, unclassified={n_null}")
        else:
            print(f"  {ticker:6s}: {len(sub):3d} holds — not yet classified")

    if "hold_subtype" in cls.columns:
        print(f"\n=== Overall Distribution ===")
        print(cls["hold_subtype"].value_counts(dropna=False).to_string())
        owns_t = cls[cls["hold_subtype"] == "cramer_owns"]["ticker_symbol"].nunique()
        rec_t  = cls[cls["hold_subtype"] == "hold_recommendation"]["ticker_symbol"].nunique()
        print(f"\n  cramer_owns tickers:        {owns_t}")
        print(f"  hold_recommendation tickers: {rec_t}")
        print(f"  Target cramer_owns range: 100–250 unique tickers (7 years of Trust turnover)")


def classify():
    """Run LLM classification using grok-4-fast-non-reasoning via xAI REST API.

    Reads hold_signals_to_classify.csv, calls the xAI API for each signal,
    and saves results to hold_classifications.csv.

    Resume-safe: if hold_classifications.csv already exists, skips already-classified
    signal_ids so a partial run can be continued without re-processing.

    Requires XAI_API_KEY in environment (copy env.example to .env or export).
    """
    MODEL = "grok-4-fast-non-reasoning"
    XAI_URL = "https://api.x.ai/v1/chat/completions"
    DELAY_SECONDS = 0.5

    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("ERROR: XAI_API_KEY not set. Copy env.example to .env or export XAI_API_KEY.")
        sys.exit(1)

    input_path = BASE / "hold_signals_to_classify.csv"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        print("The file hold_signals_to_classify.csv should be in the data/ directory.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} signals from hold_signals_to_classify.csv")

    out_path = BASE / "hold_classifications.csv"
    already_done = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        already_done = set(existing["signal_id"].tolist())
        print(f"Resuming — {len(already_done):,} already classified, {len(df) - len(already_done):,} remaining")

    to_classify = df[~df["signal_id"].isin(already_done)].copy()
    if len(to_classify) == 0:
        print("All signals already classified. Run 'merge' next.")
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    SYSTEM_PROMPT = (
        "You classify Jim Cramer stock signals. "
        "Reply with exactly one word: cramer_owns or hold_recommendation. "
        "No explanation. No punctuation. Only one of those two words."
    )

    def build_user_prompt(row) -> str:
        proof = str(row.get("proof_segments_text", "") or "").strip()
        summary = str(row.get("reasoning_summary", "") or "").strip()
        segment = str(row.get("show_segment_heuristic", "") or "").strip()

        evidence = ""
        if proof:
            evidence += f'Verbatim transcript:\n"{proof}"\n\n'
        if summary:
            evidence += f'Reasoning summary:\n"{summary}"\n\n'

        return (
            f"Ticker: {row['ticker_symbol']} | Date: {row['signal_date']} | Segment: {segment}\n\n"
            f"{evidence}"
            "cramer_owns — Cramer personally holds this (Charitable Trust, Investing Club, personal portfolio).\n"
            "  Indicators: 'I own', 'we hold', 'my trust owns', 'in the trust', 'charitable trust',\n"
            "              'investing club', 'our position', 'don't trade', 'trust favorite'.\n"
            "  IMPORTANT: 'own it' in advisory context ('you should own it') = hold_recommendation.\n\n"
            "hold_recommendation — Cramer advises holding but does NOT indicate personal ownership.\n"
            "  Indicators: 'solid hold', 'keep holding', 'stay with it', 'I'd hold', caller Q&A.\n\n"
            "When ambiguous, reply hold_recommendation (precision over recall).\n\n"
            "Reply with one word only:"
        )

    results = []
    errors = []
    valid_labels = {"cramer_owns", "hold_recommendation"}

    print(f"\nClassifying {len(to_classify):,} signals with {MODEL}...")
    print("─" * 60)

    for i, (_, row) in enumerate(to_classify.iterrows(), 1):
        signal_id = row["signal_id"]
        ticker = row["ticker_symbol"]
        date = row["signal_date"]

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)},
            ],
            "max_completion_tokens": 10,
            "temperature": 0.0,
        }

        try:
            resp = requests.post(XAI_URL, headers=headers, json=payload, timeout=30.0)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
            label = raw.rstrip(".,;:").strip()

            if label not in valid_labels:
                if "cramer_owns" in label:
                    label = "cramer_owns"
                elif "hold_recommendation" in label:
                    label = "hold_recommendation"
                else:
                    print(f"  [{i:4d}] {ticker:6s} ({date})  unexpected response: '{raw}' -> defaulting to hold_recommendation")
                    label = "hold_recommendation"

            results.append({"signal_id": signal_id, "hold_subtype": label})

            owns_marker = "owns" if label == "cramer_owns" else " rec"
            if i % 50 == 0 or i <= 5:
                n_owns = sum(1 for r in results if r["hold_subtype"] == "cramer_owns")
                print(f"  [{i:4d}/{len(to_classify)}] {ticker:6s} ({date})  {owns_marker}   [owns so far: {n_owns}]")

        except Exception as e:
            print(f"  [{i:4d}] {ticker:6s} ({date})  ERROR: {e}")
            errors.append({"signal_id": signal_id, "error": str(e)})
            label = None

        if i % 100 == 0 and results:
            _save_classifications(out_path, already_done, results)
            print(f"  -> Progress saved ({len(already_done) + len(results):,} total)")

        if i < len(to_classify):
            time.sleep(DELAY_SECONDS)

    _save_classifications(out_path, already_done, results)

    n_owns = sum(1 for r in results if r["hold_subtype"] == "cramer_owns")
    n_rec = sum(1 for r in results if r["hold_subtype"] == "hold_recommendation")
    total_done = len(already_done) + len(results)

    print("\n-- Classification complete --")
    print(f"  This run:   {len(results):,} classified ({n_owns} cramer_owns, {n_rec} hold_recommendation)")
    print(f"  Errors:     {len(errors)}")
    print(f"  Total done: {total_done:,} / {len(df):,}")
    if len(errors) > 0:
        print(f"  Re-run 'classify' to retry {len(errors)} failed signals")
    if total_done == len(df):
        print(f"\nAll signals classified. Run 'merge' next.")
    else:
        print(f"\n  Run 'classify' again to resume the remaining {len(df) - total_done:,} signals.")


def _save_classifications(out_path: Path, already_done: set, new_results: list):
    """Append new results to the classifications CSV (merge with existing rows)."""
    if not new_results:
        return
    new_df = pd.DataFrame(new_results)
    if out_path.exists() and already_done:
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["signal_id"], keep="last")
        combined.to_csv(out_path, index=False)
    else:
        new_df.to_csv(out_path, index=False)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reclassify_holds.py [classify|merge|validate]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "classify":
        classify()
    elif cmd == "merge":
        merge()
    elif cmd == "validate":
        validate()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python reclassify_holds.py [classify|merge|validate]")
        sys.exit(1)
