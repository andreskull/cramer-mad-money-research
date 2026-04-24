import pandas as pd
import yfinance as yf
import json
import time
import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
CACHE_FILE = os.path.join(_DATA_DIR, 'gics_sector_map.json')
SNAPSHOT_FILE = os.path.join(_DATA_DIR, 'cramer_snapshot_full.csv')

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def main():
    df = pd.read_csv(SNAPSHOT_FILE)
    tickers = sorted(df['ticker_symbol'].unique())
    print(f"Total unique tickers: {len(tickers)}")

    cache = load_cache()
    already = len([t for t in tickers if t in cache])
    print(f"Already cached: {already}, remaining: {len(tickers) - already}")

    failed = []
    for i, ticker in enumerate(tickers):
        if ticker in cache:
            continue
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', None)
            if sector:
                cache[ticker] = sector
                print(f"[{i+1}/{len(tickers)}] {ticker} -> {sector}")
            else:
                cache[ticker] = 'Unknown'
                print(f"[{i+1}/{len(tickers)}] {ticker} -> Unknown (no sector in info)")
        except Exception as e:
            cache[ticker] = 'Unknown'
            failed.append(ticker)
            print(f"[{i+1}/{len(tickers)}] {ticker} -> FAILED: {e}")

        if (i + 1) % 50 == 0:
            save_cache(cache)
            print(f"  ... checkpoint saved ({len(cache)} mapped)")
            time.sleep(1)

    save_cache(cache)

    # Apply to snapshot
    df['gics_sector'] = df['ticker_symbol'].map(cache).fillna('Unknown')
    df.to_csv(SNAPSHOT_FILE, index=False)

    print(f"\nDone. Mapped {len(cache)} tickers.")
    print(f"Failed: {len(failed)} -> {failed[:20]}")
    print(f"\nSector distribution:")
    print(df[df['requested_horizon'] == '3m']['gics_sector'].value_counts())

if __name__ == '__main__':
    main()
