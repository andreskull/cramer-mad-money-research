#!/usr/bin/env python3
"""
validate_cramer_performance.py

Reproducibility Kit for "Jim Cramer Performance Research (2018-2024)".
This script demonstrates the "Corrected Signal-to-Performance Processing Pipeline"
used to generate the research results. It allows public researchers to verify
both the T+1 Open entry logic and the "Implicit Close" (flip) position boundary logic.

Requirements:
    pandas

Usage:
    python validate_cramer_performance.py --signals actionable_signals.csv --prices prices.csv

Data Formats Expected:
- actionable_signals.csv:
    id, financial_instrument_id, signal_date, signal_type, position_action, section_raw, referenced_source_raw
- prices.csv:
    financial_instrument_id, trade_date, adj_open, adj_close
"""
import sys
import argparse
import logging
from datetime import timedelta
try:
    import pandas as pd
except ImportError:
    print("Please install pandas to run the reproducibility kit: pip install pandas")
    sys.exit(1)

# Horizon mapping (Trading Days)
HORIZON_MAP = {
    '1w': 5,
    '1m': 21,
    '3m': 63,
    '6m': 126,
    '1y': 252
}

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce Cramer Performance Metrics.")
    parser.add_argument('--signals', required=True, help='Path to actionable signals CSV extract')
    parser.add_argument('--prices', required=True, help='Path to prices CSV extract')
    return parser.parse_args()


def apply_implicit_close_logic(signals_df):
    """
    Applies the "Implicit Close" tracking to compute valid signal boundaries.
    """
    # Sort signals chronologically per instrument
    df = signals_df.sort_values(by=['financial_instrument_id', 'signal_date']).copy()
    
    # We will compute a 'next_close_date' for each opening signal
    results = []
    
    # Group by instrument
    for instrument_id, group in df.groupby('financial_instrument_id'):
        rows = group.to_dict('records')
        for i, current_row in enumerate(rows):
            if current_row['position_action'] not in ('start_long', 'hold_long', 'start_short', 'hold_short'):
                continue
            
            is_long = current_row['position_action'] in ('start_long', 'hold_long')
            
            next_close_date = None
            
            # Look ahead for a closing event
            for j in range(i + 1, len(rows)):
                future_row = rows[j]
                
                # Explicit close
                if is_long and future_row['position_action'] == 'close_long':
                    next_close_date = future_row['signal_date']
                    break
                elif not is_long and future_row['position_action'] == 'close_short':
                    next_close_date = future_row['signal_date']
                    break
                
                # Implicit close (direction flip)
                if is_long and future_row['position_action'] in ('start_short', 'hold_short'):
                    next_close_date = future_row['signal_date']
                    break
                elif not is_long and future_row['position_action'] in ('start_long', 'hold_long'):
                    next_close_date = future_row['signal_date']
                    break
            
            current_row['next_close_date'] = next_close_date
            results.append(current_row)
            
    return pd.DataFrame(results)

def calculate_performance(signals_df, prices_df):
    """
    Calculates T+1 entry performance across all standard horizons.
    """
    records = []
    for _, signal in signals_df.iterrows():
        inst_id = signal['financial_instrument_id']
        sig_date = signal['signal_date']
        next_close = signal['next_close_date']
        is_long = signal['position_action'] in ('start_long', 'hold_long')
        
        # Get prices for this instrument
        inst_prices = prices_df[prices_df['financial_instrument_id'] == inst_id].sort_values('trade_date')
        
        # Determine entry date (T+1 Open)
        future_prices = inst_prices[inst_prices['trade_date'] > sig_date]
        if future_prices.empty:
            continue
        
        entry_row = future_prices.iloc[0]
        entry_date = entry_row['trade_date']
        start_price = entry_row['adj_open']
        
        if pd.isna(start_price) or start_price <= 0:
            continue
            
        # Calculate for each horizon
        for horizon_name, days in HORIZON_MAP.items():
            # Get the row at T+1+days index
            entry_idx = inst_prices.index.get_loc(entry_row.name)
            eval_idx = entry_idx + days
            
            if eval_idx >= len(inst_prices):
                continue # Horizon incomplete
                
            eval_row = inst_prices.iloc[eval_idx]
            eval_date = eval_row['trade_date']
            
            # Apply Implicit Close truncation check
            if next_close and eval_date >= next_close:
                # Discard corrupted horizon
                continue
                
            end_price = eval_row['adj_close']
            
            if is_long:
                abs_ret = (end_price - start_price) / start_price
            else:
                abs_ret = (start_price - end_price) / start_price
                
            records.append({
                'signal_id': signal['id'],
                'requested_horizon': horizon_name,
                'entry_date': entry_date,
                'eval_date': eval_date,
                'abs_return': abs_ret
            })
            
    return pd.DataFrame(records)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    logging.info(f"Loading signals from {args.signals}")
    try:
        signals_df = pd.read_csv(args.signals, parse_dates=['signal_date'])
    except Exception as e:
        logging.error(f"Failed to read signals: {e}")
        return
        
    logging.info(f"Loading prices from {args.prices}")
    try:
        prices_df = pd.read_csv(args.prices, parse_dates=['trade_date'])
    except Exception as e:
        logging.error(f"Failed to read prices: {e}")
        return
        
    logging.info("Applying Implicit Close tracking...")
    signals_enhanced = apply_implicit_close_logic(signals_df)
    
    logging.info("Computing T+1 horizon performance...")
    perf_df = calculate_performance(signals_enhanced, prices_df)
    
    logging.info("Replication Complete.")
    print("\\n--- Sample Validated Performance Outputs ---")
    print(perf_df.head(20).to_string())
    
    # Save the replicated output
    out_file = 'replicated_cramer_performance.csv'
    perf_df.to_csv(out_file, index=False)
    logging.info(f"Saved reproducible dataset to {out_file}")


if __name__ == "__main__":
    main()
