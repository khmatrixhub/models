#!/usr/bin/env python3
import argparse
import os
import bz2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detect_mo_columns(csv_path, date_col, prefix):
    # Read only header to detect columns
    # Using pandas read_csv with nrows=0 to parse columns (works with bz2 path directly)
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    mo_cols = [c for c in cols if c.startswith(prefix)]
    if date_col not in cols:
        raise ValueError(f"Date column '{date_col}' not found in {csv_path}. Available: {cols[:10]}...")
    if not mo_cols:
        raise ValueError(f"No columns starting with '{prefix}' found in {csv_path}.")
    return [date_col] + mo_cols, mo_cols

def aggregate_daily_means_mo(csv_path, date_col, mo_cols, chunksize=250_000, limit_rows=None):
    # We'll compute per-date sums and counts (non-null) for each mo column, chunk by chunk
    sum_acc = None
    count_acc = None
    rows_read = 0

    usecols = [date_col] + mo_cols
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        if limit_rows is not None and rows_read >= limit_rows:
            break
        if limit_rows is not None:
            remaining = max(0, limit_rows - rows_read)
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        rows_read += len(chunk)

        chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
        chunk = chunk.dropna(subset=[date_col])
        # group sums and counts per day
        grp = chunk.groupby(date_col)
        sum_chunk = grp[mo_cols].sum(min_count=1)
        cnt_chunk = grp[mo_cols].count()

        if sum_acc is None:
            sum_acc = sum_chunk
            count_acc = cnt_chunk
        else:
            # align on index (dates)
            sum_acc = sum_acc.add(sum_chunk, fill_value=0.0)
            count_acc = count_acc.add(cnt_chunk, fill_value=0.0)

    if sum_acc is None:
        raise RuntimeError("No data aggregated. Check inputs or limit_rows too small.")

    daily_means = sum_acc / count_acc.replace(0, np.nan)
    daily_means = daily_means.sort_index()
    daily_means.index.name = date_col
    return daily_means

def make_barh_plot(corr_series, out_png):
    # Simple horizontal bar chart (sorted)
    corr_sorted = corr_series.sort_values()
    plt.figure(figsize=(8, max(4, 0.25*len(corr_sorted))))
    y = np.arange(len(corr_sorted))
    plt.barh(y, corr_sorted.values)
    plt.yticks(y, corr_sorted.index)
    plt.axvline(0, linewidth=1)
    plt.title("Correlation of shifted mo_00xx features vs next-day mean profit (Strategy 1)")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Compute correlation between shifted mo_00xx features (daily averaged) and next-day mean profit for Strategy 1.")
    ap.add_argument("--mo-file", required=True, help="Path to mo_test_eur_long.csv.bz2 (or similar)")
    ap.add_argument("--strategy-file", required=True, help="Path to Strategy 1 daily CSV (e.g., sliding_63_daily_all.csv.bz2)")
    ap.add_argument("--date-col", default="trade_date", help="Date column name (default: trade_date)")
    ap.add_argument("--mean-col", default="mean", help="Strategy file mean-profit column (default: mean)")
    ap.add_argument("--prefix", default="mo_00", help="Prefix for mo features (default: mo_00)")
    ap.add_argument("--chunksize", type=int, default=250_000, help="Chunk size for reading MO file")
    ap.add_argument("--limit-rows", type=int, default=None, help="Optional limit on rows read from MO file (for quick testing)")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--out-base", default="mo_corr_strategy1", help="Base name for outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.out_base}.csv")
    out_png = os.path.join(args.out_dir, f"{args.out_base}.png")

    # 1) Detect mo columns
    usecols, mo_cols = detect_mo_columns(args.mo_file, args.date_col, args.prefix)

    # 2) Aggregate daily means (chunked)
    daily_means = aggregate_daily_means_mo(
        args.mo_file,
        args.date_col,
        mo_cols,
        chunksize=args.chunksize,
        limit_rows=args.limit_rows
    )

    # 3) Shift by 1 day to make them predictive
    daily_means_shifted = daily_means.shift(1)

    # 4) Load strategy file and merge
    strat = pd.read_csv(args.strategy_file, usecols=[args.date_col, args.mean_col])
    strat[args.date_col] = pd.to_datetime(strat[args.date_col], errors='coerce')
    strat = strat.dropna(subset=[args.date_col]).sort_values(args.date_col)

    merged = strat.merge(
        daily_means_shifted.reset_index(),
        on=args.date_col,
        how="inner"
    ).dropna()

    if merged.empty:
        raise RuntimeError("Merged dataframe is empty. Check date alignment between files.")

    # 5) Compute correlations (each mo column vs strategy mean)
    corr_series = merged[mo_cols + [args.mean_col]].corr()[args.mean_col].drop(args.mean_col)

    # 6) Save outputs
    corr_series.sort_values(ascending=False).to_csv(out_csv, header=["correlation"])

    # 7) Plot
    make_barh_plot(corr_series, out_png)

    # 8) Print quick summary to console
    print("Top positive correlations:\n", corr_series.sort_values(ascending=False).head(10))
    print("\nTop negative correlations:\n", corr_series.sort_values(ascending=True).head(10))
    print(f"\nSaved correlation CSV -> {out_csv}")
    print(f"Saved plot PNG       -> {out_png}")

if __name__ == "__main__":
    main()
