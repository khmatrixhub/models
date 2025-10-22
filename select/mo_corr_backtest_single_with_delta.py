#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def detect_mo_columns(csv_path, date_col, prefix):
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    mo_cols = [c for c in cols if c.startswith(prefix)]
    if date_col not in cols:
        raise ValueError(f"Date column '{date_col}' not found in {csv_path}. First 10 cols: {cols[:10]}")
    if not mo_cols:
        raise ValueError(f"No MO columns starting with '{prefix}' in {csv_path}.")
    # Find a baseline column that ends with '0000'
    base_candidates = [c for c in cols if c.startswith(prefix) and c.endswith('0000')]
    if not base_candidates:
        raise ValueError("No baseline column ending with '0000' found among MO cols; required for delta computation.")
    return [date_col] + mo_cols, mo_cols

def aggregate_daily_means_mo(csv_path, date_col, mo_cols, chunksize=250_000, limit_rows=None):
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
        grp = chunk.groupby(date_col)
        sum_chunk = grp[mo_cols].sum(min_count=1)
        cnt_chunk = grp[mo_cols].count()

        if sum_acc is None:
            sum_acc = sum_chunk
            count_acc = cnt_chunk
        else:
            sum_acc = sum_acc.add(sum_chunk, fill_value=0.0)
            count_acc = count_acc.add(cnt_chunk, fill_value=0.0)

    if sum_acc is None:
        raise RuntimeError("No data aggregated from MO file. Check inputs.")
    daily_means = sum_acc / count_acc.replace(0, np.nan)
    daily_means = daily_means.sort_index()
    daily_means.index.name = date_col
    return daily_means

def zscore_fit(train_df):
    mean = train_df.mean()
    std = train_df.std().replace(0, np.nan)
    return mean, std

def zscore_apply(df, mean, std):
    return (df - mean) / std

def plot_barh(corr_df, out_png, title):
    corr_sorted = corr_df.sort_values('abs_corr')
    plt.figure(figsize=(8, max(4, 0.3*len(corr_sorted))))
    y = np.arange(len(corr_sorted))
    plt.barh(y, corr_sorted['corr'].values)
    plt.yticks(y, corr_sorted.index)
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel('Pearson r (with next-day Strategy 1 mean)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_equity(dates, eq1, eq2, eq_adapt, out_png, title):
    plt.figure(figsize=(10,6))
    plt.plot(dates, eq1, label='Always Strategy 1')
    plt.plot(dates, eq2, label='Always Strategy 2 (inverse)')
    plt.plot(dates, eq_adapt, label='Adaptive (top-k composite)', linewidth=2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Single-strategy: MO delta features -> correlations and adaptive backtest.")
    ap.add_argument("--mo-file", required=True, help="Path to mo_test_eur_long.csv.bz2 (intraday MO features)")
    ap.add_argument("--strategy1-file", required=True, help="Path to Strategy 1 daily CSV (trend)")
    ap.add_argument("--date-col", default="trade_date", help="Date column name")
    ap.add_argument("--mean-col", default="mean", help="Mean profit per trade column in strategy file")
    ap.add_argument("--count-col", default="count", help="Signal count column in strategy file (for PnL)")
    ap.add_argument("--prefix", default="mo_00", help="Prefix for MO features to include (requires {prefix}0000 baseline)")
    ap.add_argument("--chunksize", type=int, default=250_000, help="Chunk size for MO file")
    ap.add_argument("--limit-rows", type=int, default=None, help="Limit rows from MO file (for faster test)")
    ap.add_argument("--top-k", type=int, default=5, help="Number of top-|corr| MO delta features to use")
    ap.add_argument("--train-split", type=float, default=0.7, help="Fraction of data (by time) for training")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--out-base", default="mo_delta_corr_bt", help="Base name for outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_corr_csv = os.path.join(args.out_dir, f"{args.out_base}_correlations.csv")
    out_corr_png  = os.path.join(args.out_dir, f"{args.out_base}_correlations.png")
    out_equity_png = os.path.join(args.out_dir, f"{args.out_base}_equity.png")
    out_debug_csv = os.path.join(args.out_dir, f"{args.out_base}_debug_merged.csv")

    # 1) Detect MO columns and aggregate daily means
    usecols, mo_cols = detect_mo_columns(args.mo_file, args.date_col, args.prefix)
    mo_daily = aggregate_daily_means_mo(args.mo_file, args.date_col, mo_cols, chunksize=args.chunksize, limit_rows=args.limit_rows)

    # 2) Build delta features: mo_xx := mo_00xx - mo_0000
    base_candidates = [c for c in mo_daily.columns if c.endswith('0000')]
    if not base_candidates:
        raise ValueError("Baseline column ending with '0000' missing after aggregation.")
    base_col = base_candidates[0]
    # Create delta DataFrame excluding the baseline itself
    delta_cols = [c for c in mo_daily.columns if c != base_col]
    mo_delta = mo_daily[delta_cols].subtract(mo_daily[base_col], axis=0)
    # Rename to indicate delta (optional)
    mo_delta.columns = [c.replace(args.prefix, f"{args.prefix}delta_") for c in mo_delta.columns]

    # 3) Shift forward by 1 day (today predicts tomorrow)
    mo_shift = mo_delta.shift(1)

    # 4) Load strategy 1 (trend)
    s1 = pd.read_csv(args.strategy1_file, usecols=[args.date_col, args.mean_col, args.count_col])
    s1[args.date_col] = pd.to_datetime(s1[args.date_col], errors='coerce')
    s1 = s1.dropna(subset=[args.date_col]).sort_values(args.date_col)

    # 5) Merge with shifted deltas; drop NaNs
    merged = s1.merge(mo_shift.reset_index(), on=args.date_col, how='inner').dropna().reset_index(drop=True)

    # 6) Correlations (each delta vs next-day Strategy 1 mean)
    mo_delta_cols = mo_shift.columns.tolist()
    corr_rows = []
    for col in mo_delta_cols:
        r, p = pearsonr(merged[col], merged[args.mean_col])
        corr_rows.append((col, r, abs(r), p))
    corr_df = pd.DataFrame(corr_rows, columns=['feature','corr','abs_corr','p_value']).set_index('feature').sort_values('abs_corr', ascending=False)
    corr_df.to_csv(out_corr_csv)
    plot_barh(corr_df, out_corr_png, "Shifted MO delta features vs next-day Strategy 1 mean")

    # 7) Build composite from top-k
    topk_feats = corr_df.head(args.top_k).index.tolist()
    X = merged[topk_feats].copy()

    # Walk-forward split (by time)
    n = len(merged)
    split_idx = int(n * args.train_split)
    train_idx = merged.index[:split_idx]
    test_idx  = merged.index[split_idx:]

    # z-score on train, apply to both
    mu, sd = zscore_fit(X.loc[train_idx])
    Xz_train = zscore_apply(X.loc[train_idx], mu, sd)
    Xz_test  = zscore_apply(X.loc[test_idx],  mu, sd)

    # Signs/weights from train correlations
    train_corrs = []
    for col in topk_feats:
        r, p = pearsonr(X.loc[train_idx, col], merged.loc[train_idx, args.mean_col])
        train_corrs.append((col, r, abs(r)))
    train_corrs = pd.DataFrame(train_corrs, columns=['feature','r','abs_r']).set_index('feature').loc[topk_feats]
    w = np.sign(train_corrs['r']).values * train_corrs['abs_r'].values

    # Composite score
    score_full = (zscore_apply(merged[topk_feats], mu, sd).fillna(0.0).values * w).sum(axis=1)

    # 8) Backtest: compute PnL for Strategy 1 and its inverse (Strategy 2)
    merged['pnl_1'] = merged[args.mean_col] * merged[args.count_col]
    merged['pnl_2'] = -merged['pnl_1']  # inverse

    merged['eq1'] = merged['pnl_1'].cumsum()
    merged['eq2'] = merged['pnl_2'].cumsum()

    choose_1 = (score_full > 0).astype(int)
    merged['pnl_adapt'] = choose_1 * merged['pnl_1'] + (1 - choose_1) * merged['pnl_2']
    merged['eq_adapt'] = merged['pnl_adapt'].cumsum()

    # Accuracy on train/test for "winner" label
    merged['winner'] = (merged[args.mean_col] > 0).astype(int)  # Strategy 1 wins if its mean > 0
    acc_train = (choose_1[:split_idx] == merged['winner'].iloc[:split_idx]).mean() if split_idx > 0 else np.nan
    acc_test  = (choose_1[split_idx:] == merged['winner'].iloc[split_idx:]).mean() if split_idx < n else np.nan

    # 9) Save debug merged
    debug_cols = [args.date_col, args.mean_col, args.count_col, 'pnl_1','pnl_2','eq1','eq2','pnl_adapt','eq_adapt','winner','score']
    merged['score'] = score_full
    merged[debug_cols + topk_feats].to_csv(out_debug_csv, index=False)

    # 10) Plot equity
    plot_equity(merged[args.date_col], merged['eq1'], merged['eq2'], merged['eq_adapt'], out_equity_png,
                f"Adaptive vs Always-1/2 using MO deltas (top-k={args.top_k})\nTrain acc={acc_train:.3f}, Test acc={acc_test:.3f}")

    print("== Top-k delta features ==", topk_feats)
    print("\nTrain correlations on top-k (weights):")
    print(train_corrs)
    print(f"\nTrain accuracy: {acc_train:.3f}   Test accuracy: {acc_test:.3f}")
    print(f"\nSaved correlations CSV: {out_corr_csv}")
    print(f"Saved correlations PNG: {out_corr_png}")
    print(f"Saved equity PNG:       {out_equity_png}")
    print(f"Saved debug CSV:        {out_debug_csv}")

if __name__ == "__main__":
    main()
