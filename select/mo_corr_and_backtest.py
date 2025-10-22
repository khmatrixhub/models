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
        for c in mo_cols:
            chunk[c] = chunk[c] - chunk['mo_0000']
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
    plt.plot(dates, eq2, label='Always Strategy 2')
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
    ap = argparse.ArgumentParser(description="MO features -> correlations, regime classifier (top-k), and adaptive backtest.")
    ap.add_argument("--mo-file", required=True, help="Path to mo_test_eur_long.csv.bz2 (intraday MO features)")
    ap.add_argument("--strategy1-file", required=True, help="Path to Strategy 1 daily CSV (trend)")
    ap.add_argument("--strategy2-file", required=True, help="Path to Strategy 2 daily CSV (mean-revert)")
    ap.add_argument("--date-col", default="trade_date", help="Date column name")
    ap.add_argument("--mean-col", default="mean", help="Mean profit per trade column in strategy files")
    ap.add_argument("--count-col", default="count", help="Signal count column in strategy files (for PnL)")
    ap.add_argument("--prefix", default="mo_00", help="Prefix for MO features to include")
    ap.add_argument("--chunksize", type=int, default=250_000, help="Chunk size for MO file")
    ap.add_argument("--limit-rows", type=int, default=None, help="Limit rows from MO file (for faster test)")
    ap.add_argument("--top-k", type=int, default=5, help="Number of top-|corr| MO features to use")
    ap.add_argument("--train-split", type=float, default=0.7, help="Fraction of data (by time) for training")
    ap.add_argument("--out-dir", default=".", help="Output directory")
    ap.add_argument("--out-base", default="mo_corr_backtest", help="Base name for outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_corr_csv = os.path.join(args.out_dir, f"{args.out_base}_correlations.csv")
    out_corr_png  = os.path.join(args.out_dir, f"{args.out_base}_correlations.png")
    out_equity_png = os.path.join(args.out_dir, f"{args.out_base}_equity.png")
    out_debug_csv = os.path.join(args.out_dir, f"{args.out_base}_debug_merged.csv")

    # 1) Detect MO columns and aggregate daily means
    usecols, mo_cols = detect_mo_columns(args.mo_file, args.date_col, args.prefix)
        
    mo_daily = aggregate_daily_means_mo(args.mo_file, args.date_col, mo_cols, chunksize=args.chunksize, limit_rows=args.limit_rows)

    # 2) Shift forward by 1 day (today predicts tomorrow)
    mo_shift = mo_daily.shift(1)

    # 3) Load strategies
    s1 = pd.read_csv(args.strategy1_file, usecols=[args.date_col, args.mean_col, args.count_col])
    s2 = pd.read_csv(args.strategy2_file, usecols=[args.date_col, args.mean_col, args.count_col])
    for s in (s1, s2):
        s[args.date_col] = pd.to_datetime(s[args.date_col], errors='coerce')
        s.dropna(subset=[args.date_col], inplace=True)
        s.sort_values(args.date_col, inplace=True)

    # 4) Merge: we only keep overlapping dates across both strategies and mo_shift
    merged = s1.merge(s2, on=args.date_col, suffixes=('_1','_2'))
    merged = merged.merge(mo_shift.reset_index(), on=args.date_col, how='inner')
    merged = merged.dropna().reset_index(drop=True)

    # 5) Compute correlations (each MO vs next-day S1 mean); also p-values
    corr_rows = []
    for col in mo_cols:
        r, p = pearsonr(merged[col], merged[f"{args.mean_col}_1"])
        corr_rows.append((col, r, abs(r), p))
    corr_df = pd.DataFrame(corr_rows, columns=['feature','corr','abs_corr','p_value']).set_index('feature').sort_values('abs_corr', ascending=False)
    corr_df.to_csv(out_corr_csv)
    plot_barh(corr_df, out_corr_png, "Shifted MO features vs next-day Strategy 1 mean")

    # 6) Build labels and composite score
    # Label winner by next-day mean (1 if Strat1 wins, else 0)
    merged['winner'] = (merged[f"{args.mean_col}_1"] > merged[f"{args.mean_col}_2"]).astype(int)

    # Select top-k features by |corr| (avoid NaNs)
    topk_feats = corr_df.head(args.top_k).index.tolist()
    X = merged[topk_feats].copy()

    # Walk-forward split (by time order)
    n = len(merged)
    split_idx = int(n * args.train_split)
    train_idx = merged.index[:split_idx]
    test_idx  = merged.index[split_idx:]

    # Fit z-score on train, apply to both
    mu, sd = zscore_fit(X.loc[train_idx])
    Xz_train = zscore_apply(X.loc[train_idx], mu, sd)
    Xz_test  = zscore_apply(X.loc[test_idx],  mu, sd)

    # Signs and weights from train correlations (to avoid leakage)
    train_corrs = []
    for col in topk_feats:
        r, p = pearsonr(X.loc[train_idx, col], merged.loc[train_idx, f"{args.mean_col}_1"])
        train_corrs.append((col, r, abs(r)))
    train_corrs = pd.DataFrame(train_corrs, columns=['feature','r','abs_r']).set_index('feature').loc[topk_feats]

    # Composite score = sum(sign(r) * abs(r) * z_i)
    w = np.sign(train_corrs['r']).values * train_corrs['abs_r'].values
    score_train = (Xz_train.values * w).sum(axis=1)
    score_test  = (Xz_test.values  * w).sum(axis=1)

    # Predict winner: >0 => Strategy 1, else Strategy 2
    y_train = merged.loc[train_idx, 'winner'].values
    y_test  = merged.loc[test_idx,  'winner'].values
    pred_train = (score_train > 0).astype(int)
    pred_test  = (score_test  > 0).astype(int)

    acc_train = (pred_train == y_train).mean() if len(y_train) else np.nan
    acc_test  = (pred_test  == y_test ).mean() if len(y_test) else np.nan

    # 7) Backtest
    # Compute daily PnL for both strategies
    merged['pnl_1'] = merged[f"{args.mean_col}_1"] * merged[f"{args.count_col}_1"]
    merged['pnl_2'] = merged[f"{args.mean_col}_2"] * merged[f"{args.count_col}_2"]

    # Always-1 and Always-2
    merged['eq1'] = merged['pnl_1'].cumsum()
    merged['eq2'] = merged['pnl_2'].cumsum()

    # Adaptive: use train params across full sample, but *decisions* are based on score>0 per period
    # For fairness, we recompute z-scores using train mu/sd, and weight vector w from train
    Xz_full = zscore_apply(merged[topk_feats], mu, sd).fillna(0.0)
    score_full = (Xz_full.values * w).sum(axis=1)
    choose_1 = (score_full > 0).astype(int)
    merged['pnl_adapt'] = choose_1 * merged['pnl_1'] + (1 - choose_1) * merged['pnl_2']
    merged['eq_adapt'] = merged['pnl_adapt'].cumsum()

    # 8) Save debug merged (dates + winners + scores)
    debug_cols = [args.date_col, 'winner'] + [f"{args.mean_col}_1", f"{args.mean_col}_2", 'pnl_1','pnl_2','eq1','eq2','pnl_adapt','eq_adapt'] + topk_feats
    merged[debug_cols].to_csv(out_debug_csv, index=False)

    # 9) Plot equity curves
    plot_equity(merged[args.date_col], merged['eq1'], merged['eq2'], merged['eq_adapt'], out_equity_png,
                f"Adaptive switching vs Always-1/2 (top-k={args.top_k})\nTrain acc={acc_train:.3f}, Test acc={acc_test:.3f}")

    # 10) Console summary
    print("== Top-k features ==", topk_feats)
    print("\nTrain correlations on top-k (to set weights):")
    print(train_corrs)
    print(f"\nTrain accuracy: {acc_train:.3f}   Test accuracy: {acc_test:.3f}")
    print(f"\nSaved correlations CSV: {out_corr_csv}")
    print(f"Saved correlations PNG: {out_corr_png}")
    print(f"Saved equity PNG:       {out_equity_png}")
    print(f"Saved debug CSV:        {out_debug_csv}")

if __name__ == "__main__":
    main()
