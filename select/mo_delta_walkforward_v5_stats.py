#!/usr/bin/env python3
import argparse, os, sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def log(msg):
    print(msg); sys.stdout.flush()

def detect_mo_columns(csv_path, date_col, prefix):
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    mo_cols = [c for c in cols if c.startswith(prefix)]
    base_candidates = [c for c in mo_cols if c.endswith('0000')]
    if date_col not in cols:
        raise ValueError(f"Date column '{date_col}' not found.")
    if not mo_cols:
        raise ValueError(f"No MO columns with prefix '{prefix}'.")
    if not base_candidates:
        raise ValueError("No baseline MO column ending with '0000'.")
    return [date_col] + mo_cols, mo_cols, base_candidates[0]

def aggregate_daily_means(csv_path, date_col, mo_cols, chunksize=250_000, limit_rows=None):
    sum_acc = None; cnt_acc = None; read = 0
    usecols = [date_col] + mo_cols
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        if limit_rows is not None and read >= limit_rows: break
        if limit_rows is not None:
            keep = max(0, limit_rows - read)
            if len(chunk) > keep: chunk = chunk.iloc[:keep]
        read += len(chunk)
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
        chunk.dropna(subset=[date_col], inplace=True)
        g = chunk.groupby(date_col)
        s = g[mo_cols].sum(min_count=1); c = g[mo_cols].count()
        if sum_acc is None:
            sum_acc, cnt_acc = s, c
        else:
            sum_acc = sum_acc.add(s, fill_value=0.0)
            cnt_acc = cnt_acc.add(c, fill_value=0.0)
    if sum_acc is None:
        raise RuntimeError("No data aggregated from MO file.")
    daily = (sum_acc / cnt_acc.replace(0, np.nan)).sort_index()
    daily.index.name = date_col
    return daily

def zfit(X: pd.DataFrame):
    X = X.apply(pd.to_numeric, errors="coerce").infer_objects()
    mu = X.mean(numeric_only=True)
    sd = X.std(numeric_only=True).replace(0, np.nan)
    return mu, sd

def zapp(X: pd.DataFrame, mu: pd.Series, sd: pd.Series):
    X = X.apply(pd.to_numeric, errors="coerce").infer_objects()
    Z = (X - mu) / sd
    return Z.astype(float).fillna(0.0)

def equity_plot(dates, eq1, eq2, eq_adapt, out_png, title):
    plt.figure(figsize=(10,6))
    plt.plot(dates, eq1, label="Always 1")
    plt.plot(dates, eq2, label="Always 2")
    plt.plot(dates, eq_adapt, label="Adaptive", linewidth=2)
    plt.grid(True); plt.legend(); plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def compute_topk_and_weights(Xtr: pd.DataFrame, ytr: pd.Series, top_k: int):
    stats = []
    for c in Xtr.columns:
        x = pd.to_numeric(Xtr[c], errors="coerce")
        if x.notna().sum() < 2:
            r = 0.0
        else:
            r, _ = pearsonr(x.fillna(0.0).values, ytr.values)
            if not np.isfinite(r): r = 0.0
        stats.append((c, r, abs(r)))
    stats.sort(key=lambda x: x[2], reverse=True)
    top = [c for c, r, ar in stats[:top_k]]
    w = np.array([np.sign([r for c2,r,_ in stats if c2==c][0]) * [abs(r) for c2,r,_ in stats if c2==c][0] for c in top], dtype=float) if top else np.array([])
    return top, w

def max_drawdown(series: np.ndarray):
    # series is equity curve
    if len(series) == 0:
        return 0.0
    peaks = np.maximum.accumulate(series)
    dd = peaks - series
    return float(dd.max()) if len(dd) else 0.0

def sharpe_ratio(pnl: np.ndarray, periods_per_year: int = 252):
    # Use daily pnl (not %), risk-free ~0; handle zero std
    if len(pnl) == 0:
        return 0.0
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(mu / sd * np.sqrt(periods_per_year))

def compute_stats(counts_eval: np.ndarray, pnl_series: np.ndarray, label: str):
    total_trades = float(counts_eval.sum())
    total_pnl = float(pnl_series.sum())
    avg_pnl_per_trade = float(total_pnl / total_trades) if total_trades > 0 else 0.0
    equity = np.cumsum(pnl_series)
    mdd = max_drawdown(equity)
    sr = sharpe_ratio(pnl_series)
    return {
        "strategy": label,
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "max_drawdown": mdd,
        "sharpe_ratio": sr,
    }

def main():
    ap = argparse.ArgumentParser(description="Walk-forward with GLOBAL inverse mapping when calibration favors it (v5) + stats.")
    ap.add_argument("--mo-file", required=True)
    ap.add_argument("--strategy1-file", required=True)
    ap.add_argument("--date-col", default="trade_date")
    ap.add_argument("--mean-col", default="mean")
    ap.add_argument("--count-col", default="count")
    ap.add_argument("--prefix", default="mo_00")
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--lookback-days", type=int, default=120)
    ap.add_argument("--calibration-days", type=int, default=None)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--out-base", default="mo_walkfwd_v5_stats")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_equity = os.path.join(args.out_dir, f"{args.out_base}_equity.png")
    out_log    = os.path.join(args.out_dir, f"{args.out_base}_decisions.csv")
    out_feats  = os.path.join(args.out_dir, f"{args.out_base}_feature_usage.csv")
    out_stats  = os.path.join(args.out_dir, f"{args.out_base}_stats.csv")

    log("Loading & aligning data...")
    usecols, mo_cols, base_col = detect_mo_columns(args.mo_file, args.date_col, args.prefix)
    mo_daily = aggregate_daily_means(args.mo_file, args.date_col, mo_cols, args.chunksize, args.limit_rows)
    delta_cols = [c for c in mo_daily.columns if c != base_col]
    mo_delta = mo_daily[delta_cols].subtract(mo_daily[base_col], axis=0)
    mo_delta.columns = [c.replace(args.prefix, f"{args.prefix}delta_") for c in mo_delta.columns]
    mo_shift = mo_delta.shift(1).dropna()

    s1 = pd.read_csv(args.strategy1_file, usecols=[args.date_col, args.mean_col, args.count_col])
    s1[args.date_col] = pd.to_datetime(s1[args.date_col], errors='coerce')
    s1.dropna(subset=[args.date_col], inplace=True)
    s1.sort_values(args.date_col, inplace=True)

    df = s1.merge(mo_shift.reset_index(), on=args.date_col, how="inner").dropna().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Merged dataset is empty after alignment. Check dates/shift.")
    dates = df[args.date_col].values
    y_mean = pd.to_numeric(df[args.mean_col], errors="coerce").fillna(0.0).values
    counts = pd.to_numeric(df[args.count_col], errors="coerce").fillna(0.0).values
    pnl1 = y_mean * counts
    pnl2 = -pnl1

    feat_cols = [c for c in df.columns if c.startswith(f"{args.prefix}delta_")]
    n = len(df); look = args.lookback_days
    log(f"Rows={n}, features={len(feat_cols)}, lookback={look}")
    if len(feat_cols) == 0:
        raise RuntimeError("No delta features found after processing.")
    if n < look + 3:
        look = max(20, n//4)
        log(f"Adjusted lookback to {look} due to limited rows.")

    # --- Global calibration ---
    calib_days = args.calibration_days or min(n//3, look*2)
    calib_days = max(calib_days, look+5)
    calib_end = min(n-2, calib_days)
    log(f"Calibrating mapping over indices [{look}, {calib_end}) ...")

    pnl_adapt_pos_sum = 0.0
    pnl_adapt_neg_sum = 0.0
    for t in range(look, calib_end):
        train_slice = slice(t-look, t)
        Xtr, ytr = df.loc[train_slice, feat_cols], df.loc[train_slice, args.mean_col]
        # top-k & weights
        stats = []
        for c in Xtr.columns:
            x = pd.to_numeric(Xtr[c], errors="coerce")
            if x.notna().sum() < 2:
                r = 0.0
            else:
                r, _ = pearsonr(x.fillna(0.0).values, ytr.values)
                if not np.isfinite(r): r = 0.0
            stats.append((c, r, abs(r)))
        stats.sort(key=lambda x: x[2], reverse=True)
        top = [c for c, r, ar in stats[:args.top_k]]
        if not top:
            continue
        mu = Xtr[top].apply(pd.to_numeric, errors="coerce").infer_objects().mean(numeric_only=True)
        sd = Xtr[top].apply(pd.to_numeric, errors="coerce").infer_objects().std(numeric_only=True).replace(0, np.nan)
        w = np.array([np.sign([r for c2,r,_ in stats if c2==c][0]) * [abs(r) for c2,r,_ in stats if c2==c][0] for c in top], dtype=float)
        Xt = df.loc[t, top]
        Xt_z = ((Xt - mu) / sd).astype(float).fillna(0.0).values
        score_t = float(np.dot(Xt_z, w))
        pnl1_next, pnl2_next = pnl1[t+1], pnl2[t+1]
        pnl_adapt_pos_sum += (pnl1_next if score_t > 0 else pnl2_next)
        pnl_adapt_neg_sum += (pnl1_next if score_t < 0 else pnl2_next)

    global_mapping = "S1_if_neg" if pnl_adapt_pos_sum >= pnl_adapt_neg_sum else "S1_if_pos"
    log(f"GLOBAL mapping selected: {global_mapping}  (pos_sum={pnl_adapt_pos_sum:.4f}, neg_sum={pnl_adapt_neg_sum:.4f})")

    # --- Walk-forward with fixed mapping ---
    decisions = []; feat_usage_rows = []
    for t in range(look, n-1):
        train_slice = slice(t-look, t)
        Xtr, ytr = df.loc[train_slice, feat_cols], df.loc[train_slice, args.mean_col]
        # top-k & weights
        stats = []
        for c in Xtr.columns:
            x = pd.to_numeric(Xtr[c], errors="coerce")
            if x.notna().sum() < 2:
                r = 0.0
            else:
                r, _ = pearsonr(x.fillna(0.0).values, ytr.values)
                if not np.isfinite(r): r = 0.0
            stats.append((c, r, abs(r)))
        stats.sort(key=lambda x: x[2], reverse=True)
        top = [c for c, r, ar in stats[:args.top_k]]
        if not top:
            choose1 = 1 if global_mapping == "S1_if_neg" else 0
            score_t = 0.0
        else:
            mu = Xtr[top].apply(pd.to_numeric, errors="coerce").infer_objects().mean(numeric_only=True)
            sd = Xtr[top].apply(pd.to_numeric, errors="coerce").infer_objects().std(numeric_only=True).replace(0, np.nan)
            w = np.array([np.sign([r for c2,r,_ in stats if c2==c][0]) * [abs(r) for c2,r,_ in stats if c2==c][0] for c in top], dtype=float)
            Xt = df.loc[t, top]
            Xt_z = ((Xt - mu) / sd).astype(float).fillna(0.0).values
            score_t = float(np.dot(Xt_z, w))
            if global_mapping == "S1_if_pos":
                choose1 = 1 if score_t > 0 else 0
            else:
                choose1 = 1 if score_t < 0 else 0

        decisions.append({
            "date": df.loc[t+1, args.date_col],
            "score": score_t,
            "mapping": global_mapping,
            "choose_strategy1": choose1,
            "pnl_1_next": pnl1[t+1],
            "pnl_2_next": pnl2[t+1],
            "pnl_adapt_next": pnl1[t+1] if choose1==1 else pnl2[t+1],
            "count_next": counts[t+1]
        })
        if top:
            feat_usage_rows.append({"date": df.loc[t, args.date_col], **{f"used_{c}": 1 for c in top}})

    dec_df = pd.DataFrame(decisions)
    dec_df.to_csv(out_log, index=False)
    fus = pd.DataFrame(feat_usage_rows).fillna(0.0).drop(columns=["date"], errors="ignore").sum().rename("count").to_frame()
    fus.to_csv(out_feats)

    # Build equities aligned with decisions window
    eq_idx_start = look+1
    eq_dates = df[args.date_col].values[eq_idx_start:]
    pnl1_eval = pnl1[eq_idx_start:]
    pnl2_eval = pnl2[eq_idx_start:]
    pnlA_eval = dec_df["pnl_adapt_next"].values
    count_eval = counts[eq_idx_start:]  # same trade count regardless of direction

    # Stats
    def series_stats(label, pnl_series, counts_eval):
        equity = np.cumsum(pnl_series)
        mdd = (np.maximum.accumulate(equity) - equity).max() if len(equity) else 0.0
        sr = (pnl_series.mean() / (pnl_series.std(ddof=1) if pnl_series.std(ddof=1) != 0 else np.nan)) * np.sqrt(252)
        sr = 0.0 if np.isnan(sr) else float(sr)
        total_trades = float(counts_eval.sum())
        total_pnl = float(pnl_series.sum())
        avg_per_trade = float(total_pnl / total_trades) if total_trades > 0 else 0.0
        return {
            "strategy": label,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_per_trade,
            "max_drawdown": float(mdd),
            "sharpe_ratio": sr
        }

    stats_rows = [
        series_stats("Always_1", pnl1_eval, count_eval),
        series_stats("Always_2", pnl2_eval, count_eval),
        series_stats("Adaptive", pnlA_eval, count_eval),
    ]
    pd.DataFrame(stats_rows).to_csv(out_stats, index=False)

    # Plot equity
    equity_plot(eq_dates, np.cumsum(pnl1_eval), np.cumsum(pnl2_eval), np.cumsum(pnlA_eval), out_equity,
                f"Walk-forward top-{args.top_k}, lookback={look} (GLOBAL={global_mapping})")

    log(f"Saved equity: {out_equity}")
    log(f"Saved decisions: {out_log}")
    log(f"Saved feature usage: {out_feats}")
    log(f"Saved stats: {out_stats}")

if __name__ == "__main__":
    main()
