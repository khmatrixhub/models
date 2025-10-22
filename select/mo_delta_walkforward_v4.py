#!/usr/bin/env python3
import argparse, os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

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
    X = X.apply(pd.to_numeric, errors="coerce").infer_objects(copy=False)
    mu = X.mean(numeric_only=True)
    sd = X.std(numeric_only=True).replace(0, np.nan)
    return mu, sd

def zapp(X: pd.DataFrame, mu: pd.Series, sd: pd.Series):
    X = X.apply(pd.to_numeric, errors="coerce").infer_objects(copy=False)
    Z = (X - mu) / sd
    # ensure float dtype and fill NaNs
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
        r, p = pearsonr(x.fillna(0.0).values, ytr.values) if x.notna().any() else (0.0, 1.0)
        if not np.isfinite(r): r = 0.0
        stats.append((c, r, abs(r)))
    stats.sort(key=lambda x: x[2], reverse=True)
    top = [c for c, r, ar in stats[:top_k]]
    # weights by sign*|r| from train
    w = np.array([np.sign([r for c2,r,_ in stats if c2==c][0]) * [abs(r) for c2,r,_ in stats if c2==c][0] for c in top], dtype=float)
    return top, w

def main():
    print("running")
    ap = argparse.ArgumentParser(description="Walk-forward with global mapping lock (PnL-calibrated), using MO delta features.")
    ap.add_argument("--mo-file", required=True)
    ap.add_argument("--strategy1-file", required=True)
    ap.add_argument("--date-col", default="trade_date")
    ap.add_argument("--mean-col", default="mean")
    ap.add_argument("--count-col", default="count")
    ap.add_argument("--prefix", default="mo_00")
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--limit-rows", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--lookback-days", type=int, default=120, help="Rolling training window length for selection/weights.")
    ap.add_argument("--calibration-days", type=int, default=None, help="Optional initial days for global mapping calibration; default 2*lookback (bounded by data).")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--out-base", default="mo_walkfwd_v4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_equity = os.path.join(args.out_dir, f"{args.out_base}_equity.png")
    out_log    = os.path.join(args.out_dir, f"{args.out_base}_decisions.csv")
    out_feats  = os.path.join(args.out_dir, f"{args.out_base}_feature_usage.csv")

    # MO daily means -> deltas -> shift by 1 day
    usecols, mo_cols, base_col = detect_mo_columns(args.mo_file, args.date_col, args.prefix)
    mo_daily = aggregate_daily_means(args.mo_file, args.date_col, mo_cols, args.chunksize, args.limit_rows)
    delta_cols = [c for c in mo_daily.columns if c != base_col]
    mo_delta = mo_daily[delta_cols].subtract(mo_daily[base_col], axis=0)
    mo_delta.columns = [c.replace(args.prefix, f"{args.prefix}delta_") for c in mo_delta.columns]
    mo_shift = mo_delta.shift(1).dropna()

    # Strategy 1 daily
    s1 = pd.read_csv(args.strategy1_file, usecols=[args.date_col, args.mean_col, args.count_col])
    s1[args.date_col] = pd.to_datetime(s1[args.date_col], errors='coerce')
    s1.dropna(subset=[args.date_col], inplace=True)
    s1.sort_values(args.date_col, inplace=True)

    df = s1.merge(mo_shift.reset_index(), on=args.date_col, how="inner").dropna().reset_index(drop=True)
    dates = df[args.date_col].values
    y_mean = pd.to_numeric(df[args.mean_col], errors="coerce").fillna(0.0).values
    counts = pd.to_numeric(df[args.count_col], errors="coerce").fillna(0.0).values
    pnl1 = y_mean * counts
    pnl2 = -pnl1

    feat_cols = [c for c in df.columns if c.startswith(f"{args.prefix}delta_")]
    n = len(df)
    look = args.lookback_days
    if n < look + 3:
        raise RuntimeError("Not enough days for requested lookback.")

    # ---------- Global mapping calibration ----------
    calib_days = args.calibration_days or min(n//3, look*2)
    calib_days = max(calib_days, look+5)  # ensure enough decisions
    calib_end = min(n-1, calib_days)

    pnl_adapt_pos_sum = 0.0
    pnl_adapt_neg_sum = 0.0

    print(look, calib_end)
    for t in range(look, calib_end):  # calibrate mapping within this initial span
        train_slice = slice(t-look, t)
        Xtr_full = df.loc[train_slice, feat_cols]
        ytr_full = df.loc[train_slice, args.mean_col]

        top, w = compute_topk_and_weights(Xtr_full, ytr_full, args.top_k)
        mu, sd = zfit(Xtr_full[top])
        Xtrz_full = zapp(Xtr_full[top], mu, sd).values

        # scores for train window, decisions apply next day
        scores_tr = Xtrz_full @ w
        # day t decision uses Xt at t (no look-ahead)
        Xt = df.loc[t, top]
        Xt_z = ((Xt - mu) / sd).astype(float).fillna(0.0).values
        score_t = float(np.dot(Xt_z, w))

        # Evaluate both mappings on the next day t+1
        pnl1_next = pnl1[t+1]; pnl2_next = pnl2[t+1]
        # Option A: S1 if score>0
        choose1_pos = 1 if score_t > 0 else 0
        pnl_adapt_pos_sum += choose1_pos * pnl1_next + (1 - choose1_pos) * pnl2_next
        # Option B: S1 if score<0
        choose1_neg = 1 if score_t < 0 else 0
        pnl_adapt_neg_sum += choose1_neg * pnl1_next + (1 - choose1_neg) * pnl2_next

    global_mapping = "S1_if_pos" if pnl_adapt_pos_sum >= pnl_adapt_neg_sum else "S1_if_neg"

    # ---------- Walk-forward with fixed mapping ----------
    decisions = []
    feat_usage_rows = []

    for t in range(look, n-1):
        train_slice = slice(t-look, t)
        Xtr_full = df.loc[train_slice, feat_cols]
        ytr_full = df.loc[train_slice, args.mean_col]

        top, w = compute_topk_and_weights(Xtr_full, ytr_full, args.top_k)
        mu, sd = zfit(Xtr_full[top])

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
            "pnl_adapt_next": pnl1[t+1] if choose1==1 else pnl2[t+1]
        })
        feat_usage_rows.append({"date": df.loc[t, args.date_col], **{f"used_{c}": 1 for c in top}})

    pnl_adapt = np.array([d["pnl_adapt_next"] for d in decisions])
    eq1 = np.cumsum(pnl1[look+1:])
    eq2 = np.cumsum(pnl2[look+1:])
    eqA = np.cumsum(pnl_adapt)
    use_dates = dates[look+1:]

    dec_df = pd.DataFrame(decisions)
    dec_df.to_csv(out_log, index=False)

    fus = pd.DataFrame(feat_usage_rows).fillna(0.0).drop(columns=["date"], errors="ignore").sum().rename("count").to_frame()
    fus.to_csv(out_feats)

    equity_plot(use_dates, eq1, eq2, eqA, out_equity, 
                f"Walk-forward top-{args.top_k}, lookback={args.lookback_days}d (GLOBAL mapping={global_mapping})")

    print(f"Global mapping chosen on calibration: {global_mapping}")
    print(f"Saved equity plot: {out_equity}")
    print(f"Saved decisions log: {out_log}")
    print(f"Saved feature usage: {out_feats}")

if __name__ == "__main__":
    main()

