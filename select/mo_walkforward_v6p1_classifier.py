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

def equity_plot(dates, curves: dict, out_png, title):
    plt.figure(figsize=(10,6))
    for label, pnl in curves.items():
        plt.plot(dates, np.cumsum(pnl), label=label, linewidth=2 if label=="Adaptive" else 1)
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

def max_drawdown_from_pnl(pnl: np.ndarray):
    eq = np.cumsum(pnl)
    peaks = np.maximum.accumulate(eq)
    return float(np.max(peaks - eq)) if len(eq) else 0.0

def sharpe_ratio(pnl: np.ndarray, periods_per_year: int = 252):
    if len(pnl) == 0:
        return 0.0
    sd = pnl.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(pnl.mean() / sd * np.sqrt(periods_per_year))

def stats_row(label: str, pnl: np.ndarray, counts: np.ndarray):
    total_trades = float(np.nansum(counts))
    total_pnl = float(np.nansum(pnl))
    avg_per_trade = float(total_pnl / total_trades) if total_trades > 0 else 0.0
    mdd = max_drawdown_from_pnl(pnl)
    sr = sharpe_ratio(pnl)
    return {"strategy": label, "total_trades": total_trades, "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_per_trade, "max_drawdown": mdd, "sharpe_ratio": sr}

def main():
    ap = argparse.ArgumentParser(description="Walk-forward v6.1: supports different signal counts (prob>0.5 already filtered), outer-joins S1/S2.")
    ap.add_argument("--mo-file", required=True)
    ap.add_argument("--strategy1-file", required=True)
    ap.add_argument("--strategy2-file", default=None, help="Optional Strategy 2 file; if omitted, Strategy 2 = inverse of Strategy 1 (same counts).")
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
    ap.add_argument("--out-base", default="wf_v6p1")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_equity = os.path.join(args.out_dir, f"{args.out_base}_equity.png")
    out_decisions = os.path.join(args.out_dir, f"{args.out_base}_decisions.csv")
    out_feats  = os.path.join(args.out_dir, f"{args.out_base}_feature_usage.csv")
    out_stats  = os.path.join(args.out_dir, f"{args.out_base}_stats.csv")

    # --- Load MOs (delta shift) ---
    usecols, mo_cols, base_col = detect_mo_columns(args.mo_file, args.date_col, args.prefix)
    mo_daily = aggregate_daily_means(args.mo_file, args.date_col, mo_cols, args.chunksize, args.limit_rows)
    delta_cols = [c for c in mo_daily.columns if c != base_col]
    mo_delta = mo_daily[delta_cols].subtract(mo_daily[base_col], axis=0)
    mo_delta.columns = [c.replace(args.prefix, f"{args.prefix}delta_") for c in mo_delta.columns]
    mo_shift = mo_delta.shift(1)

    # --- Load strategies ---
    s1 = pd.read_csv(args.strategy1_file, usecols=[args.date_col, args.mean_col, args.count_col])
    s1[args.date_col] = pd.to_datetime(s1[args.date_col], errors='coerce')
    s1.dropna(subset=[args.date_col], inplace=True)
    s1.sort_values(args.date_col, inplace=True)
    s1.rename(columns={args.mean_col: "s1_mean", args.count_col: "s1_count"}, inplace=True)

    if args.strategy2_file:
        s2 = pd.read_csv(args.strategy2_file, usecols=[args.date_col, args.mean_col, args.count_col])
        s2[args.date_col] = pd.to_datetime(s2[args.date_col], errors='coerce')
        s2.dropna(subset=[args.date_col], inplace=True)
        s2.sort_values(args.date_col, inplace=True)
        s2.rename(columns={args.mean_col: "s2_mean", args.count_col: "s2_count"}, inplace=True)
        base = pd.merge(s1, s2, on=args.date_col, how="outer")
    else:
        base = s1.copy()
        base["s2_mean"] = -base["s1_mean"]
        base["s2_count"] = base["s1_count"]

    # Fill missing means/counts with 0 so PnL=0 on missing days
    for c in ["s1_mean","s2_mean","s1_count","s2_count"]:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    # Merge with MOs (inner on dates present in MO features)
    df = pd.merge(base, mo_shift.reset_index(), on=args.date_col, how="inner").sort_values(args.date_col)
    df.dropna(subset=[args.date_col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise RuntimeError("Merged dataset is empty after alignment. Check dates/shift.")
    dates = df[args.date_col].values

    s1_mean = df["s1_mean"].values
    s2_mean = df["s2_mean"].values
    s1_count = df["s1_count"].values
    s2_count = df["s2_count"].values
    pnl1 = s1_mean * s1_count
    pnl2 = s2_mean * s2_count

    feat_cols = [c for c in df.columns if c.startswith(f"{args.prefix}delta_")]
    n = len(df); look = args.lookback_days
    if len(feat_cols) == 0:
        raise RuntimeError("No delta features found after processing.")
    if n < look + 3:
        look = max(20, n//4)

    # --- Calibration (global mapping; uses true pnl1/pnl2) ---
    calib_days = args.calibration_days or min(n//3, look*2)
    calib_days = max(calib_days, look+5)
    calib_end = min(n-2, calib_days)

    pnl_pos_sum = 0.0
    pnl_neg_sum = 0.0
    for t in range(look, calib_end):
        Xtr = df.loc[t-look:t-1, feat_cols]
        ytr = df.loc[t-look:t-1, "s1_mean"]
        # top-k & weights
        stats = []
        for c in Xtr.columns:
            x = pd.to_numeric(Xtr[c], errors="coerce")
            r = 0.0 if x.notna().sum() < 2 else pearsonr(x.fillna(0.0).values, ytr.values)[0]
            r = 0.0 if not np.isfinite(r) else r
            stats.append((c, r, abs(r)))
        stats.sort(key=lambda x: x[2], reverse=True)
        top = [c for c, r, ar in stats[:args.top_k]]
        if not top: 
            continue
        mu = Xtr[top].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)
        sd = Xtr[top].apply(pd.to_numeric, errors="coerce").std(numeric_only=True).replace(0, np.nan)
        w = np.array([np.sign([r for c2,r,_ in stats if c2==c][0]) * [abs(r) for c2,r,_ in stats if c2==c][0] for c in top], dtype=float)
        Xt = df.loc[t, top]
        Xt_z = ((Xt - mu) / sd).astype(float).fillna(0.0).values
        score_t = float(np.dot(Xt_z, w))
        pnl1_next, pnl2_next = pnl1[t+1], pnl2[t+1]
        pnl_pos_sum += (pnl1_next if score_t > 0 else pnl2_next)
        pnl_neg_sum += (pnl1_next if score_t < 0 else pnl2_next)

    global_mapping = "S1_if_neg" if pnl_pos_sum >= pnl_neg_sum else "S1_if_pos"

    # --- Walk-forward decisions ---
    decisions = []; feat_usage_rows = []; choose1_list = []
    for t in range(look, n-1):
        Xtr = df.loc[t-look:t-1, feat_cols]
        ytr = df.loc[t-look:t-1, "s1_mean"]
        stats = []
        for c in Xtr.columns:
            x = pd.to_numeric(Xtr[c], errors="coerce")
            r = 0.0 if x.notna().sum() < 2 else pearsonr(x.fillna(0.0).values, ytr.values)[0]
            r = 0.0 if not np.isfinite(r) else r
            stats.append((c, r, abs(r)))
        stats.sort(key=lambda x: x[2], reverse=True)
        top = [c for c, r, ar in stats[:args.top_k]]
        if not top:
            score_t = 0.0
            choose1 = 1 if global_mapping == "S1_if_neg" else 0
        else:
            mu = Xtr[top].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)
            sd = Xtr[top].apply(pd.to_numeric, errors="coerce").std(numeric_only=True).replace(0, np.nan)
            w = np.array([np.sign([r for c2,r,_ in stats if c2==c][0]) * [abs(r) for c2,r,_ in stats if c2==c][0] for c in top], dtype=float)
            Xt = df.loc[t, top]
            Xt_z = ((Xt - mu) / sd).astype(float).fillna(0.0).values
            score_t = float(np.dot(Xt_z, w))
            if global_mapping == "S1_if_pos":
                choose1 = 1 if score_t > 0 else 0
            else:
                choose1 = 1 if score_t < 0 else 0
        choose1_list.append(choose1)
        decisions.append({
            "date": df.loc[t+1, args.date_col],
            "score": score_t,
            "mapping": global_mapping,
            "choose_strategy1": choose1,
            "pnl_1_next": pnl1[t+1],
            "pnl_2_next": pnl2[t+1],
            "count_1_next": s1_count[t+1],
            "count_2_next": s2_count[t+1],
        })
        if top:
            feat_usage_rows.append({"date": df.loc[t, args.date_col], **{f"used_{c}": 1 for c in top}})

    dec_df = pd.DataFrame(decisions)

    # Align eval arrays (decisions apply to t+1)
    idx_start = look+1
    dates_eval = df[args.date_col].values[idx_start:]
    pnl1_eval = pnl1[idx_start:]
    pnl2_eval = pnl2[idx_start:]
    c1_eval = s1_count[idx_start:]
    c2_eval = s2_count[idx_start:]
    choose1_arr = np.array(choose1_list[1:])  # shift to align with eval window
    min_len = min(len(dates_eval), len(pnl1_eval), len(pnl2_eval), len(c1_eval), len(c2_eval), len(choose1_arr))
    dates_eval = dates_eval[:min_len]
    pnl1_eval = pnl1_eval[:min_len]
    pnl2_eval = pnl2_eval[:min_len]
    c1_eval = c1_eval[:min_len]
    c2_eval = c2_eval[:min_len]
    choose1_arr = choose1_arr[:min_len]

    adaptive_pnl = choose1_arr * pnl1_eval + (1 - choose1_arr) * pnl2_eval
    adaptive_counts = choose1_arr * c1_eval + (1 - choose1_arr) * c2_eval

    # Stats
    curves = {"Always 1": pnl1_eval, "Always 2": pnl2_eval, "Adaptive": adaptive_pnl}
    stats_rows = [
        {"strategy":"Always_1", "total_trades": float(np.nansum(c1_eval)),
         "total_pnl": float(np.nansum(pnl1_eval)),
         "avg_pnl_per_trade": float(np.nansum(pnl1_eval)/np.nansum(c1_eval) if np.nansum(c1_eval)>0 else 0.0),
         "max_drawdown": float(np.max(np.maximum.accumulate(np.cumsum(pnl1_eval)) - np.cumsum(pnl1_eval))) if len(pnl1_eval) else 0.0,
         "sharpe_ratio": float((pnl1_eval.mean()/pnl1_eval.std(ddof=1))*np.sqrt(252)) if len(pnl1_eval)>1 and pnl1_eval.std(ddof=1)>0 else 0.0},
        {"strategy":"Always_2", "total_trades": float(np.nansum(c2_eval)),
         "total_pnl": float(np.nansum(pnl2_eval)),
         "avg_pnl_per_trade": float(np.nansum(pnl2_eval)/np.nansum(c2_eval) if np.nansum(c2_eval)>0 else 0.0),
         "max_drawdown": float(np.max(np.maximum.accumulate(np.cumsum(pnl2_eval)) - np.cumsum(pnl2_eval))) if len(pnl2_eval) else 0.0,
         "sharpe_ratio": float((pnl2_eval.mean()/pnl2_eval.std(ddof=1))*np.sqrt(252)) if len(pnl2_eval)>1 and pnl2_eval.std(ddof=1)>0 else 0.0},
        {"strategy":"Adaptive", "total_trades": float(np.nansum(adaptive_counts)),
         "total_pnl": float(np.nansum(adaptive_pnl)),
         "avg_pnl_per_trade": float(np.nansum(adaptive_pnl)/np.nansum(adaptive_counts) if np.nansum(adaptive_counts)>0 else 0.0),
         "max_drawdown": float(np.max(np.maximum.accumulate(np.cumsum(adaptive_pnl)) - np.cumsum(adaptive_pnl))) if len(adaptive_pnl) else 0.0,
         "sharpe_ratio": float((adaptive_pnl.mean()/adaptive_pnl.std(ddof=1))*np.sqrt(252)) if len(adaptive_pnl)>1 and adaptive_pnl.std(ddof=1)>0 else 0.0},
    ]

    # Save
    dec_df.to_csv(out_decisions, index=False)
    fus = pd.DataFrame(feat_usage_rows).fillna(0.0).drop(columns=["date"], errors="ignore").sum().rename("count").to_frame()
    fus.to_csv(out_feats)
    pd.DataFrame(stats_rows).to_csv(out_stats, index=False)
    equity_plot(dates_eval, curves, out_equity, f"Walk-forward top-{args.top_k}, lookback={look} (GLOBAL mapping={global_mapping})")

    log(f"Saved equity: {out_equity}")
    log(f"Saved decisions: {out_decisions}")
    log(f"Saved feature usage: {out_feats}")
    log(f"Saved stats: {out_stats}")

if __name__ == "__main__":
    main()
