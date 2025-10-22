# equity_from_decisions.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple

REQUIRED = ("date", "choose_strategy1", "pnl_1_next", "pnl_2_next",
            "count_1_next", "count_2_next")  # <-- counts required

def _load_decisions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"decisions csv missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def _extract_arrays(df: pd.DataFrame):
    p1 = pd.to_numeric(df["pnl_1_next"], errors="coerce").fillna(0.0).to_numpy()
    p2 = pd.to_numeric(df["pnl_2_next"], errors="coerce").fillna(0.0).to_numpy()
    c  = pd.to_numeric(df["choose_strategy1"], errors="coerce").fillna(0.0).astype(int).to_numpy()
    n1 = pd.to_numeric(df["count_1_next"], errors="coerce").fillna(0.0).to_numpy()
    n2 = pd.to_numeric(df["count_2_next"], errors="coerce").fillna(0.0).to_numpy()
    pa = c * p1 + (1 - c) * p2
    na = c * n1 + (1 - c) * n2
    return p1, p2, pa, c, n1, n2, na

def _max_drawdown_from_pnl(pnl: np.ndarray) -> float:
    eq = np.cumsum(pnl)
    if len(eq) == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    return float((peaks - eq).max())

def _sharpe(pnl: np.ndarray, periods_per_year: int = 252) -> float:
    if len(pnl) < 2:
        return 0.0
    sd = pnl.std(ddof=1)
    return float(pnl.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0

def _per_trade_stats(total_pnl: float, total_trades: float, fee_per_trade: float):
    gross_avg = float(total_pnl / total_trades) if total_trades > 0 else 0.0
    net_total = float(total_pnl - fee_per_trade * total_trades)
    net_avg   = float(net_total / total_trades) if total_trades > 0 else 0.0
    return gross_avg, net_total, net_avg

def stats_from_decisions(
    decisions_csv: str,
    fee_per_trade: float = 0.0,   # round-turn fee per filled trade
    include_daily_metrics: bool = True
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      total_trades, gross_total_pnl, gross_avg_pnl_per_trade,
      net_total_pnl, net_avg_pnl_per_trade,
      (optional) daily_sharpe, daily_max_drawdown  -- computed on daily PnL streams
    """
    df = _load_decisions(decisions_csv)
    p1, p2, pa, c, n1, n2, na = _extract_arrays(df)

    def _row(pnl, ntr):
        total_trades = float(ntr.sum())
        gross_total = float(pnl.sum())
        gross_avg, net_total, net_avg = _per_trade_stats(gross_total, total_trades, fee_per_trade)
        out = {
            "total_trades": total_trades,
            "gross_total_pnl": gross_total,
            "gross_avg_pnl_per_trade": gross_avg,
            "net_total_pnl": net_total,
            "net_avg_pnl_per_trade": net_avg,
        }
        if include_daily_metrics:
            out.update({
                "daily_sharpe": _sharpe(pnl),
                "daily_max_drawdown": _max_drawdown_from_pnl(pnl),
            })
        return out

    rows = {
        "Always_1": _row(p1, n1),
        "Always_2": _row(p2, n2),
        "Adaptive": _row(pa, na),
    }
    return pd.DataFrame.from_dict(rows, orient="index")

def write_stats(
    decisions_csv: str,
    out_csv: str,
    fee_per_trade: float = 0.0,
    include_daily_metrics: bool = True
) -> None:
    stats_from_decisions(decisions_csv, fee_per_trade, include_daily_metrics).to_csv(out_csv)

def plot_from_decisions(decisions_csv: str, out_png: Optional[str] = None, title: Optional[str] = None) -> None:
    df = _load_decisions(decisions_csv)
    p1, p2, pa, *_ = _extract_arrays(df)
    dates = df["date"].to_numpy()
    eq1, eq2, eqA = np.cumsum(p1), np.cumsum(p2), np.cumsum(pa)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, eq1, label="Always 1")
    plt.plot(dates, eq2, label="Always 2")
    plt.plot(dates, eqA, label="Adaptive", linewidth=2)
    plt.grid(True); plt.legend(); plt.title(title or "Equity from decisions.csv"); plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150); plt.close()
    else:
        plt.show()

def plot_multi(files: List[str], labels: Optional[List[str]] = None, out_png: Optional[str] = None, title: Optional[str] = None) -> None:
    if labels is None:
        labels = [f"run{i+1}" for i in range(len(files))]
    assert len(labels) == len(files), "labels and files must be same length"
    plt.figure(figsize=(12, 6))
    for f, lab in zip(files, labels):
        df = _load_decisions(f)
        _, _, pa, *_ = _extract_arrays(df)
        plt.plot(df["date"].to_numpy(), np.cumsum(pa), label=lab)
    plt.grid(True); plt.legend(); plt.title(title or "Adaptive equity comparison"); plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150); plt.close()
    else:
        plt.show()

