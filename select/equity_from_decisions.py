"""
Reusable equity plotting & stats from decisions.csv files.

Required columns in the CSV:
- date
- choose_strategy1   (0/1)
- pnl_1_next
- pnl_2_next

Public functions:
- plot_from_decisions(decisions_csv, out_png=None, title=None)
- stats_from_decisions(decisions_csv) -> dict
- plot_multi(files, labels=None, out_png=None, title=None)

All plots use matplotlib (no seaborn). Equities are rebuilt
from the PnL columns so they’re consistent across scripts.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

def _load_decisions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("decisions csv must have a 'date' column")
    for col in ("pnl_1_next", "pnl_2_next", "choose_strategy1"):
        if col not in df.columns:
            raise ValueError(f"decisions csv missing required column: '{col}'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def _equities_from_df(df: pd.DataFrame):
    p1 = pd.to_numeric(df["pnl_1_next"], errors="coerce").fillna(0.0).values
    p2 = pd.to_numeric(df["pnl_2_next"], errors="coerce").fillna(0.0).values
    c  = pd.to_numeric(df["choose_strategy1"], errors="coerce").fillna(0.0).astype(int).values
    e1 = np.cumsum(p1)
    e2 = np.cumsum(p2)
    ea = np.cumsum(c*p1 + (1-c)*p2)
    return df["date"].values, e1, e2, ea, p1, p2, (c*p1 + (1-c)*p2)

def _stats_from_pnl(pnl: np.ndarray) -> Dict[str, float]:
    eq = np.cumsum(pnl)
    peaks = np.maximum.accumulate(eq) if len(eq) else np.array([])
    mdd = float((peaks - eq).max()) if len(eq) else 0.0
    sr = float(pnl.mean()/pnl.std(ddof=1)*np.sqrt(252)) if len(pnl)>1 and pnl.std(ddof=1)>0 else 0.0
    return {
        "total_pnl": float(pnl.sum() if len(pnl) else 0.0),
        "avg_pnl": float(pnl.mean() if len(pnl) else 0.0),
        "max_drawdown": mdd,
        "sharpe": sr,
        "n_days": int(len(pnl)),
    }

def stats_from_decisions(decisions_csv: str) -> Dict[str, Dict[str, float]]:
    df = _load_decisions(decisions_csv)
    _, _, _, _, p1, p2, pa = _equities_from_df(df)
    return {
        "Always_1": _stats_from_pnl(p1),
        "Always_2": _stats_from_pnl(p2),
        "Adaptive": _stats_from_pnl(pa),
    }

def plot_from_decisions(decisions_csv: str, out_png: Optional[str]=None, title: Optional[str]=None):
    df = _load_decisions(decisions_csv)
    dates, e1, e2, ea, *_ = _equities_from_df(df)
    plt.figure(figsize=(12,6))
    plt.plot(dates, e1, label="Always 1")
    plt.plot(dates, e2, label="Always 2")
    plt.plot(dates, ea, label="Adaptive", linewidth=2)
    plt.grid(True); plt.legend(); plt.title(title or "Equity from decisions.csv"); plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150); plt.close()
    else:
        plt.show()

def plot_multi(files: List[str], labels: Optional[List[str]]=None, out_png: Optional[str]=None, title: Optional[str]=None):
    if labels is None: labels = [f"run{i+1}" for i in range(len(files))]
    assert len(labels) == len(files), "labels and files must be same length"
    plt.figure(figsize=(12,6))
    for f, lab in zip(files, labels):
        df = _load_decisions(f)
        dates, *_ , ea_p1, ea_p2, ea = df["date"].values, None, None, None, None, None, None
        # Recompute only adaptive equity for overlay
        _, _, _, ea_curve, *_ = _equities_from_df(df)
        plt.plot(df["date"].values, ea_curve, label=lab)
    plt.grid(True); plt.legend(); plt.title(title or "Adaptive equity comparison"); plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150); plt.close()
    else:
        plt.show()


