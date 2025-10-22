#!/usr/bin/env python3
import argparse
from equity_from_decisions import plot_from_decisions, stats_from_decisions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions_csv")
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    s = stats_from_decisions(args.decisions_csv)
    print("== Stats ==")
    for k, v in s.items():
        print(k, v)

    plot_from_decisions(args.decisions_csv, out_png=args.out_png, title=args.title)

if __name__ == "__main__":
    main()

