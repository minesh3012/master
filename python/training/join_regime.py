"""
join_regime.py — Merge regime context onto trade rows
======================================================
Reads all trade CSVs and all regime CSVs from data/raw/,
joins them on timestamp (nearest regime bar <= trade entry time),
writes merged file to data/processed/trades_with_regime.csv

Usage:
    python join_regime.py
    python join_regime.py --raw data\raw\ --out data\processed\

The 5 regime features added to each trade row:
    MTF_Align        — 5-min and 15-min trend agreement (-1/0/1)
    VolRegime        — volatility state (0=compressed 1=normal 2=expanded)
    BBSqueeze        — Bollinger compression flag (0/1)
    VWAPDistNorm     — (close - VWAP) / ATR at regime bar time
    RegimeFlipUp     — regime just flipped bullish (0/1)
    RegimeFlipDown   — regime just flipped bearish (0/1)
    KalmanTrend      — Kalman direction at regime bar (-1/0/1)
    SessionType      — 0=pre-market 1=NY open 2=mid-session
    BiasLong         — composite long bias score
    BiasShort        — composite short bias score
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TRADE_COLS_REQUIRED = ["EntryDate", "EntryTime", "Outcome"]
REGIME_COLS_REQUIRED = [
    "Time", "MTF_Align", "VolRegime", "BBSqueeze",
    "VWAPDistNorm", "RegimeFlipUp", "RegimeFlipDown",
    "KalmanTrend", "SessionType", "BiasLong", "BiasShort"
]

# These are the regime features we attach to each trade
REGIME_FEATURES = [
    "MTF_Align", "VolRegime", "BBSqueeze",
    "VWAPDistNorm", "VWAPAbove", "VWAPSlope",
    "KalmanDistNorm", "KalmanSlope", "KalmanTrend",
    "DistORH", "DistORL", "AboveORH", "BelowORL",
    "StructPos", "BOSUp", "BOSDown",
    "TrendDir_5m", "TrendStrength_5m",
    "TrendDir_15m", "TrendStrength_15m",
    "SessionType", "TOD_Sin", "TOD_Cos",
    "BiasLong", "BiasShort", "BiasSlope",
    "RegimeFlipUp", "RegimeFlipDown", "RegimeStable"
]


def load_trade_files(raw_folder: str) -> pd.DataFrame:
    """Load all trade CSVs — both v8 (no Phase col) and v9 (has Phase col)."""
    files = glob.glob(os.path.join(raw_folder, "*.csv"))
    regime_files = {f for f in files if "RegimeMap" in os.path.basename(f)}
    trade_files  = [f for f in files if f not in regime_files]

    if not trade_files:
        sys.exit(f"No trade CSV files found in {raw_folder}")

    frames = []
    for f in sorted(trade_files):
        try:
            df = pd.read_csv(f)

            # v9 files: keep only Exit/Trade rows
            if "Phase" in df.columns and "RowType" in df.columns:
                df = df[(df["Phase"] == "Exit") & (df["RowType"] == "Trade")].copy()

            # Must have outcome column
            if "Outcome" not in df.columns:
                print(f"  SKIP (no Outcome column): {os.path.basename(f)}")
                continue

            # Keep only rows with valid outcomes
            df = df[df["Outcome"].isin([0, 1])].copy()

            if len(df) == 0:
                print(f"  SKIP (no valid rows): {os.path.basename(f)}")
                continue

            frames.append(df)
            print(f"  trade  {len(df):>5} rows  {os.path.basename(f)}")
        except Exception as e:
            print(f"  SKIP {f}: {e}")

    if not frames:
        sys.exit("No valid trade rows found")

    data = pd.concat(frames, ignore_index=True)

    # Parse entry timestamp
    data["EntryDate"] = pd.to_datetime(data["EntryDate"])
    data["_ts"] = pd.to_datetime(
        data["EntryDate"].dt.strftime("%Y-%m-%d") + " " + data["EntryTime"]
    )
    data = data.sort_values("_ts").reset_index(drop=True)

    # Tag instrument for regime matching
    if "Instrument" not in data.columns:
        data["Instrument"] = "Unknown"

    print(f"\n  Total trade rows : {len(data)}")
    print(f"  Win rate         : {data['Outcome'].mean():.1%}")
    print(f"  Date range       : {data['_ts'].min().date()} -> {data['_ts'].max().date()}")
    return data


def load_regime_files(raw_folder: str) -> dict:
    """
    Load all regime CSVs. Returns dict keyed by instrument prefix.
    e.g. {'ES': DataFrame, 'NQ': DataFrame}
    """
    files = glob.glob(os.path.join(raw_folder, "*RegimeMap*.csv"))

    if not files:
        sys.exit(f"No regime CSV files found in {raw_folder}\n"
                 f"Run RegimeMapBuilder backtest first.")

    regimes = {}
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            df["_ts"] = pd.to_datetime(df["Time"])
            df = df.sort_values("_ts").reset_index(drop=True)

            # Determine which instrument this belongs to
            basename = os.path.basename(f)
            if basename.startswith("ES"):
                key = "ES"
            elif basename.startswith("NQ"):
                key = "NQ"
            else:
                key = basename.split("_")[0]

            # Keep only the regime features that exist in this file
            available = [c for c in REGIME_FEATURES if c in df.columns]
            missing   = [c for c in REGIME_FEATURES if c not in df.columns]
            if missing:
                print(f"  NOTE: {basename} missing regime cols: {missing}")

            regimes[key] = df[["_ts"] + available].copy()
            print(f"  regime {len(df):>6} rows  {basename}  [{key}]  {len(available)} features")

        except Exception as e:
            print(f"  SKIP {f}: {e}")

    return regimes


def match_instrument(instrument_str: str) -> str:
    """Extract ES or NQ from instrument full name."""
    s = str(instrument_str).upper()
    if "NQ" in s: return "NQ"
    if "ES" in s: return "ES"
    return s.split()[0] if " " in s else s


def join_regime(trades: pd.DataFrame, regimes: dict) -> pd.DataFrame:
    """
    For each trade, find the most recent regime bar at or before entry time.
    Uses pandas merge_asof for efficient sorted merge.
    """
    result_frames = []

    for inst_key, group in trades.groupby(
            trades["Instrument"].apply(match_instrument)):

        group = group.copy()

        if inst_key not in regimes:
            print(f"  WARNING: no regime data for {inst_key} "
                  f"— regime features will be NaN for {len(group)} rows")
            result_frames.append(group)
            continue

        reg = regimes[inst_key]

        # merge_asof: for each trade timestamp, find nearest regime bar <= ts
        merged = pd.merge_asof(
            group.sort_values("_ts"),
            reg.rename(columns={"_ts": "_regime_ts"}),
            left_on="_ts",
            right_on="_regime_ts",
            direction="backward",   # most recent regime bar at or before trade
            tolerance=pd.Timedelta("60min")  # skip if no regime bar within 60 min
        )

        # How many trades got a regime match
        matched_cols = [c for c in REGIME_FEATURES if c in merged.columns]
        n_matched = merged[matched_cols[0]].notna().sum() if matched_cols else 0
        print(f"  {inst_key}: {len(group)} trades -> "
              f"{n_matched} matched to regime ({n_matched/len(group):.1%})")

        result_frames.append(merged)

    return pd.concat(result_frames, ignore_index=True).sort_values("_ts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/",
                        help="Folder containing trade and regime CSVs")
    parser.add_argument("--out", default="data/processed/",
                        help="Output folder for merged file")
    args = parser.parse_args()

    print("=" * 60)
    print("Regime Join — merging market context onto trade rows")
    print("=" * 60)

    print("\nLoading trade files...")
    trades = load_trade_files(args.raw)

    print("\nLoading regime files...")
    regimes = load_regime_files(args.raw)

    print("\nJoining regime context to trades...")
    merged = join_regime(trades, regimes)

    # Summary of regime feature coverage
    regime_cols_present = [c for c in REGIME_FEATURES if c in merged.columns]
    print(f"\nRegime features attached: {len(regime_cols_present)}")
    print(f"  {regime_cols_present}")

    # Drop helper columns
    merged = merged.drop(columns=["_ts", "_regime_ts"],
                         errors="ignore")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "trades_with_regime.csv")
    merged.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"Rows  : {len(merged)}")
    print(f"Cols  : {len(merged.columns)}")

    # Quick sanity check
    if "MTF_Align" in merged.columns:
        print("\n── MTF_Align win rate breakdown ──────────────────────────")
        for val, grp in merged.groupby("MTF_Align"):
            label = {-1: "bearish", 0: "neutral", 1: "bullish"}.get(val, str(val))
            print(f"  MTF_Align={val:+d} ({label:7s}): "
                  f"{len(grp):>4} trades  win={grp['Outcome'].mean():.1%}")

    if "VolRegime" in merged.columns:
        print("\n── VolRegime win rate breakdown ──────────────────────────")
        for val, grp in merged.groupby("VolRegime"):
            label = {0: "compressed", 1: "normal", 2: "expanded"}.get(val, str(val))
            print(f"  VolRegime={val} ({label:10s}): "
                  f"{len(grp):>4} trades  win={grp['Outcome'].mean():.1%}")

    if "BBSqueeze" in merged.columns:
        print("\n── BBSqueeze win rate breakdown ──────────────────────────")
        for val, grp in merged.groupby("BBSqueeze"):
            label = "squeeze" if val == 1 else "normal"
            print(f"  BBSqueeze={val} ({label:7s}): "
                  f"{len(grp):>4} trades  win={grp['Outcome'].mean():.1%}")

    if "SessionType" in merged.columns:
        print("\n── SessionType win rate breakdown ────────────────────────")
        for val, grp in merged.groupby("SessionType"):
            label = {0: "pre-market", 1: "NY open", 2: "mid-session"}.get(val, str(val))
            print(f"  SessionType={val} ({label:11s}): "
                  f"{len(grp):>4} trades  win={grp['Outcome'].mean():.1%}")

    print("\nNext: python train_model.py --data data/processed/trades_with_regime.csv")


if __name__ == "__main__":
    main()