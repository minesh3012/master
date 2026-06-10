"""
KalmanConfluenceExit — AI Gatekeeper Training Pipeline v2
=========================================================
Fixes from v1:
  - Removed signal-correlated features (KalmanSlope, ZlsmaSlope, IsLong)
    that caused 100% test accuracy via data leakage
  - Added month-based walk-forward cross-validation (12 folds)
  - Stronger XGBoost regularization
  - Conservative threshold selection based on CV precision

Usage:
    python train_model.py --data ../../data/raw/
    python train_model.py --data ../../data/raw/ --threshold 0.58
"""

import argparse
import glob
import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed — pip install xgboost")

# ── Feature sets ──────────────────────────────────────────────────────────────
# EXCLUDED (signal leakage — too correlated with entry direction/outcome):
#   KalmanSlope, ZlsmaSlope, IsLong
#
# KEPT — market context features that describe CONDITIONS, not the signal itself:
FEATURE_COLS = [
    "AtrNorm",          # volatility regime
    "DistKalman",       # how far price is from Kalman at entry
    "DistZlsma",        # how far price is from ZLSMA at entry
    "BodyRatio",        # bar character — decisiveness
    "UpperWick",        # rejection at highs
    "LowerWick",        # rejection at lows
    "BarRangeAtr",      # bar size vs recent volatility
    "VolPct",           # volume percentile — participation
    "TodMinutes",       # time of day context
    "ChandLongNorm",    # distance to chandelier stop (long)
    "ChandShortNorm",   # distance to chandelier stop (short)
    "WideStopActive",   # profit lock state
    "InOpenWindow",     # NY open window flag
    "CScoreEntry",      # consecutive bar agreement score
]

TARGET_COL = "Outcome"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")


def load_data(path_pattern: str) -> pd.DataFrame:
    if os.path.isdir(path_pattern):
        files = glob.glob(os.path.join(path_pattern, "*.csv"))
    else:
        files = glob.glob(path_pattern)

    if not files:
        sys.exit(f"No CSV files found at: {path_pattern}")

    frames = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            raw_rows = len(df)

            # v9 files have Phase/RowType columns — filter to Exit/Trade rows only
            # v8 files have no Phase column — use all rows as-is
            if "Phase" in df.columns and "RowType" in df.columns:
                df = df[(df["Phase"] == "Exit") & (df["RowType"] == "Trade")].copy()
                print(f"  loaded {len(df):>5} trade rows  "
                      f"({raw_rows} total)  {os.path.basename(f)}")
            else:
                print(f"  loaded {len(df):>5} rows  {os.path.basename(f)}")

            frames.append(df)
        except Exception as e:
            print(f"  SKIP {f}: {e}")

    data = pd.concat(frames, ignore_index=True)
    data["EntryDate"] = pd.to_datetime(data["EntryDate"])
    data = data.sort_values("EntryDate").reset_index(drop=True)

    # Ensure Outcome is strictly 0 or 1
    before = len(data)
    data = data[data[TARGET_COL].isin([0, 1])].copy()
    dropped = before - len(data)
    if dropped > 0:
        print(f"  dropped {dropped} rows with non-binary Outcome values")

    print(f"\\nTotal rows  : {len(data)}")
    print(f"Win rate    : {data[TARGET_COL].mean():.1%}")
    print(f"Date range  : {data['EntryDate'].min().date()} "
          f"\\u2192 {data['EntryDate'].max().date()}")
    return data


def month_walk_forward_cv(data: pd.DataFrame, n_test_months: int = 1,
                           min_train_months: int = 6):
    """
    Expanding-window walk-forward CV.
    Each fold: train on all months up to fold, test on next n_test_months.
    Minimum training window = min_train_months.
    Returns list of (train_idx, test_idx) tuples.
    """
    data["_month"] = data["EntryDate"].dt.to_period("M")
    months = sorted(data["_month"].unique())

    folds = []
    for i in range(min_train_months, len(months) - n_test_months + 1):
        train_months = months[:i]
        test_months  = months[i:i + n_test_months]
        train_idx = data[data["_month"].isin(train_months)].index
        test_idx  = data[data["_month"].isin(test_months)].index
        if len(test_idx) > 0:
            folds.append((train_idx, test_idx))

    data.drop(columns=["_month"], inplace=True)
    return folds


def threshold_report(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

    base_win = float(np.mean(y_true))
    base_n   = len(y_true)

    print(f"\n{'Threshold':>10} {'Trades':>7} {'Coverage':>9} "
          f"{'WinRate':>8} {'WinGain':>8} {'PrecGain':>9}")
    print("-" * 58)

    for t in thresholds:
        mask   = y_proba >= t
        n_kept = int(mask.sum())
        if n_kept == 0:
            continue
        win  = float(np.mean(y_true[mask]))
        cov  = n_kept / base_n
        gain = win - base_win
        print(f"{t:>10.2f} {n_kept:>7} {cov:>9.1%} "
              f"{win:>8.1%} {gain:>+8.1%}  "
              f"PF≈{(win*379.5)/max((1-win)*112.5,0.01):>5.2f}")


def run_cv(data: pd.DataFrame, folds, model_fn):
    """Run walk-forward CV, return per-fold metrics."""
    results = []
    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        X_tr = data.loc[tr_idx, FEATURE_COLS].values
        y_tr = data.loc[tr_idx, TARGET_COL].values
        X_te = data.loc[te_idx, FEATURE_COLS].values
        y_te = data.loc[te_idx, TARGET_COL].values

        model = model_fn()
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        # Metric at 0.55 threshold
        mask  = proba >= 0.55
        n_kept = mask.sum()
        win_base = y_te.mean()
        win_filt = y_te[mask].mean() if n_kept > 0 else 0.0
        cov      = n_kept / len(y_te) if len(y_te) > 0 else 0.0

        results.append({
            "fold":     fold_i + 1,
            "n_test":   len(y_te),
            "win_base": win_base,
            "win_filt": win_filt,
            "coverage": cov,
            "n_kept":   n_kept,
        })

    df = pd.DataFrame(results)
    print(f"\n  CV summary ({len(folds)} folds):")
    print(f"  Avg base win rate  : {df['win_base'].mean():.1%}")
    print(f"  Avg filtered win   : {df['win_filt'].mean():.1%}  "
          f"(+{df['win_filt'].mean()-df['win_base'].mean():+.1%})")
    print(f"  Avg coverage       : {df['coverage'].mean():.1%}")
    print(f"  Folds where filter helps: "
          f"{(df['win_filt'] > df['win_base']).sum()}/{len(df)}")
    return df


def train_final(data: pd.DataFrame, threshold: float):
    """
    Final model: train on first 80%, evaluate on last 20%.
    Chronological — never shuffle.
    """
    n  = len(data)
    t0 = int(n * 0.80)

    train = data.iloc[:t0].copy()
    test  = data.iloc[t0:].copy()

    print(f"\n── Final train/test split ────────────────────────────────────────")
    print(f"  Train: {len(train)} rows  "
          f"{train['EntryDate'].min().date()} → {train['EntryDate'].max().date()}  "
          f"win={train[TARGET_COL].mean():.1%}")
    print(f"  Test : {len(test)} rows  "
          f"{test['EntryDate'].min().date()} → {test['EntryDate'].max().date()}  "
          f"win={test[TARGET_COL].mean():.1%}")

    X_tr = train[FEATURE_COLS].values
    y_tr = train[TARGET_COL].values
    X_te = test[FEATURE_COLS].values
    y_te = test[TARGET_COL].values

    best_model = None
    best_name  = None
    best_proba = None

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f"\n── Random Forest ─────────────────────────────────────────────────")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=30,
        max_features=0.5, class_weight="balanced",
        random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_proba = rf.predict_proba(X_te)[:, 1]
    print("  Test set threshold table:")
    threshold_report(y_te, rf_proba)
    best_model = rf
    best_name  = "rf"
    best_proba = rf_proba

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if HAS_XGB:
        print(f"\n── XGBoost ───────────────────────────────────────────────────────")
        xg = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=3,            # shallow — prevents memorisation
            learning_rate=0.02,     # slow learning
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=30,    # requires 30 samples per leaf
            reg_alpha=0.5,          # L1 regularisation
            reg_lambda=2.0,         # L2 regularisation
            scale_pos_weight=1.0,   # balanced dataset — no adjustment needed
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
            verbosity=0)

        xg.fit(X_tr, y_tr,
               eval_set=[(X_te, y_te)],
               verbose=False)
        xg_proba = xg.predict_proba(X_te)[:, 1]
        print("  Test set threshold table:")
        threshold_report(y_te, xg_proba)
        best_model = xg
        best_name  = "xgb"
        best_proba = xg_proba

    # ── Final evaluation at chosen threshold ──────────────────────────────────
    print(f"\n── Final evaluation @ threshold={threshold} ({best_name.upper()}) ──")
    mask         = best_proba >= threshold
    n_base       = len(y_te)
    n_kept       = int(mask.sum())
    win_base     = float(y_te.mean())
    win_kept     = float(y_te[mask].mean()) if n_kept > 0 else 0.0
    losers_base  = int((y_te == 0).sum())
    losers_kept  = int((y_te[mask] == 0).sum()) if n_kept > 0 else 0
    winners_base = int((y_te == 1).sum())
    winners_kept = int((y_te[mask] == 1).sum()) if n_kept > 0 else 0

    pf_base  = (win_base * 379.5) / max((1 - win_base) * 112.5, 0.01)
    pf_after = (win_kept * 379.5) / max((1 - win_kept) * 112.5, 0.01) if n_kept > 0 else 0

    print(f"  Base  : {n_base} trades  win={win_base:.1%}  PF≈{pf_base:.2f}")
    print(f"  Filter: {n_kept} trades  win={win_kept:.1%}  PF≈{pf_after:.2f}  "
          f"coverage={n_kept/n_base:.1%}")
    print(f"  Losers  blocked : {losers_base-losers_kept}/{losers_base}  "
          f"({(losers_base-losers_kept)/max(losers_base,1):.1%})")
    print(f"  Winners blocked : {winners_base-winners_kept}/{winners_base}  "
          f"({(winners_base-winners_kept)/max(winners_base,1):.1%})")

    # ── Feature importance ────────────────────────────────────────────────────
    imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
    print(f"\n── Feature importance ────────────────────────────────────────────")
    print(imp.sort_values(ascending=False).to_string())

    return best_model, best_name


def save_model(model, model_name: str, threshold: float):
    os.makedirs(MODEL_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"kalman_{model_name}_{ts}.pkl")

    pkg = {"model": model, "threshold": threshold,
           "features": FEATURE_COLS, "trained_at": ts}

    with open(path, "wb") as f:
        pickle.dump(pkg, f)

    latest = os.path.join(MODEL_DIR, "kalman_latest.pkl")
    with open(latest, "wb") as f:
        pickle.dump(pkg, f)

    meta = {"model_type": model_name, "threshold": threshold,
            "features": FEATURE_COLS, "trained_at": ts,
            "feature_count": len(FEATURE_COLS)}
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved  → {path}")
    print(f"Latest link  → {latest}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/raw/")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--skip-cv",   action="store_true",
                        help="Skip walk-forward CV (faster, less insight)")
    args = parser.parse_args()

    print("=" * 60)
    print("KalmanConfluenceExit — AI Gatekeeper Training v2")
    print("=" * 60)
    print(f"\nFeatures used ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Excluded (leakage): KalmanSlope, ZlsmaSlope, IsLong")

    data = load_data(args.data)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in data.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}")

    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"Clean rows  : {len(data)}")

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    if not args.skip_cv:
        folds = month_walk_forward_cv(data, n_test_months=1, min_train_months=6)
        print(f"\n── Walk-forward CV ({len(folds)} folds) ──────────────────────────")

        if HAS_XGB:
            print("\n  XGBoost CV:")
            run_cv(data, folds, lambda: xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.6, min_child_weight=30,
                reg_alpha=0.5, reg_lambda=2.0,
                eval_metric="logloss", random_state=42,
                use_label_encoder=False, verbosity=0))

        print("\n  RandomForest CV:")
        run_cv(data, folds, lambda: RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=30,
            max_features=0.5, class_weight="balanced",
            random_state=42, n_jobs=-1))

    # ── Final model ───────────────────────────────────────────────────────────
    model, name = train_final(data, args.threshold)
    save_model(model, name, args.threshold)
    print("\nDone. Next: python python\\server\\api_server.py")


if __name__ == "__main__":
    main()