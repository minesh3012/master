import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DATA_LABELS_DIR = PROJECT_ROOT / "data" / "labels"

MODELS_XGB_DIR = PROJECT_ROOT / "models" / "xgboost"
MODELS_XGB_FINAL_DIR = MODELS_XGB_DIR / "final"
MODELS_XGB_VERSIONED_DIR = MODELS_XGB_DIR / "versioned"

for d in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_FEATURES_DIR,
    DATA_LABELS_DIR,
    MODELS_XGB_DIR,
    MODELS_XGB_FINAL_DIR,
    MODELS_XGB_VERSIONED_DIR,
]:
    os.makedirs(d, exist_ok=True)

FEATURE_COLUMNS = [
    "Close","Open","High","Low","Body","UpperWick","LowerWick","Range",
    "C1","H1","L1","B1","C2","H2","L2","B2","C3","H3","L3","B3","C4","H4","L4","B4","C5","H5","L5","B5",
    "EMA9","EMA21","EMA50","EMA200","FastSlope","SlopeAccel","TrendUp","TrendDown",
    "DistEMA200","Mom","ADX","ATR","BBWidth",
    "Vol","VolSMA","VolNorm","VolPressure",
    "LocalHigh10","LocalLow10","DistLocalHigh10","DistLocalLow10",
    "SweepUp","SweepDown",
    "IsTradeBar","IsLong","IsShort","BarsSinceEntry"
]

TARGET_COLUMN = "ShortSuccess"
ID_COLUMNS = ["Instrument","StrategyName","TradeId","Phase","BarIndex","Time"]
