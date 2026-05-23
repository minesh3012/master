import pandas as pd
from .config import DATA_FEATURES_DIR, DATA_LABELS_DIR, TARGET_COLUMN

TICK_SIZE = 0.25  # adjust if needed
GOOD_SHORT_TICKS = -8  # <= -8 ticks
BAD_SHORT_TICKS = 5    # >= +5 ticks

def add_future_returns(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    df = df.sort_values(["Time"]).reset_index(drop=True)
    df["FutureClose"] = df["Close"].shift(-horizon)
    df["FutureRet"] = (df["FutureClose"] - df["Close"]) / TICK_SIZE
    return df

def build_labels() -> pd.DataFrame:
    src_path = DATA_FEATURES_DIR / "features.csv"
    df = pd.read_csv(src_path)

    df = add_future_returns(df, horizon=10)

    def label_short(row):
        if pd.isna(row["FutureRet"]):
            return -1  # unknown / ignore
        if row["FutureRet"] <= GOOD_SHORT_TICKS:
            return 1
        if row["FutureRet"] >= BAD_SHORT_TICKS:
            return 0
        return -1

    df[TARGET_COLUMN] = df.apply(label_short, axis=1)
    df = df[df[TARGET_COLUMN] != -1].reset_index(drop=True)

    out_path = DATA_LABELS_DIR / "labeled.csv"
    df.to_csv(out_path, index=False)
    return df
