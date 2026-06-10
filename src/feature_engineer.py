import pandas as pd
from .config import DATA_PROCESSED_DIR, DATA_FEATURES_DIR, FEATURE_COLUMNS

def build_features() -> pd.DataFrame:
    src_path = DATA_PROCESSED_DIR / "merged_raw.csv"
    df = pd.read_csv(src_path)

    # basic cleaning
    df = df.dropna(subset=["Close","Open","High","Low"])

    # ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    feat_df = df.copy()
    out_path = DATA_FEATURES_DIR / "features.csv"
    feat_df.to_csv(out_path, index=False)
    return feat_df
