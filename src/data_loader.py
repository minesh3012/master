import pandas as pd
from pathlib import Path
from .config import DATA_RAW_DIR, DATA_PROCESSED_DIR

def load_raw_csvs() -> pd.DataFrame:
    files = list(Path(DATA_RAW_DIR).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {DATA_RAW_DIR}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    out_path = DATA_PROCESSED_DIR / "merged_raw.csv"
    merged.to_csv(out_path, index=False)
    return merged
