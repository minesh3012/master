import sys
from pathlib import Path

# ---------------------------------------------------------
# 1. Add project root to PYTHONPATH BEFORE importing src
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------
# 2. Now import from src
# ---------------------------------------------------------
from src.data_loader import load_raw_csvs

# ---------------------------------------------------------
# 3. Run merge
# ---------------------------------------------------------
if __name__ == "__main__":
    load_raw_csvs()
