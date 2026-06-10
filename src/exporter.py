import joblib
import numpy as np
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from .config import FEATURE_COLUMNS, MODELS_XGB_FINAL_DIR, MODELS_XGB_DIR

def export_xgb_to_onnx():
    model_path = MODELS_XGB_FINAL_DIR / "xgb_model_latest.pkl"
    model = joblib.load(model_path)

    n_features = len(FEATURE_COLUMNS)
    initial_type = [("input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=13,
    )

    onnx_path = MODELS_XGB_DIR / "xgb_model_latest.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return onnx_path
