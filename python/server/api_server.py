"""
KalmanConfluenceExit — AI Gatekeeper FastAPI Server

Startup:
    pip install fastapi uvicorn scikit-learn xgboost numpy
    python api_server.py

Endpoints:
    POST /predict   — main inference endpoint called by NinjaTrader
    GET  /health    — model status and metadata
    POST /reload    — hot-reload model from disk without restart
"""

import os
import pickle
import time
import json
import logging
from datetime import datetime
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "kalman_latest.pkl")

FEATURE_NAMES = [
    "KalmanSlope", "ZlsmaSlope", "AtrNorm", "DistKalman", "DistZlsma",
    "BodyRatio", "UpperWick", "LowerWick", "BarRangeAtr", "VolPct",
    "TodMinutes", "IsLong", "ChandLongNorm", "ChandShortNorm",
    "UnrealizedAtr", "WideStopActive", "InOpenWindow", "CScoreEntry"
]

app        = FastAPI(title="Kalman AI Gatekeeper", version="1.0")
model_pkg  = {"model": None, "threshold": 0.55, "features": FEATURE_NAMES,
              "trained_at": None, "loaded_at": None}
call_count = 0
block_count = 0


# ── Request / response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    approved:   bool
    confidence: float
    threshold:  float
    blocked_pct: float


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(path: str) -> bool:
    global model_pkg
    try:
        with open(path, "rb") as f:
            pkg = pickle.load(f)
        model_pkg["model"]      = pkg["model"]
        model_pkg["threshold"]  = pkg.get("threshold", 0.55)
        model_pkg["features"]   = pkg.get("features", FEATURE_NAMES)
        model_pkg["trained_at"] = pkg.get("trained_at", "unknown")
        model_pkg["loaded_at"]  = datetime.now().isoformat()
        log.info(f"Model loaded: {path}  trained_at={model_pkg['trained_at']}")
        return True
    except FileNotFoundError:
        log.warning(f"Model file not found: {path}  — server will fail-open until model is available")
        return False
    except Exception as e:
        log.error(f"Model load error: {e}")
        return False


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model(MODEL_PATH)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    global call_count, block_count

    call_count += 1

    # Fail-open: if no model loaded, always approve
    if model_pkg["model"] is None:
        log.warning("No model loaded — failing open (trade approved)")
        return PredictResponse(
            approved=True, confidence=1.0,
            threshold=model_pkg["threshold"],
            blocked_pct=0.0)

    expected = len(model_pkg["features"])
    if len(req.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(req.features)}")

    X = np.array(req.features, dtype=np.float32).reshape(1, -1)

    t0         = time.perf_counter()
    proba      = model_pkg["model"].predict_proba(X)[0, 1]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    threshold = model_pkg["threshold"]
    approved  = bool(proba >= threshold)

    if not approved:
        block_count += 1

    blocked_pct = block_count / call_count if call_count > 0 else 0.0

    log.info(f"predict  conf={proba:.3f}  thr={threshold:.2f}  "
             f"{'APPROVE' if approved else 'BLOCK '}  "
             f"{elapsed_ms:.1f}ms  total={call_count}  blocked={blocked_pct:.1%}")

    return PredictResponse(
        approved=approved,
        confidence=float(proba),
        threshold=threshold,
        blocked_pct=blocked_pct)


@app.get("/health")
async def health():
    return {
        "status":       "ok" if model_pkg["model"] is not None else "no_model",
        "model_loaded": model_pkg["model"] is not None,
        "trained_at":   model_pkg["trained_at"],
        "loaded_at":    model_pkg["loaded_at"],
        "threshold":    model_pkg["threshold"],
        "features":     model_pkg["features"],
        "calls_total":  call_count,
        "calls_blocked": block_count,
        "block_rate":   f"{block_count/max(call_count,1):.1%}"
    }


@app.post("/reload")
async def reload():
    success = load_model(MODEL_PATH)
    if not success:
        raise HTTPException(status_code=500, detail="Model reload failed — check logs")
    return {"status": "reloaded", "trained_at": model_pkg["trained_at"]}


@app.get("/threshold/{new_threshold}")
async def set_threshold(new_threshold: float):
    if not 0.0 <= new_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="Threshold must be 0.0–1.0")
    old = model_pkg["threshold"]
    model_pkg["threshold"] = new_threshold
    log.info(f"Threshold changed: {old:.2f} → {new_threshold:.2f}")
    return {"old_threshold": old, "new_threshold": new_threshold}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000,
                reload=False, log_level="info")