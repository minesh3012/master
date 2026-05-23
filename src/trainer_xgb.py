import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .config import (
    DATA_LABELS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODELS_XGB_DIR,
    MODELS_XGB_FINAL_DIR,
    MODELS_XGB_VERSIONED_DIR,
)

def train_xgb_model():
    data_path = DATA_LABELS_DIR / "labeled.csv"
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_prob)
        print("ROC AUC:", auc)
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = MODELS_XGB_VERSIONED_DIR / f"v_{ts}"
    os.makedirs(version_dir, exist_ok=True)

    model_path = version_dir / "xgb_model.pkl"
    joblib.dump(model, model_path)

    final_path = MODELS_XGB_FINAL_DIR / "xgb_model_latest.pkl"
    joblib.dump(model, final_path)

    return model_path, final_path
