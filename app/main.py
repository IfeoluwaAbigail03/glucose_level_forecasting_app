"""
The FastAPI app. Loads both model bundles at startup:
  v1 (regression) -> the predicted glucose number + hyper flag
  v2 (classifier) -> the hypo advisory alert

A single /predict call runs both models on the same input and
combines their outputs into one response.
"""
from contextlib import asynccontextmanager
import xgboost as xgb
from fastapi import FastAPI, HTTPException

from app.model_loader import load_regression_bundle, load_classifier_bundle
from app.schemas import GlucosePredictionRequest, GlucosePredictionResponse
from app.config import V2_ALERT_THRESHOLD

reg_bundle = None
clf_bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global reg_bundle, clf_bundle
    reg_bundle = load_regression_bundle()
    clf_bundle = load_classifier_bundle()
    print(f"v1 (regression) loaded. Test MAE: {reg_bundle.metrics['test_mae']:.3f}")
    print(f"v2 (classifier) loaded. Test recall: {clf_bundle.metrics['test_recall']:.3f}")
    yield


app = FastAPI(title="Glucose Forecast API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "v1_loaded": reg_bundle is not None,
        "v2_loaded": clf_bundle is not None,
    }


def _build_dmatrix(row: dict, features: list[str]) -> xgb.DMatrix:
    missing = [f for f in features if f not in row]
    if missing:
        raise HTTPException(500, f"Model expects features not in request: {missing}")
    ordered_values = [row[f] for f in features]
    return xgb.DMatrix([ordered_values], feature_names=features)


@app.post("/predict", response_model=GlucosePredictionResponse)
def predict(req: GlucosePredictionRequest):
    row = req.model_dump()

    # v1: predicted glucose value
    reg_dmatrix = _build_dmatrix(row, reg_bundle.features)
    predicted_bg = float(reg_bundle.booster.predict(reg_dmatrix)[0])
    clinical_flag = "hyper_risk" if predicted_bg > 10.0 else "normal"

    # v2: hypo alert probability
    clf_dmatrix = _build_dmatrix(row, clf_bundle.features)
    hypo_prob = float(clf_bundle.booster.predict(clf_dmatrix)[0])
    hypo_alert = hypo_prob >= V2_ALERT_THRESHOLD

    advisory_note = (
        "Elevated hypo risk detected — verify with a fingerstick test before acting."
        if hypo_alert else
        "No elevated hypo risk detected by the advisory model."
    )

    return GlucosePredictionResponse(
        predicted_bg_1h=round(predicted_bg, 2),
        clinical_flag=clinical_flag,
        hypo_alert=hypo_alert,
        hypo_alert_probability=round(hypo_prob, 3),
        advisory_note=advisory_note,
    )