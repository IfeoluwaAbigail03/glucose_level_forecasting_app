"""
Loads both model bundles once at startup:
  v1 = regression model (predicted glucose value)
  v2 = classifier (hypo risk probability, advisory alert)

Same fail-loudly philosophy as before — if either bundle is missing
or malformed, the app refuses to start rather than serving predictions
from a model that silently failed to load.
"""
import json
import xgboost as xgb
from app.config import (
    V1_MODEL_PATH, V1_FEATURE_LIST_PATH, V1_METRICS_PATH,
    V2_MODEL_PATH, V2_FEATURE_LIST_PATH, V2_METRICS_PATH,
)


class ModelBundle:
    def __init__(self, booster, features, metrics):
        self.booster = booster
        self.features = features
        self.metrics = metrics


def _load_bundle(model_path, feature_path, metrics_path) -> ModelBundle:
    for path in [model_path, feature_path, metrics_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    features = json.loads(feature_path.read_text())
    metrics = json.loads(metrics_path.read_text())

    if len(features) == 0:
        raise ValueError(f"feature_list.json at {feature_path} is empty.")

    return ModelBundle(booster, features, metrics)


def load_regression_bundle() -> ModelBundle:
    return _load_bundle(V1_MODEL_PATH, V1_FEATURE_LIST_PATH, V1_METRICS_PATH)


def load_classifier_bundle() -> ModelBundle:
    return _load_bundle(V2_MODEL_PATH, V2_FEATURE_LIST_PATH, V2_METRICS_PATH)