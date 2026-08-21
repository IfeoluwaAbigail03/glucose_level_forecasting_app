"""
Central config. Now tracks two model versions:
  v1 = regression model, predicts the actual future glucose number
  v2 = classifier, predicts hypo risk probability (advisory alert)
"""
from pathlib import Path

V1_DIR = Path("models") / "v1"
V2_DIR = Path("models") / "v2"

V1_MODEL_PATH = V1_DIR / "model.json"
V1_FEATURE_LIST_PATH = V1_DIR / "feature_list.json"
V1_METRICS_PATH = V1_DIR / "metrics.json"

V2_MODEL_PATH = V2_DIR / "model.json"
V2_FEATURE_LIST_PATH = V2_DIR / "feature_list.json"
V2_METRICS_PATH = V2_DIR / "metrics.json"

# The probability threshold we validated for the hypo alert
V2_ALERT_THRESHOLD = 0.5