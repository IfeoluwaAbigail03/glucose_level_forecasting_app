# Glucose Forecast API

Predicts blood glucose 1 hour ahead from recent CGM (continuous glucose monitor) readings, and separately flags elevated hypoglycemia risk — deployed as a FastAPI service with built-in drift monitoring.

**Live API:** https://glucose-level-forecasting-app.onrender.com
**Interactive docs:** https://glucose-level-forecasting-app.onrender.com/docs

## What this does

Two models work together on every request:

- **v1 (regression)** — predicts the actual glucose value 1 hour from now. Test MAE: 1.649 mmol/L.
- **v2 (classifier)** — predicts the probability of hypoglycemia (<3.9 mmol/L) within that hour, as an advisory alert. Test recall: 59.2%, precision: 2.0%.

The classifier improved detection of rare hypoglycemia events compared with thresholding the regression model, but held-out patient performance remained limited.

## Why the low precision is intentional

The classifier was thresholded to prioritize recall because the project's objective is to reduce missed hypoglycemia events. This necessarily produces more false positives, reflected in the low test precision (2.0%). The resulting alert should therefore be interpreted as a high-sensitivity screening signal rather than a diagnosis, requiring confirmation (e.g. a fingerstick test) before any action is taken.

## Dataset limitations (disclosed honestly)

- Trained on the BrisT1D dataset — only 9 total patients (7 train, 2 held-out test).
- Held-out test performance for hypoglycemia detection was noticeably worse than cross-validation estimated, tied to the 2 test patients having faster-onset hypo episodes than the 7 training patients showed.
- Precision figures are highly sensitive to how common hypoglycemia actually is in whoever uses this — they are not a fixed model property.
- This is a learning/portfolio project, not a validated clinical tool.

## Architecture

app/
main.py - FastAPI app, /predict, /health, /metrics
config.py - paths to model files
schemas.py - request/response validation
model_loader.py - loads both model bundles at startup
prediction_log.py - logs every prediction (SQLite)
drift_detection.py - PSI-based drift detection vs. training baseline
models/
v1/ - regression model bundle (model, feature list, metrics, reference stats)
v2/ - classifier model bundle (same structure)
legacy/ - original Streamlit prototype, kept for reference
Dockerfile
render.yaml - not currently used; deployed via Render's dashboard directly
requirements.txt


## API

**POST /predict** — takes 14 glucose/time features, returns:
```json
{
  "predicted_bg_1h": 6.08,
  "clinical_flag": "normal",
  "hypo_alert": false,
  "hypo_alert_probability": 0.101,
  "advisory_note": "No elevated hypo risk detected by the advisory model."
}
```

**GET /health** — confirms both models loaded.

**GET /metrics** — PSI drift report comparing recent logged predictions against training data distributions. Returns `insufficient_data` until at least 30 predictions have been logged.

## Known limitation: monitoring persistence

Prediction logs are stored in SQLite on Render's local disk, which is wiped on every redeploy (free tier has no persistent disk). Drift detection works correctly within a deploy's lifetime but doesn't currently accumulate history across deploys. A production version would need a Render persistent disk or an external database.

## How the models were built

Full pipeline: leakage-fixed feature engineering → patient-level train/val/test split → GroupKFold cross-validated hyperparameter tuning (never touching the test set) → one-time honest test evaluation → documented limitations. See `models/v1/metrics.json` and `models/v2/metrics.json` for full evaluation details, including known limitations recorded at training time.

## Local development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# -> http://localhost:8000/docs
```

