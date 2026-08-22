"""
Logs every prediction request: the input features, both models'
outputs, and when it happened. This becomes the data source for
drift detection — comparing what the API is actually being asked
against what the models were trained on.

NOTE: SQLite file lives at /tmp by default in this setup, which
Render's free tier wipes on every redeploy. Fine for learning the
mechanism; for real persistence you'd need a Render Disk or an
external database.
"""
import sqlite3
import json
import time
from pathlib import Path

DB_PATH = Path("prediction_log.db")

FEATURE_COLUMNS = [
    "bg_past_0_00", "bg_prev_1", "bg_prev_2", "bg_slope_15min",
    "bg_slope_60min", "bg_change_15min", "bg_change_60min",
    "bg_rolling_mean_3", "bg_rolling_std_3", "roc_15min", "roc_30min",
    "acceleration", "hour", "hypo_risk",
]


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ts REAL,
            features_json TEXT,
            predicted_bg_1h REAL,
            hypo_alert_probability REAL,
            hypo_alert INTEGER
        )
    """)
    return conn


def log_prediction(features: dict, predicted_bg: float, hypo_prob: float, hypo_alert: bool):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO predictions (ts, features_json, predicted_bg_1h, hypo_alert_probability, hypo_alert) VALUES (?, ?, ?, ?, ?)",
        (time.time(), json.dumps(features), predicted_bg, hypo_prob, int(hypo_alert)),
    )
    conn.commit()
    conn.close()


def fetch_recent(limit: int = 1000) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT ts, features_json, predicted_bg_1h, hypo_alert_probability, hypo_alert FROM predictions ORDER BY ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    out = []
    for ts, features_json, pred_bg, hypo_prob, hypo_alert in rows:
        rec = json.loads(features_json)
        rec.update({"ts": ts, "predicted_bg_1h": pred_bg, "hypo_alert_probability": hypo_prob, "hypo_alert": bool(hypo_alert)})
        out.append(rec)
    return out