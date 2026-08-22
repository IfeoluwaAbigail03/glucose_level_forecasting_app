"""
Population Stability Index (PSI) drift detection — compares live
prediction traffic against the training-time feature distributions
saved in reference_stats.json.

PSI rule of thumb:
  < 0.1   -> no significant drift
  0.1-0.2 -> moderate drift, worth watching
  > 0.2   -> significant drift, investigate

No external dependency needed — it's straightforward with numpy.
"""
import json
import numpy as np
from pathlib import Path

REFERENCE_PATH = Path("models") / "v1" / "reference_stats.json"


def _psi_for_feature(reference_values: list[float], live_values: list[float], bins: int = 10) -> float:
    ref = np.asarray(reference_values, dtype=float)
    live = np.asarray(live_values, dtype=float)
    if len(ref) == 0 or len(live) == 0:
        return 0.0

    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0  # feature is near-constant in training, PSI isn't meaningful here

    ref_counts, _ = np.histogram(ref, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    live_pct = np.clip(live_counts / max(live_counts.sum(), 1), 1e-4, None)

    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))


def compute_drift_report(recent_records: list[dict], min_samples: int = 30) -> dict:
    reference_stats = json.loads(REFERENCE_PATH.read_text())
    features = [f for f in reference_stats.keys() if not f.startswith("_")]

    if len(recent_records) < min_samples:
        return {"status": "insufficient_data", "n_logged": len(recent_records), "min_required": min_samples}

    report = {}
    any_drift = False
    for feat in features:
        live_vals = [r[feat] for r in recent_records if feat in r]
        ref_vals = reference_stats[feat]["values"]

        psi = _psi_for_feature(ref_vals, live_vals)
        status = "drift" if psi > 0.2 else "watch" if psi > 0.1 else "stable"
        any_drift = any_drift or (status == "drift")
        report[feat] = {"psi": round(psi, 4), "status": status}

    return {
        "status": "ok",
        "n_samples": len(recent_records),
        "drift_detected": any_drift,
        "features": report,
    }