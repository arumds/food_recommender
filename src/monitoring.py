"""
In-process monitoring for the recommendation API: request latency percentiles,
feature drift (PSI against a reference snapshot saved by main.py) and model staleness.
"""
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np

N_BINS = 10
PSI_WARN = 0.1
PSI_ALERT = 0.25
MIN_DRIFT_SAMPLES = 200


def _bucketize(values, ref):
    # categorical: one bin per seen value + a last bin for unseen values
    if "categories" in ref:
        cats = np.asarray(ref["categories"])
        idx = np.searchsorted(cats, values).clip(max=len(cats) - 1)
        bins = np.where(cats[idx] == values, idx, len(cats))
        return np.bincount(bins, minlength=len(cats) + 1)
    edges = np.asarray(ref["edges"])
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)


def build_feature_reference(feature_df, feature_cols, source):
    """Bin each feature so live traffic can be compared against it later."""
    reference = {"source": source, "n_rows": int(len(feature_df)), "features": {}}
    for col in feature_cols:
        values = feature_df[col].astype(float).to_numpy()
        uniques = np.unique(values)
        if len(uniques) <= N_BINS:
            ref = {"categories": uniques.tolist()}
        else:
            ref = {"edges": np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)[1:-1])).tolist()}
        counts = _bucketize(values, ref)
        ref["expected"] = (counts / counts.sum()).tolist()
        ref["mean"] = float(values.mean())
        reference["features"][col] = ref
    return reference


def psi(expected, actual, eps=1e-4):
    expected = np.clip(np.asarray(expected, dtype=float), eps, None)
    actual = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _status(value, warn, alert):
    if value is None:
        return "n/a"
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


class Monitor:
    def __init__(self, models_dir, window=5000, latency_slo_ms=None, max_model_age_hours=None):
        self.started_at = time.time()
        self.latency_slo_ms = latency_slo_ms or float(os.environ.get("LATENCY_SLO_MS", 100))
        self.max_model_age_hours = max_model_age_hours or float(os.environ.get("MAX_MODEL_AGE_HOURS", 168))
        self._lock = threading.Lock()
        self._latency = defaultdict(lambda: deque(maxlen=window))
        self._status_codes = defaultdict(int)
        self._request_times = deque(maxlen=window)
        self._features = defaultdict(lambda: deque(maxlen=window * 10))
        self.reference = self._read_json(os.path.join(models_dir, "feature_reference.json"))
        self.metadata = self._read_json(os.path.join(models_dir, "model_metadata.json"))

    @staticmethod
    def _read_json(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def record_request(self, path, status_code, latency_ms):
        with self._lock:
            self._status_codes[status_code] += 1
            self._request_times.append(time.time())
            self._latency[f"{path}:total_ms"].append(latency_ms)

    def record_stage_timings(self, path, timings):
        with self._lock:
            for stage, ms in timings.items():
                if stage != "total_ms":
                    self._latency[f"{path}:{stage}"].append(ms)

    def record_features(self, feature_df):
        with self._lock:
            for col in feature_df.columns:
                self._features[col].extend(feature_df[col].astype(float).tolist())

    def latency_summary(self):
        with self._lock:
            series = {k: np.asarray(v) for k, v in self._latency.items() if v}
        summary = {}
        for key, values in sorted(series.items()):
            p50, p95, p99 = np.percentile(values, [50, 95, 99])
            summary[key] = {
                "count": int(len(values)),
                "p50": round(float(p50), 2),
                "p95": round(float(p95), 2),
                "p99": round(float(p99), 2),
                "max": round(float(values.max()), 2),
            }
        return summary

    def drift_summary(self):
        if not self.reference:
            return {}
        with self._lock:
            live = {k: np.asarray(v) for k, v in self._features.items()}
        summary = {}
        for col, ref in self.reference["features"].items():
            values = live.get(col)
            n = 0 if values is None else len(values)
            if n < MIN_DRIFT_SAMPLES:
                summary[col] = {"n": n, "psi": None, "status": "n/a"}
                continue
            counts = _bucketize(values, ref)
            value = psi(ref["expected"], counts / counts.sum())
            summary[col] = {
                "n": n,
                "psi": round(value, 4),
                "reference_mean": round(ref["mean"], 4),
                "live_mean": round(float(values.mean()), 4),
                "status": _status(value, PSI_WARN, PSI_ALERT),
            }
        return summary

    def staleness(self):
        if not self.metadata or "trained_at" not in self.metadata:
            return {"trained_at": None, "age_hours": None, "status": "n/a"}
        trained_at = datetime.fromisoformat(self.metadata["trained_at"])
        age_hours = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600
        return {
            "trained_at": self.metadata["trained_at"],
            "age_hours": round(age_hours, 2),
            "max_age_hours": self.max_model_age_hours,
            "status": "alert" if age_hours > self.max_model_age_hours else "ok",
        }

    def snapshot(self):
        latency = self.latency_summary()
        drift = self.drift_summary()
        staleness = self.staleness()
        with self._lock:
            status_codes = dict(self._status_codes)
            recent = [t for t in self._request_times if t > time.time() - 60]

        alerts = []
        total = latency.get("/recommend:total_ms")
        if total and total["p95"] > self.latency_slo_ms:
            alerts.append(f"/recommend p95 latency {total['p95']}ms exceeds SLO {self.latency_slo_ms}ms")
        for col, d in drift.items():
            if d["status"] == "alert":
                alerts.append(f"feature drift on '{col}': PSI {d['psi']} >= {PSI_ALERT}")
        if staleness["status"] == "alert":
            alerts.append(f"model is {staleness['age_hours']}h old (max {self.max_model_age_hours}h), retrain")
        n_requests = sum(status_codes.values())
        n_errors = sum(v for k, v in status_codes.items() if k >= 500)
        if n_requests >= 20 and n_errors / n_requests > 0.01:
            alerts.append(f"5xx error rate {n_errors / n_requests:.1%}")

        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "requests_total": n_requests,
            "requests_last_60s": len(recent),
            "status_codes": status_codes,
            "latency_ms": latency,
            "latency_slo_ms": self.latency_slo_ms,
            "feature_drift": drift,
            "drift_reference": {k: v for k, v in (self.reference or {}).items() if k != "features"},
            "model": {**(self.metadata or {}), "staleness": staleness},
            "alerts": alerts,
        }
