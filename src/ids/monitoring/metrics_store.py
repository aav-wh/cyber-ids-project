"""
ids.monitoring.metrics_store
-----------------------------
Persistent JSON-lines metrics store for production monitoring.

Appends one JSON record per evaluation run to a .jsonl file so that
long-term performance trends can be analysed offline.
"""

from __future__ import annotations

import json
import os
import time
from threading import RLock


class MetricsStore:
    """
    Append-only JSON-lines store for performance metrics.

    Parameters
    ----------
    filepath : path to the .jsonl metrics file
    """

    def __init__(self, filepath: str) -> None:
        self._path = filepath
        self._lock = RLock()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def record(self, metrics: dict, source: str = "") -> None:
        """
        Append a metrics snapshot to the store.

        Parameters
        ----------
        metrics : dict of scalar metric values
        source  : label for the source (e.g. 'daily_eval', 'live_window')
        """
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source":    source,
            **{k: v for k, v in metrics.items() if not isinstance(v, str) or k == "model"},
        }

        with self._lock:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def read_all(self) -> list[dict]:
        """Read all stored metric records."""
        if not os.path.exists(self._path):
            return []
        with self._lock:
            with open(self._path) as f:
                return [json.loads(line) for line in f if line.strip()]

    def read_last_n(self, n: int) -> list[dict]:
        """Return the last n records."""
        records = self.read_all()
        return records[-n:]

    def trend(self, metric: str, source: str | None = None) -> list[tuple[str, float]]:
        """
        Extract a time series for a single metric.

        Parameters
        ----------
        metric : metric key (e.g. 'f1')
        source : filter by source label

        Returns
        -------
        list of (timestamp, value) tuples
        """
        records = self.read_all()
        if source:
            records = [r for r in records if r.get("source") == source]
        return [
            (r["timestamp"], r[metric])
            for r in records
            if metric in r and isinstance(r[metric], (int, float))
        ]

    def clear(self) -> None:
        """Delete all stored records (use with caution)."""
        with self._lock:
            if os.path.exists(self._path):
                os.remove(self._path)
