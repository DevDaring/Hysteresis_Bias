"""
GPU usage monitoring and cost tracking.

Tracks elapsed GPU-hours and estimated cost for budget management.
Maintains a running log at results/gpu_usage.json.

# ============================================================
# PAPER CITATIONS
# Budget: ~50 GPU-hours ($170 at ~$3.39/hr) on H100
# ============================================================
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.utils.config import get_project_root


class GPUTracker:
    """
    Track GPU usage time and estimated cost.

    Usage:
        tracker = GPUTracker(cost_per_hour=3.50)
        tracker.start("phase1_injection")
        ... run experiment ...
        tracker.stop()
        tracker.report()
    """

    def __init__(self, cost_per_hour: float = 3.50):
        self.cost_per_hour = cost_per_hour
        self.start_time = None
        self.experiment_name = None
        self.log_path = get_project_root() / "results" / "gpu_usage.json"

    def start(self, experiment_name: str = "unnamed"):
        """Start tracking GPU time for an experiment."""
        self.experiment_name = experiment_name
        self.start_time = time.time()
        print(f"[GPU Tracker] Started: {experiment_name} at {datetime.now().isoformat()}")

    def stop(self) -> dict:
        """
        Stop tracking and log the result.

        Returns:
            Dict with elapsed time, cost, and metadata.
        """
        if self.start_time is None:
            raise RuntimeError("GPUTracker.start() was not called before stop().")

        elapsed_seconds = time.time() - self.start_time
        elapsed_hours = elapsed_seconds / 3600.0
        cost = elapsed_hours * self.cost_per_hour

        entry = {
            "experiment": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "elapsed_hours": round(elapsed_hours, 4),
            "estimated_cost_usd": round(cost, 2),
            "cost_per_hour": self.cost_per_hour,
        }

        # Append to running log
        self._append_log(entry)

        self.start_time = None
        return entry

    def report(self):
        """Print a summary of all GPU usage to date."""
        if not self.log_path.exists():
            print("[GPU Tracker] No usage data found.")
            return

        with open(self.log_path, "r") as f:
            log = json.load(f)

        total_hours = sum(e.get("elapsed_hours", 0) for e in log)
        total_cost = sum(e.get("estimated_cost_usd", 0) for e in log)

        print("=" * 60)
        print("GPU USAGE REPORT")
        print("=" * 60)
        for entry in log:
            print(
                f"  {entry['experiment']:30s} | "
                f"{entry['elapsed_hours']:6.2f} hrs | "
                f"${entry['estimated_cost_usd']:6.2f}"
            )
        print("-" * 60)
        print(f"  {'TOTAL':30s} | {total_hours:6.2f} hrs | ${total_cost:6.2f}")
        print("=" * 60)

    def _append_log(self, entry: dict):
        """Append an entry to the JSON log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        log = []
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                log = json.load(f)

        log.append(entry)

        with open(self.log_path, "w") as f:
            json.dump(log, f, indent=2)

        print(
            f"[GPU Tracker] Stopped: {entry['experiment']} — "
            f"{entry['elapsed_hours']:.2f} hrs, ${entry['estimated_cost_usd']:.2f}"
        )
