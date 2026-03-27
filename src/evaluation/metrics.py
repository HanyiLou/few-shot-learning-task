from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.signal_detection import compute_d_prime


def load_prediction_rows(predictions_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["gold_label"] == row["predicted_label"])

    gold_counts = Counter(row["gold_label"] for row in rows)
    pred_counts = Counter(row["predicted_label"] for row in rows)

    yes_rows = [row for row in rows if row["gold_label"] == "yes"]
    no_rows = [row for row in rows if row["gold_label"] == "no"]
    hit_rate = (
        sum(1 for row in yes_rows if row["predicted_label"] == "yes") / len(yes_rows)
        if yes_rows
        else 0.0
    )
    false_alarm_rate = (
        sum(1 for row in no_rows if row["predicted_label"] == "yes") / len(no_rows)
        if no_rows
        else 0.0
    )

    metrics: dict[str, Any] = {
        "n_trials": total,
        "accuracy": (correct / total) if total else 0.0,
        "gold_label_counts": dict(gold_counts),
        "predicted_label_counts": dict(pred_counts),
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
    }

    if yes_rows and no_rows:
        metrics["d_prime"] = compute_d_prime(hit_rate, false_alarm_rate)
    return metrics
