from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import compute_accuracy, load_prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM yes/no predictions.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/predictions.jsonl"),
        help="Input predictions JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics.json"),
        help="Output metrics JSON path.",
    )
    args = parser.parse_args()

    rows = load_prediction_rows(args.predictions)
    metrics = compute_accuracy(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
