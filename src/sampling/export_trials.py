from __future__ import annotations

import json
from pathlib import Path

from src.data.schema import Trial


def export_trials_jsonl(trials: list[Trial], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(json.dumps(trial.to_dict(), ensure_ascii=False) + "\n")
