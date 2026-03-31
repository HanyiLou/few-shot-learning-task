from __future__ import annotations

import json
from pathlib import Path

from src.data.schema import Trial, TrialVideo


def load_trials(trials_path: Path) -> list[Trial]:
    trials: list[Trial] = []
    with trials_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload["learning_examples"] = [
                TrialVideo(**example) for example in payload.get("learning_examples", [])
            ]
            query_payload = payload.get("query_example")
            payload["query_example"] = TrialVideo(**query_payload) if query_payload else None
            trials.append(Trial(**payload))
    return trials
