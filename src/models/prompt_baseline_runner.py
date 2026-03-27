from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.data.schema import Prediction


def extract_prediction(raw_response: str) -> tuple[str, str]:
    raw_response = raw_response.strip()
    try:
        parsed = json.loads(raw_response)
        answer = parsed.get("answer", "").strip().lower()
        confidence = parsed.get("confidence", "unknown")
        return answer, confidence
    except json.JSONDecodeError:
        return raw_response.split()[0].strip().lower(), "unknown"


class MockProcedureRunner:
    """Offline stub for validating the human-procedure pipeline before real VLM integration."""

    backend_name = "mock_procedure_runner"

    def infer(self, prepared_input: dict[str, Any]) -> Prediction:
        start = time.perf_counter()
        gold = prepared_input["gold_label"]
        response = json.dumps({"answer": gold, "confidence": "high"})
        predicted_label, confidence = extract_prediction(response)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return Prediction(
            trial_id=prepared_input["trial_id"],
            query_video_path=prepared_input["query_video_path"],
            gold_label=gold,
            predicted_label=predicted_label,
            raw_response=response,
            confidence=confidence,
            backend=self.backend_name,
            latency_ms=latency_ms,
        )


def load_prepared_inputs(input_path: Path) -> list[dict[str, Any]]:
    prepared_inputs: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                prepared_inputs.append(json.loads(line))
    return prepared_inputs


def save_predictions(predictions: list[Prediction], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
