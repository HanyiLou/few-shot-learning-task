from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.schema import Trial, TrialVideo
from src.models.prompt_baseline_runner import MockProcedureRunner, save_predictions
from src.prompts.prompt_builder import prepare_trial_input, save_prepared_inputs


def load_trials(trials_path: Path) -> list[Trial]:
    trials: list[Trial] = []
    with trials_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload["practice_learning_examples"] = [
                TrialVideo(**example) for example in payload.get("practice_learning_examples", [])
            ]
            payload["practice_testing_examples"] = [
                TrialVideo(**example) for example in payload.get("practice_testing_examples", [])
            ]
            payload["learning_examples"] = [
                TrialVideo(**example) for example in payload.get("learning_examples", [])
            ]
            payload["review_examples"] = [
                TrialVideo(**example) for example in payload.get("review_examples", [])
            ]
            query_payload = payload.get("query_example")
            payload["query_example"] = TrialVideo(**query_payload) if query_payload else None
            trials.append(Trial(**payload))
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare procedure-aware prompt inputs and run a mock yes/no baseline."
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=Path("data/trials/trial_set_v1/trials.jsonl"),
        help="Input trial JSONL.",
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=Path("outputs/prepared_inputs.jsonl"),
        help="Prepared message JSONL output path.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("outputs/mock_predictions.jsonl"),
        help="Prediction JSONL output path.",
    )
    parser.add_argument(
        "--mode",
        choices=["plain_yes_no", "json_yes_no"],
        default="json_yes_no",
        help="Prompt mode.",
    )
    args = parser.parse_args()

    trials = load_trials(args.trials)
    prepared_inputs = [prepare_trial_input(trial, mode=args.mode) for trial in trials]
    save_prepared_inputs(prepared_inputs, args.prepared_output)

    runner = MockProcedureRunner()
    predictions = [runner.infer(prepared_input) for prepared_input in prepared_inputs]
    save_predictions(predictions, args.predictions_output)
    print(f"Prepared {len(prepared_inputs)} procedure inputs and wrote predictions to {args.predictions_output}")


if __name__ == "__main__":
    main()
