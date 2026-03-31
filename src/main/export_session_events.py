from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.trial_io import load_trials


CLASS_TO_VERB_CLASS = {
    "class_1": "continuous",
    "class_2": "instantaneous",
    "class_3": "attachment",
    "class_4": "destruction",
}


def load_prediction_rows(predictions_path: Path) -> list[dict]:
    rows: list[dict] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _format_rt_milliseconds(latency_ms: int) -> str:
    if latency_ms <= 0:
        return ""
    return str(latency_ms)


def _verb_class_for(class_name: str) -> str:
    return CLASS_TO_VERB_CLASS.get(class_name, "")


def _load_existing_session_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["session_id"] for row in reader if row.get("session_id")}


def export_session_events(
    trials_path: Path,
    predictions_path: Path,
    output_path: Path,
    session_num: int = 1,
    session_id: str = "",
) -> None:
    trials = load_trials(trials_path)
    prediction_rows = load_prediction_rows(predictions_path)

    if not trials:
        raise ValueError("No trials found")
    if not prediction_rows:
        raise ValueError("No predictions found")

    prediction_by_trial_id = {row["trial_id"]: row for row in prediction_rows}
    missing_trial_ids = [trial.trial_id for trial in trials if trial.trial_id not in prediction_by_trial_id]
    if missing_trial_ids:
        raise ValueError(f"Missing predictions for trial ids: {missing_trial_ids}")

    fieldnames = [
        "num",
        "session_id",
        "trialNum",
        "phase",
        "verb",
        "file",
        "class_name",
        "verb_class",
        "condition",
        "rt",
        "response",
        "gold_label",
        "correct",
        "playCounts",
        "trial_id",
    ]

    rows: list[dict[str, str | int]] = []
    session_groups: dict[str, list] = {}
    for trial in trials:
        trial_session_id = str(trial.metadata.get("session_id", ""))
        if not trial_session_id:
            trial_session_id = session_id or "session_001"
        session_groups.setdefault(trial_session_id, []).append(trial)

    existing_session_ids = _load_existing_session_ids(output_path)

    for group_session_id in sorted(session_groups):
        if group_session_id in existing_session_ids:
            continue
        session_trials = session_groups[group_session_id]
        first_trial = session_trials[0]
        learning_examples = first_trial.learning_examples
        if not learning_examples:
            raise ValueError("Trials contain no learning examples")

        for trial in session_trials[1:]:
            learning_ids = [example.video_id for example in trial.learning_examples]
            first_ids = [example.video_id for example in learning_examples]
            if learning_ids != first_ids:
                raise ValueError(
                    f"Session export expects trials within {group_session_id} to share one learning block"
                )

        group_session_num = int(first_trial.metadata.get("session_num", session_num))
        target_class = first_trial.target_class
        condition = _verb_class_for(target_class)
        trial_num = 1
        for example in learning_examples:
            rows.append(
                {
                    "num": group_session_num,
                    "session_id": group_session_id,
                    "trialNum": trial_num,
                    "phase": "learning",
                    "verb": example.verb,
                    "file": example.video_path,
                    "class_name": example.class_name,
                    "verb_class": _verb_class_for(example.class_name),
                    "condition": condition,
                    "rt": "",
                    "response": "",
                    "gold_label": "",
                    "correct": "",
                    "playCounts": 1,
                    "trial_id": "",
                }
            )
            trial_num += 1

        for trial in session_trials:
            prediction = prediction_by_trial_id[trial.trial_id]
            predicted_label = prediction["predicted_label"]
            gold_label = prediction["gold_label"]
            rows.append(
                {
                    "num": group_session_num,
                    "session_id": group_session_id,
                    "trialNum": trial_num,
                    "phase": "testing",
                    "verb": trial.query_example.verb,
                    "file": trial.query_example.video_path,
                    "class_name": trial.query_example.class_name,
                    "verb_class": _verb_class_for(trial.query_example.class_name),
                    "condition": condition,
                    "rt": _format_rt_milliseconds(int(prediction.get("latency_ms", 0))),
                    "response": predicted_label,
                    "gold_label": gold_label,
                    "correct": int(predicted_label == gold_label),
                    "playCounts": 1,
                    "trial_id": trial.trial_id,
                }
            )
            trial_num += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VLM outputs to a human-style session events CSV.")
    parser.add_argument(
        "--trials",
        type=Path,
        default=Path("data/trials/learning_only/trials.jsonl"),
        help="Input trial JSONL.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/predictions.jsonl"),
        help="Input predictions JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/session_events.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--session-num",
        type=int,
        default=1,
        help="Numeric session identifier, analogous to a participant number.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="",
        help="Optional string session identifier.",
    )
    args = parser.parse_args()

    export_session_events(
        trials_path=args.trials,
        predictions_path=args.predictions,
        output_path=args.output,
        session_num=args.session_num,
        session_id=args.session_id,
    )
    print(f"Wrote session events CSV to {args.output}")


if __name__ == "__main__":
    main()
