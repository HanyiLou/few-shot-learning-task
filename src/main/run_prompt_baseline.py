from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.trial_io import load_trials
from src.main.config_loader import load_task_config
from src.main.export_session_events import export_session_events
from src.models.prompt_baseline_runner import DashScopeProcedureRunner
from src.prompts.prompt_builder import prepare_trial_input, save_prepared_inputs


def append_prediction(predictions_output: Path, prediction) -> None:
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")


def load_existing_prediction_trial_ids(predictions_output: Path) -> set[str]:
    if not predictions_output.exists():
        return set()
    trial_ids: set[str] = set()
    with predictions_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            trial_id = payload.get("trial_id")
            if trial_id:
                trial_ids.add(trial_id)
    return trial_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare learning-only prompt inputs and run a DashScope yes/no baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/task_fewshot.yaml"),
        help="Task config YAML path.",
    )
    parser.add_argument(
        "--trials",
        type=Path,
        default=None,
        help="Input trial JSONL.",
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=None,
        help="Prepared message JSONL output path.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Prediction JSONL output path.",
    )
    parser.add_argument(
        "--mode",
        choices=["json_yes_no"],
        default="json_yes_no",
        help="Prompt mode.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="qwen3-vl-plus",
        help="Model id used by the API backend.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Video fps hint for DashScope local-video inputs.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="DASHSCOPE_API_KEY",
        help="Environment variable that stores the DashScope API key.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Optional DashScope base HTTP API URL. Leave empty when using SDK local-file upload.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root used to resolve local media paths into file:// URIs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for smoke testing a small number of trials.",
    )
    parser.add_argument(
        "--session-events-output",
        type=Path,
        default=None,
        help="Human-style session events CSV output path.",
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
        help="Optional string session identifier for the exported session table.",
    )
    args = parser.parse_args()

    config = load_task_config(args.config)
    trials_path = args.trials or Path(config["trials_path"])
    prepared_output = args.prepared_output or Path(config["prepared_inputs_path"])
    predictions_output = args.predictions_output or Path(config["predictions_path"])
    session_events_output = args.session_events_output or Path(config["session_events_path"])

    trials = load_trials(trials_path)
    if args.limit > 0:
        trials = trials[: args.limit]
    prepared_inputs = [prepare_trial_input(trial, mode=args.mode) for trial in trials]
    save_prepared_inputs(prepared_inputs, prepared_output)
    runner = DashScopeProcedureRunner(
        model_id=args.model_id,
        fps=args.fps,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        project_root=args.project_root,
    )

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    completed_trial_ids = load_existing_prediction_trial_ids(predictions_output)

    predictions = []
    total = len(prepared_inputs)
    for index, prepared_input in enumerate(prepared_inputs, start=1):
        trial_id = prepared_input["trial_id"]
        if trial_id in completed_trial_ids:
            print(f"[{index}/{total}] Skipping existing trial: {trial_id}", flush=True)
            continue
        print(f"[{index}/{total}] Running trial: {trial_id}", flush=True)
        prediction = runner.infer(prepared_input)
        predictions.append(prediction)
        append_prediction(predictions_output, prediction)
        completed_trial_ids.add(trial_id)
        print(
            f"[{index}/{total}] Done: {trial_id} "
            f"gold={prediction.gold_label} pred={prediction.predicted_label}",
            flush=True,
        )
    print(
        f"Prepared {len(prepared_inputs)} learning-only inputs with backend=dashscope_api "
        f"and wrote predictions to {predictions_output}"
    )
    export_session_events(
        trials_path=trials_path,
        predictions_path=predictions_output,
        output_path=session_events_output,
        session_num=args.session_num,
        session_id=args.session_id,
    )
    print(f"Wrote session-style events table to {session_events_output}")


if __name__ == "__main__":
    main()
