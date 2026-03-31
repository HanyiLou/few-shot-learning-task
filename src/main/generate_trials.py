from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.index_builder import load_video_index
from src.main.config_loader import load_task_config
from src.sampling.export_trials import export_trials_jsonl
from src.sampling.trial_sampler import build_session_specs, sample_learning_only_trials


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"next_session_index": 0}
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, next_session_index: int, total_sessions: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "next_session_index": next_session_index,
                "total_sessions": total_sessions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate learning-only yes/no category-membership trials from a video index."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/task_fewshot.yaml"),
        help="Task config YAML path.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Input video index CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output trial JSONL.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=None,
        help="Number of new sessions to generate in this run.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Path to the persistent session traversal state file.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Reset traversal to the first session before generating.",
    )
    args = parser.parse_args()

    config = load_task_config(args.config)
    index_path = args.index or Path(config["index_path"])
    output_path = args.output or Path(config["trials_path"])
    seed = args.seed if args.seed is not None else int(config.get("seed", 13))
    num_sessions = (
        args.num_sessions if args.num_sessions is not None else int(config.get("num_sessions", 1))
    )
    state_path = args.state_path or Path(config["state_path"])

    records = load_video_index(index_path)
    session_specs = build_session_specs(records)
    state = {"next_session_index": 0} if args.reset_state else load_state(state_path)
    start_index = int(state.get("next_session_index", 0))

    if num_sessions <= 0:
        raise ValueError("--num-sessions must be positive")
    if start_index >= len(session_specs):
        raise ValueError(
            "No sessions remain to generate. Use --reset-state to start traversal from the beginning."
        )

    end_index = min(start_index + num_sessions, len(session_specs))
    selected_specs = session_specs[start_index:end_index]

    all_trials = []
    for offset, spec in enumerate(selected_specs):
        session_seed = seed + start_index + offset
        all_trials.extend(
            sample_learning_only_trials(
                records,
                target_class=str(spec["target_class"]),
                seed=session_seed,
                learning_verbs=list(spec["learning_verbs"]),
                session_num=int(spec["session_num"]),
                session_id=str(spec["session_id"]),
            )
        )

    export_trials_jsonl(all_trials, output_path)
    save_state(state_path, next_session_index=end_index, total_sessions=len(session_specs))
    print(
        f"Wrote {len(all_trials)} trials across {len(selected_specs)} session(s) to {output_path}. "
        f"Advanced session cursor to {end_index}/{len(session_specs)}."
    )


if __name__ == "__main__":
    main()
