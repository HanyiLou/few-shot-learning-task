from __future__ import annotations

import argparse
from pathlib import Path

from src.data.index_builder import load_video_index
from src.sampling.export_trials import export_trials_jsonl
from src.sampling.trial_sampler import sample_human_procedure_trials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate human-procedure yes/no category-membership trials from a video index."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/metadata/video_index.csv"),
        help="Input video index CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trials/trial_set_v1/trials.jsonl"),
        help="Output trial JSONL.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="class_4",
        help="Target class used for the learning phase.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--attention-check-path",
        type=str,
        default="",
        help="Optional attention-check video path relative to the model video root.",
    )
    args = parser.parse_args()

    records = load_video_index(args.index)
    trials = sample_human_procedure_trials(
        records,
        target_class=args.target_class,
        seed=args.seed,
        include_attention_check=bool(args.attention_check_path),
        attention_check_path=args.attention_check_path or None,
    )
    export_trials_jsonl(trials, args.output)
    print(f"Wrote {len(trials)} human-procedure trials to {args.output}")


if __name__ == "__main__":
    main()
