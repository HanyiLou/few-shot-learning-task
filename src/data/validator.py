from __future__ import annotations

from src.data.schema import Trial


def validate_trial(trial: Trial) -> None:
    if not trial.trial_id:
        raise ValueError("trial_id is required")
    if not trial.task_type:
        raise ValueError("task_type is required")
    if trial.task_type != "category_membership":
        raise ValueError("Only category_membership is supported in the current pipeline")
    if not trial.target_class:
        raise ValueError("target_class is required")
    if not trial.learning_examples:
        raise ValueError("learning_examples are required")
    if trial.query_example is None:
        raise ValueError("query_example is required")
    if not trial.expected_output_format:
        raise ValueError("expected_output_format is required")
    if not trial.task_instruction:
        raise ValueError("task_instruction is required")
