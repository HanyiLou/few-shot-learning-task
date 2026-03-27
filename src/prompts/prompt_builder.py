from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.schema import Trial, TrialVideo
from src.data.validator import validate_trial


PLAIN_YES_NO_PROMPT = """
You are participating in a category-learning experiment based on human action videos.

Procedure:
1. In the learning phase, watch example videos from one hidden action category.
2. In the review phase, watch several additional videos.
3. In the testing phase, decide whether the query video belongs to the same hidden category learned in the learning phase.

Rules:
- Focus on the shared action pattern, not background details.
- Do not try to explain your reasoning.
- Respond with exactly one word: yes or no.
""".strip()


JSON_YES_NO_PROMPT = """
You are participating in a category-learning experiment based on human action videos.

Infer the hidden action category from the learning videos, observe the review videos, and decide whether the query video belongs to the same category.

Return your answer in JSON:
{
  "answer": "<yes|no>",
  "confidence": "<low|medium|high>"
}

Rules:
- Focus on the action pattern shared by the learning videos.
- Ignore irrelevant background details.
- Do not output any text outside the JSON object.
""".strip()


def build_prompt(task_type: str = "category_membership", mode: str = "plain_yes_no") -> str:
    if task_type != "category_membership":
        raise ValueError(f"Unsupported task_type={task_type}")
    if mode == "plain_yes_no":
        return PLAIN_YES_NO_PROMPT
    if mode == "json_yes_no":
        return JSON_YES_NO_PROMPT
    raise ValueError(f"Unsupported mode={mode}")


def _append_phase_video(
    content: list[dict[str, Any]],
    example: TrialVideo,
    phase_label: str,
    index: int,
) -> None:
    content.append(
        {
            "type": "text",
            "text": f"{phase_label} {index}: watch this video carefully.",
        }
    )
    media_type = "image" if example.video_path.lower().endswith((".jpg", ".jpeg", ".png")) else "video"
    content.append({"type": media_type, media_type: example.video_path})


def build_messages(trial: Trial, prompt_text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

    content.append(
        {
            "type": "text",
            "text": (
                "Practice phase, part 1: study these images. They all belong to the same practice "
                "category. Try to infer what they have in common."
            ),
        }
    )
    for index, example in enumerate(trial.practice_learning_examples, start=1):
        _append_phase_video(content, example, "Practice learning image", index)

    content.append(
        {
            "type": "text",
            "text": (
                "Practice phase, part 2: here are the practice test images with the correct answers. "
                "Use them to understand the yes/no decision rule before the real experiment."
            ),
        }
    )
    for index, example in enumerate(trial.practice_testing_examples, start=1):
        _append_phase_video(content, example, "Practice test image", index)
        content.append(
            {
                "type": "text",
                "text": f"Correct answer for practice test image {index}: {example.label}.",
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "Learning phase: these videos belong to the same hidden action category. "
                "Infer the shared pattern from them."
            ),
        }
    )
    for index, example in enumerate(trial.learning_examples, start=1):
        _append_phase_video(content, example, "Learning video", index)

    content.append(
        {
            "type": "text",
            "text": (
                "Review phase: watch these additional videos before making the final test decision. "
                "Some may belong to the hidden category and some may not."
            ),
        }
    )
    for index, example in enumerate(trial.review_examples, start=1):
        _append_phase_video(content, example, "Review video", index)

    content.append(
        {
            "type": "text",
            "text": (
                "Testing phase: decide whether the next query video belongs to the same hidden "
                "category as the learning videos."
            ),
        }
    )
    _append_phase_video(content, trial.query_example, "Query video", 1)
    content.append(
        {
            "type": "text",
            "text": trial.task_instruction,
        }
    )

    return [{"role": "user", "content": content}]


def prepare_trial_input(trial: Trial, mode: str = "plain_yes_no") -> dict[str, Any]:
    validate_trial(trial)
    prompt_text = build_prompt(task_type=trial.task_type, mode=mode)
    return {
        "trial_id": trial.trial_id,
        "task_type": trial.task_type,
        "target_class": trial.target_class,
        "practice_learning_count": len(trial.practice_learning_examples),
        "practice_testing_count": len(trial.practice_testing_examples),
        "query_video_path": trial.query_example.video_path,
        "gold_label": trial.query_example.label,
        "candidate_labels": trial.candidate_labels,
        "expected_output_format": trial.expected_output_format,
        "messages": build_messages(trial, prompt_text),
        "metadata": trial.metadata,
    }


def save_prepared_inputs(prepared_inputs: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in prepared_inputs:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
