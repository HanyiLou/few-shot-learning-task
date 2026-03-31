from __future__ import annotations


DEFAULT_TASK_TYPE = "category_membership"
DEFAULT_CANDIDATE_LABELS = ["yes", "no"]
DEFAULT_OUTPUT_FORMAT = {"answer": "<yes|no>", "confidence": "<low|medium|high>"}
DEFAULT_TASK_INSTRUCTION = (
    "Infer the shared action pattern from the learning videos, then decide whether the query "
    "video matches that same action category. Set the JSON field 'answer' to either 'yes' or 'no'."
)
