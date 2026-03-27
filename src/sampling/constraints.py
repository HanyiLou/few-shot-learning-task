from __future__ import annotations


DEFAULT_TASK_TYPE = "category_membership"
DEFAULT_CANDIDATE_LABELS = ["yes", "no"]
DEFAULT_OUTPUT_FORMAT = {"answer": "<yes|no>", "confidence": "<low|medium|high>"}
DEFAULT_TASK_INSTRUCTION = (
    "Learn the hidden action category from the learning videos, observe the review videos, "
    "and decide whether the query video belongs to the same category. Answer only yes or no."
)
