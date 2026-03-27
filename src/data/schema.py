from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class VideoRecord:
    video_id: str
    class_name: str
    verb: str
    file_path: str
    source_filename: str
    split: str = "unspecified"
    source: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialVideo:
    video_id: str
    video_path: str
    verb: str
    class_name: str
    phase: str
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trial:
    trial_id: str
    task_type: str
    target_class: str
    practice_learning_examples: list[TrialVideo] = field(default_factory=list)
    practice_testing_examples: list[TrialVideo] = field(default_factory=list)
    learning_examples: list[TrialVideo] = field(default_factory=list)
    review_examples: list[TrialVideo] = field(default_factory=list)
    query_example: TrialVideo | None = None
    candidate_labels: list[str] = field(default_factory=lambda: ["yes", "no"])
    task_instruction: str = ""
    expected_output_format: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["practice_learning_examples"] = [
            example.to_dict() for example in self.practice_learning_examples
        ]
        payload["practice_testing_examples"] = [
            example.to_dict() for example in self.practice_testing_examples
        ]
        payload["learning_examples"] = [example.to_dict() for example in self.learning_examples]
        payload["review_examples"] = [example.to_dict() for example in self.review_examples]
        payload["query_example"] = self.query_example.to_dict() if self.query_example else None
        return payload


@dataclass
class Prediction:
    trial_id: str
    query_video_path: str
    gold_label: str
    predicted_label: str
    raw_response: str
    confidence: str = "unknown"
    backend: str = "mock"
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
