from __future__ import annotations

from src.data.schema import TrialVideo


PRACTICE_LEARNING_IMAGES = [
    "data/raw/practice/learning/practice_learning_cat.jpg",
    "data/raw/practice/learning/practice_learning_cat2.jpg",
    "data/raw/practice/learning/practice_learning_panda.jpg",
    "data/raw/practice/learning/practice_learning_panda2.jpg",
    "data/raw/practice/learning/practice_learning_horse.jpg",
    "data/raw/practice/learning/practice_learning_horse2.jpg",
]

PRACTICE_TESTING_IMAGES = [
    ("data/raw/practice/testing/practice_testing_tiger.jpg", "yes"),
    ("data/raw/practice/testing/practice_testing_butterfly.jpg", "no"),
    ("data/raw/practice/testing/practice_testing_chicken.jpg", "no"),
    ("data/raw/practice/testing/practice_testing_fish.jpg", "no"),
]

DEFAULT_ATTENTION_CHECK_PATH = "data/raw/check/check.mp4"


def build_practice_learning_examples() -> list[TrialVideo]:
    return [
        TrialVideo(
            video_id=path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
            video_path=path,
            verb="practice_category",
            class_name="practice",
            phase="practice_learning",
        )
        for path in PRACTICE_LEARNING_IMAGES
    ]


def build_practice_testing_examples() -> list[TrialVideo]:
    return [
        TrialVideo(
            video_id=path.rsplit("/", 1)[-1].rsplit(".", 1)[0],
            video_path=path,
            verb="practice_category",
            class_name="practice",
            phase="practice_testing",
            label=label,
        )
        for path, label in PRACTICE_TESTING_IMAGES
    ]
