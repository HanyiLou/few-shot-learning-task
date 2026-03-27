from __future__ import annotations

import random
from collections import defaultdict

from src.data.schema import Trial, TrialVideo, VideoRecord
from src.sampling.constraints import (
    DEFAULT_CANDIDATE_LABELS,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_TASK_INSTRUCTION,
    DEFAULT_TASK_TYPE,
)
from src.sampling.materials import (
    DEFAULT_ATTENTION_CHECK_PATH,
    build_practice_learning_examples,
    build_practice_testing_examples,
)


def _group_by_class_and_verb(records: list[VideoRecord]) -> dict[str, dict[str, list[VideoRecord]]]:
    grouped: dict[str, dict[str, list[VideoRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.class_name][record.verb].append(record)
    return {class_name: dict(verb_map) for class_name, verb_map in grouped.items()}


def _to_trial_video(record: VideoRecord, phase: str, label: str | None = None) -> TrialVideo:
    return TrialVideo(
        video_id=record.video_id,
        video_path=record.file_path,
        verb=record.verb,
        class_name=record.class_name,
        phase=phase,
        label=label,
    )


def _pick_records(
    rng: random.Random,
    records: list[VideoRecord],
    n_select: int,
    used_ids: set[str],
) -> list[VideoRecord]:
    available = [record for record in records if record.video_id not in used_ids]
    if len(available) < n_select:
        raise ValueError("Not enough unused records to satisfy sampling request")
    selected = rng.sample(available, n_select)
    for record in selected:
        used_ids.add(record.video_id)
    return selected


def sample_human_procedure_trials(
    records: list[VideoRecord],
    target_class: str = "class_4",
    seed: int = 13,
    include_attention_check: bool = True,
    attention_check_path: str | None = DEFAULT_ATTENTION_CHECK_PATH,
) -> list[Trial]:
    rng = random.Random(seed)
    grouped = _group_by_class_and_verb(records)
    practice_learning_examples = build_practice_learning_examples()
    practice_testing_examples = build_practice_testing_examples()

    if target_class not in grouped:
        raise ValueError(f"Target class {target_class} not found in records")

    used_ids: set[str] = set()
    target_verbs = sorted(grouped[target_class])
    if len(target_verbs) < 5:
        raise ValueError("Expected at least 5 verbs in the target class")

    learning_verbs = rng.sample(target_verbs, 3)
    learning_examples: list[TrialVideo] = []
    for verb in learning_verbs:
        selected = _pick_records(rng, grouped[target_class][verb], 2, used_ids)
        learning_examples.extend(_to_trial_video(record, phase="learning") for record in selected)
    rng.shuffle(learning_examples)

    review_examples: list[TrialVideo] = []
    unused_target_verbs = [verb for verb in target_verbs if verb not in learning_verbs]
    for class_name in sorted(grouped):
        if class_name == target_class:
            selected_verb = rng.choice(unused_target_verbs)
        else:
            selected_verb = rng.choice(sorted(grouped[class_name]))
        selected = _pick_records(rng, grouped[class_name][selected_verb], 1, used_ids)
        review_examples.extend(_to_trial_video(record, phase="review") for record in selected)
    rng.shuffle(review_examples)

    positive_verbs = [verb for verb in target_verbs if verb not in learning_verbs]
    if len(positive_verbs) < 2:
        raise ValueError("Expected at least two held-out target verbs for positive test queries")

    query_examples: list[TrialVideo] = []
    for verb in rng.sample(positive_verbs, 2):
        selected = _pick_records(rng, grouped[target_class][verb], 2, used_ids)
        query_examples.extend(_to_trial_video(record, phase="testing", label="yes") for record in selected)

    for class_name in sorted(grouped):
        if class_name == target_class:
            continue
        negative_verbs = rng.sample(sorted(grouped[class_name]), 2)
        for verb in negative_verbs:
            selected = _pick_records(rng, grouped[class_name][verb], 2, used_ids)
            query_examples.extend(_to_trial_video(record, phase="testing", label="no") for record in selected)

    if include_attention_check and attention_check_path:
        query_examples.append(
            TrialVideo(
                video_id="attention_check",
                video_path=attention_check_path,
                verb="attention_check",
                class_name="attention_check",
                phase="testing",
                label="yes",
            )
        )

    rng.shuffle(query_examples)

    base_metadata = {
        "target_class": target_class,
        "learning_verbs": learning_verbs,
        "review_order": [example.video_id for example in review_examples],
        "seed": seed,
        "human_procedure_aligned": True,
        "attention_check_included": include_attention_check and bool(attention_check_path),
    }

    trials: list[Trial] = []
    for index, query_example in enumerate(query_examples, start=1):
        metadata = dict(base_metadata)
        metadata["query_index"] = index
        metadata["query_class"] = query_example.class_name
        metadata["query_verb"] = query_example.verb
        metadata["gold_label"] = query_example.label
        trials.append(
            Trial(
                trial_id=f"{target_class}_query_{index:02d}_{query_example.video_id}",
                task_type=DEFAULT_TASK_TYPE,
                target_class=target_class,
                practice_learning_examples=list(practice_learning_examples),
                practice_testing_examples=list(practice_testing_examples),
                learning_examples=list(learning_examples),
                review_examples=list(review_examples),
                query_example=query_example,
                candidate_labels=list(DEFAULT_CANDIDATE_LABELS),
                task_instruction=DEFAULT_TASK_INSTRUCTION,
                expected_output_format=dict(DEFAULT_OUTPUT_FORMAT),
                metadata=metadata,
            )
        )
    return trials
