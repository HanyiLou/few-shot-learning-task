from __future__ import annotations

import itertools
import random
from collections import defaultdict

from src.data.schema import Trial, TrialVideo, VideoRecord
from src.sampling.constraints import (
    DEFAULT_CANDIDATE_LABELS,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_TASK_INSTRUCTION,
    DEFAULT_TASK_TYPE,
)


def _group_by_class_and_verb(records: list[VideoRecord]) -> dict[str, dict[str, list[VideoRecord]]]:
    grouped: dict[str, dict[str, list[VideoRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.class_name][record.verb].append(record)
    return {class_name: dict(verb_map) for class_name, verb_map in grouped.items()}


def build_session_specs(records: list[VideoRecord]) -> list[dict[str, object]]:
    grouped = _group_by_class_and_verb(records)
    session_specs: list[dict[str, object]] = []
    session_num = 1
    for target_class in sorted(grouped):
        verbs = sorted(grouped[target_class])
        for learning_verbs in itertools.combinations(verbs, 3):
            session_specs.append(
                {
                    "session_num": session_num,
                    "session_id": f"session_{session_num:03d}_{target_class}_{'_'.join(learning_verbs)}",
                    "target_class": target_class,
                    "learning_verbs": list(learning_verbs),
                }
            )
            session_num += 1
    return session_specs


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


def sample_learning_only_trials(
    records: list[VideoRecord],
    target_class: str = "class_4",
    seed: int = 13,
    learning_verbs: list[str] | None = None,
    session_num: int | None = None,
    session_id: str | None = None,
) -> list[Trial]:
    rng = random.Random(seed)
    grouped = _group_by_class_and_verb(records)

    if target_class not in grouped:
        raise ValueError(f"Target class {target_class} not found in records")

    used_ids: set[str] = set()
    target_verbs = sorted(grouped[target_class])
    if len(target_verbs) < 5:
        raise ValueError("Expected at least 5 verbs in the target class")

    if learning_verbs is None:
        selected_learning_verbs = rng.sample(target_verbs, 3)
    else:
        selected_learning_verbs = list(learning_verbs)
        if len(selected_learning_verbs) != 3:
            raise ValueError("learning_verbs must contain exactly 3 verbs")
        missing_verbs = [verb for verb in selected_learning_verbs if verb not in target_verbs]
        if missing_verbs:
            raise ValueError(f"learning_verbs not found in {target_class}: {missing_verbs}")

    learning_examples: list[TrialVideo] = []
    for verb in selected_learning_verbs:
        selected = _pick_records(rng, grouped[target_class][verb], 2, used_ids)
        learning_examples.extend(_to_trial_video(record, phase="learning") for record in selected)
    rng.shuffle(learning_examples)

    positive_verbs = [verb for verb in target_verbs if verb not in selected_learning_verbs]
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

    rng.shuffle(query_examples)

    base_metadata = {
        "target_class": target_class,
        "learning_verbs": selected_learning_verbs,
        "seed": seed,
        "vlm_learning_only": True,
    }
    if session_num is not None:
        base_metadata["session_num"] = session_num
    if session_id is not None:
        base_metadata["session_id"] = session_id

    trials: list[Trial] = []
    trial_prefix = session_id if session_id is not None else target_class
    for index, query_example in enumerate(query_examples, start=1):
        metadata = dict(base_metadata)
        metadata["query_index"] = index
        metadata["query_class"] = query_example.class_name
        metadata["query_verb"] = query_example.verb
        metadata["gold_label"] = query_example.label
        trials.append(
            Trial(
                trial_id=f"{trial_prefix}_query_{index:02d}_{query_example.video_id}",
                task_type=DEFAULT_TASK_TYPE,
                target_class=target_class,
                learning_examples=list(learning_examples),
                query_example=query_example,
                candidate_labels=list(DEFAULT_CANDIDATE_LABELS),
                task_instruction=DEFAULT_TASK_INSTRUCTION,
                expected_output_format=dict(DEFAULT_OUTPUT_FORMAT),
                metadata=metadata,
            )
        )
    return trials
