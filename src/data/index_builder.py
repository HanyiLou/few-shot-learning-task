from __future__ import annotations

import csv
import re
from pathlib import Path

from src.data.schema import VideoRecord


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FILENAME_PATTERN = re.compile(r"^(class\d+)_(?P<verb>[a-z]+)_(?P<index>\d+)$")


def parse_video_record(video_path: Path, root_dir: Path) -> VideoRecord:
    stem = video_path.stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Unsupported video filename format: {video_path.name}")

    class_name = video_path.parent.name
    if not class_name.startswith("class_"):
        raise ValueError(f"Unsupported class directory format: {video_path.parent}")

    return VideoRecord(
        video_id=stem,
        class_name=class_name,
        verb=match.group("verb"),
        file_path=video_path.as_posix(),
        source_filename=video_path.name,
    )


def build_video_index(video_root: Path) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    for video_path in sorted(video_root.rglob("*")):
        if not video_path.is_file():
            continue
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        records.append(parse_video_record(video_path, video_root))
    return records


def write_video_index(records: list[VideoRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "class_name",
        "verb",
        "file_path",
        "source_filename",
        "split",
        "source",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def load_video_index(index_csv: Path) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    with index_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(VideoRecord(**row))
    return records
