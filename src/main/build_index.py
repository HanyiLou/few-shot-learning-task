from __future__ import annotations

import argparse
from pathlib import Path

from src.data.index_builder import build_video_index, write_video_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a video index CSV from raw video files.")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/processed/video_loop_2p1s"),
        help="Directory that contains class_* video folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/video_index_loop_2p1s.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    records = build_video_index(args.video_root)
    write_video_index(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
