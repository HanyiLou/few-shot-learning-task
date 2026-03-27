# VLM Task Protocol

This pipeline now mirrors the human experiment procedure instead of direct verb naming.

Per trial:

1. Practice phase: 6 learning images followed by 4 test images with known yes/no answers.
2. Learning phase: 6 videos from one target class, sampled as 3 verbs x 2 videos.
3. Review phase: 4 videos, one from each class.
4. Testing phase: one query video.
5. Model output: `yes` or `no`, indicating whether the query belongs to the same hidden category learned in the learning phase.

The project now stores the copied human-practice materials at:

- `data/raw/practice/learning/`
- `data/raw/practice/testing/`
- `data/raw/check/check.mp4`

Default target class matches the currently active human experiment: `class_4`.

Current baseline flow:

1. Build a CSV index from `data/raw/video/`.
2. Generate human-procedure trial JSONL from the index.
3. Convert each trial into a multi-video `messages` payload.
4. Run a mock baseline runner that can later be replaced by a real VLM backend.
5. Evaluate yes/no accuracy, hit rate, false-alarm rate, and d-prime.
