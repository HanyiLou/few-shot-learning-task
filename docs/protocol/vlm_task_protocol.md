# VLM Task Protocol

This pipeline uses a learning-only category-membership procedure instead of direct verb naming.

Per trial:

1. Learning phase: 6 videos from one target class, sampled as 3 verbs x 2 videos.
2. Testing phase: one query video.
3. Model output: `yes` or `no`, indicating whether the query belongs to the same hidden category learned in the learning phase.

Per session:

1. One session is defined by a unique `target_class + 3 learning verbs` combination.
2. The pipeline enumerates all such sessions without repetition.
3. You can generate only the next `N` sessions in one run and resume later from the saved session state.

Current baseline flow:

1. Build a CSV index from `data/processed/video_loop_2p1s/`.
2. Generate learning-only trial JSONL for the next requested batch of sessions.
3. Convert each trial into a multi-video `messages` payload.
4. Run the DashScope VLM runner on each trial.
5. Export a human-style session events CSV with one learning block followed by testing rows.
6. Evaluate yes/no accuracy, hit rate, false-alarm rate, and d-prime.
