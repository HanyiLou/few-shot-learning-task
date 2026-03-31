# few-shot-learning-task
This project runs a learning-only VLM category-membership task based on action videos.

The current task definition is:
- no practice phase
- no review phase
- no attention check
- one session = one `target_class + 3 learning verbs` combination
- one trial = `6 learning videos -> 1 query video -> yes/no judgment`

## Project Structure

- [configs/task_fewshot.yaml](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/configs/task_fewshot.yaml)
  Main task configuration. This is the file you will usually edit first.
- [data/metadata/video_index_loop_2p1s.csv](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/data/metadata/video_index_loop_2p1s.csv)
  Canonical video index.
- [data/trials/learning_only/trials.jsonl](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/data/trials/learning_only/trials.jsonl)
  Current batch of generated trials.
- [data/trials/learning_only/session_state.json](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/data/trials/learning_only/session_state.json)
  Persistent traversal state for session enumeration.
- [outputs/prepared_inputs.jsonl](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/outputs/prepared_inputs.jsonl)
  Prepared multimodal inputs sent to the VLM.
- [outputs/predictions.jsonl](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/outputs/predictions.jsonl)
  Raw per-trial VLM predictions.
- [outputs/session_events.csv](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/outputs/session_events.csv)
  Human-style long-format session table.

## Session Logic

Each session is uniquely defined by:
- one `target_class`
- one 3-verb learning combination from that class

There are 4 classes, each with 5 verbs. For each class, the pipeline enumerates all `C(5, 3) = 10` learning-verb combinations.

So the full session space is:
- `4 x 10 = 40 sessions`

The pipeline traverses these sessions in a fixed order and keeps the current cursor in `session_state.json`.

This means:
- you can generate only the next `N` sessions now
- later runs will continue from where you stopped
- sessions are not repeated unless you reset the state

## Main Configuration

Edit [configs/task_fewshot.yaml](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/configs/task_fewshot.yaml) to control the default workflow.

Important fields:

```yaml
seed: 13
num_sessions: 1
state_path: data/trials/learning_only/session_state.json
video_root: data/processed/video_loop_2p1s
index_path: data/metadata/video_index_loop_2p1s.csv
trials_path: data/trials/learning_only/trials.jsonl
prepared_inputs_path: outputs/prepared_inputs.jsonl
predictions_path: outputs/predictions.jsonl
session_events_path: outputs/session_events.csv
```

The most frequently edited field is:

```yaml
num_sessions: 1
```

If you want to process 3 new sessions in the next run, change it to:

```yaml
num_sessions: 3
```

## Daily Workflow

### 1. Generate the next batch of sessions

```bash
PYTHONPATH=. python3 src/main/generate_trials.py
```

This command:
- reads `configs/task_fewshot.yaml`
- reads the current session cursor from `session_state.json`
- generates the next `num_sessions` sessions
- writes the resulting trials to `data/trials/learning_only/trials.jsonl`
- advances the session cursor

### 2. Run the VLM

```bash
PYTHONPATH=. python3 src/main/run_prompt_baseline.py
```

This command:
- reads the current trials file
- builds the multimodal prompt inputs
- runs DashScope VLM inference
- writes `outputs/prepared_inputs.jsonl`
- writes `outputs/predictions.jsonl`
- writes `outputs/session_events.csv`

### 3. Evaluate predictions

```bash
PYTHONPATH=. python3 src/main/evaluate_predictions.py
```

This computes:
- accuracy
- hit rate
- false alarm rate
- d-prime

## Resetting Session Traversal

If you want to restart traversal from the beginning of the full 40-session sequence, run:

```bash
PYTHONPATH=. python3 src/main/generate_trials.py --reset-state
```

This resets the session cursor and then generates the next batch according to `num_sessions`.

## Output Formats

### `trials.jsonl`

This is the generated experimental material for the current session batch.

Each trial contains:
- one shared learning block of 6 videos within its session
- one query video
- metadata including:
  - `session_num`
  - `session_id`
  - `target_class`
  - `learning_verbs`
  - `query_class`
  - `query_verb`
  - `gold_label`

### `predictions.jsonl`

Each line stores one testing trial result:
- `trial_id`
- `query_video_path`
- `gold_label`
- `predicted_label`
- `raw_response`
- `confidence`
- `backend`
- `latency_ms`

### `session_events.csv`

This file is designed to mirror the human participant data format in session mode.

Each session is stored as:
- 6 `learning` rows
- followed by its `testing` rows

Columns:
- `num`
- `session_id`
- `trialNum`
- `phase`
- `verb`
- `file`
- `class_name`
- `verb_class`
- `condition`
- `rt`
- `response`
- `gold_label`
- `correct`
- `playCounts`
- `trial_id`

Notes:
- `condition` is retained
- `rt` is stored in milliseconds
- learning rows have empty `response`, `gold_label`, and `correct`
- testing rows contain VLM outputs

## Core Scripts

- [src/main/build_index.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/build_index.py)
  Build the canonical video index.
- [src/main/generate_trials.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/generate_trials.py)
  Generate the next batch of sessions.
- [src/main/run_prompt_baseline.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/run_prompt_baseline.py)
  Run DashScope VLM inference and export outputs.
- [src/main/export_session_events.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/export_session_events.py)
  Export human-style session tables.
- [src/main/evaluate_predictions.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/evaluate_predictions.py)
  Compute evaluation metrics.

## Quick Start

If you just want the minimal routine:

1. Edit `num_sessions` in [configs/task_fewshot.yaml](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/configs/task_fewshot.yaml)
2. Run:

```bash
PYTHONPATH=. python3 src/main/generate_trials.py
PYTHONPATH=. python3 src/main/run_prompt_baseline.py
PYTHONPATH=. python3 src/main/evaluate_predictions.py
```
