# VLM Learning-Only 流程说明

这个项目目前实现的是一个基于动作视频的 `learning-only` VLM 类别成员判断任务。

当前任务定义：
- 不包含 `practice phase`
- 不包含 `review phase`
- 不包含 `attention check`
- 一个 `session` = 一个唯一的 `target_class + 3 个 learning verbs` 组合
- 一个 `trial` = `6 个 learning videos -> 1 个 query video -> yes/no 判断`

## 项目结构

- [configs/task_fewshot.yaml](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/configs/task_fewshot.yaml)
  主配置文件。平时最常改的是这个文件。
- [data/metadata/video_index_loop_2p1s.csv](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/data/metadata/video_index_loop_2p1s.csv)
  当前使用的标准视频索引。
- [data/trials/learning_only/trials.jsonl](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/data/trials/learning_only/trials.jsonl)
  当前这一批 session 生成出来的试次材料。
- [data/trials/learning_only/session_state.json](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/data/trials/learning_only/session_state.json)
  session 遍历状态文件，用来记录现在跑到了第几个 session。
- [outputs/prepared_inputs.jsonl](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/outputs/prepared_inputs.jsonl)
  发给 VLM 的多模态输入。
- [outputs/predictions.jsonl](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/outputs/predictions.jsonl)
  VLM 对每个 testing trial 的原始预测结果。
- [outputs/session_events.csv](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/outputs/session_events.csv)
  按人类被试风格整理出来的 long-format session 表。

## Session 逻辑

每个 session 由下面两部分唯一确定：
- 一个 `target_class`
- 该 class 下一个 3-verb 的 learning 组合

当前每个 class 都有 5 个 verb，因此：
- 每个 class 的 3-verb 组合数是 `C(5,3)=10`
- 总共有 4 个 class
- 所以完整 session 空间一共是 `40` 个 session

这些 session 会按固定顺序遍历，并把当前位置记录在 `session_state.json` 中。

这意味着：
- 你可以一次只生成接下来的 `N` 个 session
- 下次继续运行时，会自动从上次停下的位置继续
- 除非手动 reset，否则不会重复生成已经遍历过的 session

## 主配置文件

请编辑 [configs/task_fewshot.yaml](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/configs/task_fewshot.yaml)。

关键字段如下：

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

最常改的是：

```yaml
num_sessions: 1
```

例如你想下一次从当前进度继续生成 5 个 session，可以改成：

```yaml
num_sessions: 5
```

## 日常工作流

### 1. 生成接下来的 session

```bash
PYTHONPATH=. python3 src/main/generate_trials.py
```

这一步会：
- 读取 `configs/task_fewshot.yaml`
- 从 `session_state.json` 读取当前游标
- 生成接下来的 `num_sessions` 个 session
- 把结果写入 `data/trials/learning_only/trials.jsonl`
- 自动更新 session 游标

### 2. 运行 VLM

```bash
PYTHONPATH=. python3 src/main/run_prompt_baseline.py
```

这一步会：
- 读取当前的 trials
- 构造多模态 prompt
- 调用 DashScope VLM
- 输出 `outputs/prepared_inputs.jsonl`
- 输出 `outputs/predictions.jsonl`
- 输出 `outputs/session_events.csv`

注意：
- `predictions.jsonl` 现在是追加模式
- `session_events.csv` 现在也是追加模式
- 已有的 `trial_id` / `session_id` 会自动跳过，避免重复写入

### 3. 计算指标

```bash
PYTHONPATH=. python3 src/main/evaluate_predictions.py
```

会输出：
- accuracy
- hit rate
- false alarm rate
- d-prime

## 从头重新开始遍历

如果你想从第一个 session 重新开始：

```bash
PYTHONPATH=. python3 src/main/generate_trials.py --reset-state
```

这会重置 session 游标，然后再按 `num_sessions` 生成新的 session 批次。

如果你还想同时清空旧的累计输出，可以额外执行：

```bash
rm -f outputs/predictions.jsonl outputs/session_events.csv outputs/prepared_inputs.jsonl
```

## 输出文件说明

### `trials.jsonl`

这是当前生成出来的实验材料。

每个 trial 包含：
- 一个共享的 6 视频 learning block
- 一个 query video
- metadata，包括：
  - `session_num`
  - `session_id`
  - `target_class`
  - `learning_verbs`
  - `query_class`
  - `query_verb`
  - `gold_label`

### `predictions.jsonl`

每一行对应一个 testing trial 的预测结果：
- `trial_id`
- `query_video_path`
- `gold_label`
- `predicted_label`
- `raw_response`
- `confidence`
- `backend`
- `latency_ms`

### `session_events.csv`

这个文件模仿人类被试的 session 数据格式。

每个 session 在表中表现为：
- 前 6 行是 `learning`
- 后面接这一 session 的 `testing` 行

列包括：
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

说明：
- `condition` 保留
- `rt` 以毫秒保存
- learning 行的 `response`、`gold_label`、`correct` 为空
- testing 行保存 VLM 的判断结果

## 核心脚本

- [src/main/build_index.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/build_index.py)
  生成标准视频索引。
- [src/main/generate_trials.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/generate_trials.py)
  生成接下来的 session 批次。
- [src/main/run_prompt_baseline.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/run_prompt_baseline.py)
  运行 DashScope VLM，并导出结果。
- [src/main/export_session_events.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/export_session_events.py)
  导出人类风格的 session 表。
- [src/main/evaluate_predictions.py](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/src/main/evaluate_predictions.py)
  计算评估指标。

## 最简使用方式

如果你只想按最短流程使用：

1. 在 [configs/task_fewshot.yaml](/Users/louhanyi/few-shot%20learning%20task/verb_fewshot_project/configs/task_fewshot.yaml) 里设置 `num_sessions`
2. 运行：

```bash
PYTHONPATH=. python3 src/main/generate_trials.py
PYTHONPATH=. python3 src/main/run_prompt_baseline.py
PYTHONPATH=. python3 src/main/evaluate_predictions.py
```
