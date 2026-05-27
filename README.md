# Emotional Computing Lab Pipeline

同学 A 负责的可复现实验 pipeline：读取 MELD 或自动 demo 样本，构造闭集分类、开放词汇生成、模态扰动三类 Qwen Batch 请求，保存原始输出并解析为 B 可直接统计的 CSV。

## Quick Start

```bash
uv run emotional-pipeline run --demo --n 21
```

常用命令：

```bash
uv run emotional-pipeline run --n 300
uv run emotional-pipeline run --demo --n 21 --no-api
uv run emotional-pipeline fetch-batch --poll-seconds 300
uv run emotional-pipeline status
uv run emotional-pipeline clean --artifacts-only
```

`.envrc` 支持：

```bash
export QWEN_API_KEY="..."
export QWEN_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export QWEN_MODEL="qwen-plus"
```

如果 `data/raw/` 下面没有 MELD CSV，pipeline 会在允许 fallback 时生成 `is_demo=true` 的样本和实验输出，保证交付结构完整。放入真实 MELD CSV 后重新运行即可生成真实样本结果。

`run` 会提交 Qwen/DashScope Batch；如果 batch 在轮询时间内还没完成，会先写入 demo fallback 结果，并把 `batch_id` 保存到 `logs/pipeline_state.json`。稍后执行 `uv run emotional-pipeline fetch-batch --poll-seconds 300` 会下载真实 batch 输出并重新生成解析后的 CSV。

## Outputs

交付给 B 的文件固定在：

```text
results/final_for_B/
├── sampled_data.csv
├── closed_set_results.csv
├── open_vocab_results.csv
├── mapping_inputs.csv
├── perturbation_results.csv
└── result_field_description.md
```

原始模型输出保存在：

```text
results/raw_outputs/
├── batch_inputs/
├── closed_set/
├── open_vocab/
└── perturbation/
```

解析后的完整结果保存在：

```text
results/parsed_outputs/
├── closed_set_results.csv
├── open_vocab_results.csv
├── mapping_inputs.csv
├── perturbation_results.csv
└── perturbation_shift_stats.csv
```

## Modalities

当前实现使用 `T` 文本、`C` 对话上下文、`I` 视频关键帧。MELD 包含音频，但本项目 v1 不处理音频。

## Data Layout

把 MELD CSV 和视频放入 `data/raw/`。CSV 字段会自动识别常见 MELD 命名，如 `Dialogue_ID`、`Utterance_ID`、`Speaker`、`Utterance`、`Emotion`。视频若存在，会按 `dia{Dialogue_ID}_utt{Utterance_ID}.mp4` 或 `.avi` 查找并尝试抽帧；没有视频时自动退化为文本 + 上下文。
