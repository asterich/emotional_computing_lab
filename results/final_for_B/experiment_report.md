# 基于多模态大模型的开放词汇情感识别实验报告

## 1. 实验概述

本实验基于 MELD 数据集，使用 `qwen-vl-plus` 通过 Qwen/DashScope Batch API 完成多模态情感识别。实验目标是比较闭集情感分类与开放词汇情绪生成加标签映射的表现，并通过模态扰动观察文本、上下文和视频关键帧对预测结果的影响。

当前实现使用的模态为：

- 文本：当前 utterance 与 speaker；
- 上下文：同一 split、同一 dialogue 内当前 utterance 之前最多 3 轮对话；
- 图像：从 utterance 视频中抽取 25%、50%、75% 三个位置的关键帧。

MELD 原始数据包含音频，但本轮实验没有使用音频。

## 2. 数据与处理

数据来源为真实 MELD Raw 数据，已解压到 `data/raw/MELD.Raw/`。样本抽取规模为 300 条，并尽量平衡 7 类 MELD 标准情绪标签。

样本分布：

| split | 样本数 |
|---|---:|
| train | 214 |
| test | 55 |
| dev | 31 |

标签分布：

| gold_label | 样本数 |
|---|---:|
| neutral | 44 |
| joy | 44 |
| anger | 43 |
| sadness | 43 |
| fear | 42 |
| surprise | 42 |
| disgust | 42 |

抽帧结果：

| 范围 | 视频数/样本数 | 成功数 | 帧数 | 说明 |
|---|---:|---:|---:|---|
| 实验样本 | 300 | 300 | 899 | 写入 `frames/extracted_frames/` |
| MELD 全量真实视频 | 13848 | 13847 | 41503 | 写入 `frames/all_extracted_frames/` |

全量视频中只有 `train/dia125_utt3.mp4` 无法被 OpenCV 解码。解压产生的 `._*.mp4` 元数据文件已从视频发现逻辑中过滤。

## 3. 实验设置

本轮运行信息：

| 项目 | 值 |
|---|---|
| run_id | `20260531_150537` |
| model | `qwen-vl-plus` |
| 主实验 batch_id | `batch_051d7edb-6af8-49f8-af65-ac4bf58cadaf` |
| 主实验 Batch 请求数 | 1500 |
| 主实验 Batch 完成数 | 1500 |
| 主实验 Batch 失败数 | 0 |
| LLM mapping 候选数 | 217 |
| LLM mapping 最终回写数 | 217 |
| 三策略 LLM Mapping 请求数 | 1200 |
| 三策略结果行数 | 3600 |
| 图像 data URL 数 | 2697 |

实验包含三部分：

| 实验 | 条件 | 输出 |
|---|---|---|
| 闭集分类 | text + context + image | 直接输出 7 类 MELD 标签 |
| 开放词汇 | text + context + image | 输出 1-3 个细粒度情绪词，再映射到 7 类标签 |
| 模态扰动 | text-only / text + context / text + context + image | 比较开放词汇结果的标签漂移和情绪词漂移 |

开放词汇映射采用扩展词典与真实 LLM fallback。流程是先用词典和词形规则完成高置信映射；如果开放情绪词无法映射，则提交第二阶段 Qwen Batch 请求，让 LLM 在 7 个 MELD 标签中选择最接近的粗标签。审计时还发现原始输出中有重复 JSON 字段和截断式 JSON，已增强解析逻辑：严格 JSON 失败时从 raw text 中提取首个 `free_emotions`、`confidence` 和 `reason`，避免把有效回答误判为空。

## 4. 结果审计

重跑与修复后，最终结果通过以下检查：

| 检查项 | 结果 |
|---|---:|
| `sampled_data.csv` 行数 | 300 |
| `closed_set_results.csv` 行数 | 300 |
| `open_vocab_results.csv` 行数 | 300 |
| `mapping_inputs.csv` 行数 | 300 |
| `perturbation_results.csv` 行数 | 900 |
| 原始 batch 输出行数 | 1500 |
| closed/open/perturbation 空 raw output | 0 |
| closed 非法标签 | 0 |
| open 非法映射标签 | 0 |
| perturbation 非法映射标签 | 0 |
| open 空开放情绪 | 0 |
| perturbation 空开放情绪 | 0 |
| 剩余 `heuristic_llm_fallback` | 0 |
| `is_demo != false` | 0 |

审计中发现并修复了两个问题：

1. 上下文构造最初只按 `dialogue_id` 分组，而 MELD 的 train/dev/test 会重复使用同一 `Dialogue_ID`，导致跨 split 混入上下文。已修复为 `source_split + dialogue_id` 分组，并重跑完整 batch。
2. 部分开放词汇 raw output 不是严格 JSON，但含有有效 `free_emotions`。已增强解析逻辑，并基于同一 batch 输出重新解析。
3. 原 `heuristic_llm_fallback` 只是本地关键词启发式，不是真实 LLM 调用。已改为第二阶段 Qwen Batch 语义映射，并补跑所有 unresolved 样本。

## 5. 主实验结果

总体性能：

| 方法 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Closed-set Direct Classification | 0.4733 | 0.4527 | 0.4528 |
| Open-vocab + Dictionary Mapping | 0.3367 | 0.3480 | 0.3485 |
| Open-vocab + LLM Mapping | 0.4600 | 0.4437 | 0.4448 |
| Open-vocab + Hybrid Mapping | 0.4300 | 0.4101 | 0.4111 |

闭集分类各类表现：

| 类别 | Support | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| anger | 43 | 21 | 0.429 | 0.488 | 0.457 |
| disgust | 42 | 7 | 0.700 | 0.167 | 0.269 |
| sadness | 43 | 17 | 0.654 | 0.395 | 0.493 |
| joy | 44 | 35 | 0.432 | 0.795 | 0.560 |
| neutral | 44 | 15 | 0.349 | 0.341 | 0.345 |
| surprise | 42 | 32 | 0.464 | 0.762 | 0.577 |
| fear | 42 | 15 | 0.682 | 0.357 | 0.469 |

开放词汇映射各类表现：

| 类别 | Support | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| anger | 43 | 17 | 0.354 | 0.395 | 0.374 |
| disgust | 42 | 5 | 0.625 | 0.119 | 0.200 |
| sadness | 43 | 19 | 0.594 | 0.442 | 0.507 |
| joy | 44 | 25 | 0.463 | 0.568 | 0.510 |
| neutral | 44 | 14 | 0.538 | 0.318 | 0.400 |
| surprise | 42 | 32 | 0.348 | 0.762 | 0.478 |
| fear | 42 | 17 | 0.425 | 0.405 | 0.415 |

闭集方法仍然是最稳定的直接分类 baseline。三组开放词汇映射中，LLM Mapping 的 Accuracy 最高，Hybrid Mapping 覆盖率最高，Dictionary Mapping 的覆盖率和准确率最低。这说明开放词汇方法的瓶颈主要在标签映射阶段，而不是模型不能生成细粒度情绪词。

## 6. 开放词汇分析

开放词汇统计：

| 指标 | 值 |
|---|---:|
| 总情绪词数 | 604 |
| 去重情绪词数 | 173 |
| 平均每样本情绪词数 | 2.01 |
| Diversity Ratio | 0.2864 |

高频开放情绪词：

| 情绪词 | 频次 |
|---|---:|
| surprise | 38 |
| frustration | 29 |
| confusion | 25 |
| curious | 18 |
| concern | 14 |
| annoyance | 14 |
| shock | 13 |
| curiosity | 12 |
| amused | 12 |
| apologetic | 12 |
| playful | 11 |
| excitement | 10 |
| hesitant | 10 |
| uncertainty | 9 |
| disappointment | 9 |

当前 Hybrid 输出中的映射方法分布：

| mapping_method | open_vocab 样本数 | perturbation 样本数 |
|---|---:|---:|
| hybrid_dictionary_plus_fallback | 121 | 365 |
| dictionary_tie_break | 73 | 206 |
| dictionary | 56 | 157 |
| llm_mapping_fallback | 48 | 169 |
| hybrid_tie_break | 2 | 3 |

开放词汇输出提供了比闭集标签更细的情绪描述，例如 `confusion`、`frustration`、`curiosity`、`apologetic`、`hesitant` 等。但这些细粒度词到 MELD 粗标签的映射存在天然歧义，例如 `confusion` 可对应 surprise，也可能对应 fear 或 neutral；这是开放词汇方法低于闭集分类的主要原因之一。

## 7. Dictionary / LLM / Hybrid 映射对比

主开放词汇实验，即 `text + context + image` 条件下的 300 条 open-vocab 结果：

| 映射策略 | 样本数 | 覆盖数 | 覆盖率 | Accuracy | Mapping Error Rate | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dictionary | 300 | 245 | 0.8167 | 0.3367 | 0.6633 | 0.3480 | 0.3485 |
| LLM | 300 | 285 | 0.9500 | 0.4600 | 0.5400 | 0.4437 | 0.4448 |
| Hybrid | 300 | 300 | 1.0000 | 0.4300 | 0.5700 | 0.4101 | 0.4111 |

三种策略的含义：

- Dictionary：只使用人工词典精确匹配开放情绪词，未覆盖样本记为 `unmapped`。
- LLM：所有样本都交给 qwen-vl-plus 做 7 类标签语义映射；模型若未按 7 类输出，则记为未覆盖。
- Hybrid：优先使用 Dictionary；Dictionary 未覆盖时使用 LLM Mapping。

结论：

- Dictionary 覆盖率为 81.67%，说明开放词汇输出中仍有不少细粒度词超出人工词典。
- LLM 覆盖率为 95.00%，Accuracy 为 46.00%，是三组开放词汇映射中分类性能最高的策略。
- Hybrid 覆盖率达到 100.00%，但 Accuracy 低于纯 LLM，原因是部分 Dictionary 覆盖项虽然可映射，但语义上并不一定比 LLM 更准确。

扰动实验中的三策略结果：

| 输入条件 | 映射策略 | 覆盖率 | Accuracy | Mapping Error Rate | Macro-F1 |
|---|---|---:|---:|---:|---:|
| text-only | Dictionary | 0.7533 | 0.3100 | 0.6900 | 0.3237 |
| text-only | LLM | 0.9633 | 0.4300 | 0.5700 | 0.4151 |
| text-only | Hybrid | 0.9967 | 0.4067 | 0.5933 | 0.3849 |
| text + context | Dictionary | 0.7700 | 0.3100 | 0.6900 | 0.3266 |
| text + context | LLM | 0.9567 | 0.4600 | 0.5400 | 0.4494 |
| text + context | Hybrid | 0.9933 | 0.4200 | 0.5800 | 0.4017 |
| text + context + image | Dictionary | 0.8167 | 0.3400 | 0.6600 | 0.3541 |
| text + context + image | LLM | 0.9433 | 0.4633 | 0.5367 | 0.4509 |
| text + context + image | Hybrid | 0.9967 | 0.4333 | 0.5667 | 0.4160 |

## 8. 模态扰动分析

扰动实验性能：

| 输入条件 | 样本数 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| text-only | 300 | 0.4100 | 0.3882 | 0.3884 |
| text + context | 300 | 0.4233 | 0.4067 | 0.4071 |
| text + context + image | 300 | 0.4333 | 0.4184 | 0.4189 |

相对 `text + context + image` 的漂移：

| 对比条件 | 可比较样本数 | Label Shift Rate | Emotion Word Shift Rate |
|---|---:|---:|---:|
| text + context | 300 | 0.1967 | 0.6500 |
| text-only | 300 | 0.2700 | 0.7633 |

结果显示，加入图像帧后开放词汇映射性能最高；移除图像或上下文会显著改变生成的细粒度情绪词。标签漂移率低于情绪词漂移率，说明模型生成的细粒度表达对模态变化更敏感，但经过粗标签映射后部分差异被压缩。

## 9. 混淆矩阵

闭集分类混淆矩阵：

| Gold \ Pred | anger | disgust | sadness | joy | neutral | surprise | fear |
|---|---:|---:|---:|---:|---:|---:|---:|
| anger | 21 | 0 | 2 | 5 | 6 | 8 | 1 |
| disgust | 15 | 7 | 1 | 6 | 7 | 6 | 0 |
| sadness | 2 | 2 | 17 | 8 | 5 | 5 | 4 |
| joy | 3 | 0 | 0 | 35 | 2 | 4 | 0 |
| neutral | 1 | 0 | 4 | 17 | 15 | 6 | 1 |
| surprise | 2 | 0 | 0 | 5 | 2 | 32 | 1 |
| fear | 5 | 1 | 2 | 5 | 6 | 8 | 15 |

开放词汇映射混淆矩阵：

| Gold \ Pred | anger | disgust | sadness | joy | neutral | surprise | fear |
|---|---:|---:|---:|---:|---:|---:|---:|
| anger | 17 | 1 | 1 | 5 | 3 | 14 | 2 |
| disgust | 15 | 5 | 2 | 4 | 4 | 11 | 1 |
| sadness | 5 | 0 | 19 | 6 | 0 | 5 | 8 |
| joy | 4 | 0 | 3 | 25 | 2 | 9 | 1 |
| neutral | 2 | 1 | 2 | 8 | 14 | 10 | 7 |
| surprise | 1 | 0 | 1 | 3 | 1 | 32 | 4 |
| fear | 4 | 1 | 4 | 3 | 2 | 11 | 17 |

主要混淆现象：

- `disgust` 容易被预测为 `anger` 或 `surprise`，说明 MELD 中厌恶类语义和强烈负面/惊讶表达边界较模糊；
- `joy` 和 `surprise` 是闭集方法表现最好的类别；
- 开放词汇方法倾向生成 `surprise/confusion/curiosity` 相关词，导致 `surprise` 预测偏多；
- `fear` 和 `sadness` 都包含负向、低唤醒表达，部分样本容易互相混淆。

## 10. 结论与局限

本轮实验完成了真实 MELD 数据下载、视频关键帧抽取、qwen-vl-plus 主实验 Batch 推理、结果解析、第二阶段 LLM mapping fallback、Dictionary / LLM / Hybrid 三组映射对照和扰动分析。闭集分类在直接分类指标上仍然最好；开放词汇三策略中，LLM Mapping 的 Accuracy 最高，Hybrid Mapping 的覆盖率最高。

局限：

1. 本轮未使用 MELD 音频，因此语音语调、停顿、音量等情绪线索没有进入模型；
2. LLM Mapping 仍有少量样本没有严格输出 7 类标签，因此覆盖率未达到 100%；
3. 样本规模为 300，适合课程项目和快速分析，但不足以代表完整 MELD 分布；
4. 视频只抽取 3 个关键帧，无法覆盖完整动作变化。

后续改进方向：

1. 加入音频特征或音频转写特征；
2. 对 LLM Mapping 的无效输出增加自动重试或强制修复策略；
3. 增大样本数并固定随机种子做多轮评估；
4. 对错误样本进行人工归因，区分标签噪声、语境依赖、视觉误导和映射错误。

## 11. 文件位置

核心交付文件：

```text
results/final_for_B/
├── sampled_data.csv
├── closed_set_results.csv
├── open_vocab_results.csv
├── mapping_inputs.csv
├── perturbation_results.csv
├── mapping_strategy_results.csv
├── mapping_strategy_metrics.csv
└── result_field_description.md
```

其他审计文件：

```text
logs/frame_extraction_summary.json
logs/all_frame_extraction_summary.json
logs/pipeline_state.json
results/raw_outputs/20260531_150537_batch_output.jsonl
results/raw_outputs/20260531_150537_llm_mapping_output.jsonl
results/raw_outputs/20260531_150537_mapping_strategy_llm_output.jsonl
```
