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
| run_id | `20260528_155301` |
| model | `qwen-vl-plus` |
| batch_id | `batch_41b545da-2beb-483a-aca8-0e86f5ccc10c` |
| Batch 请求数 | 1500 |
| Batch 完成数 | 1500 |
| Batch 失败数 | 0 |
| 图像 data URL 数 | 2697 |

实验包含三部分：

| 实验 | 条件 | 输出 |
|---|---|---|
| 闭集分类 | text + context + image | 直接输出 7 类 MELD 标签 |
| 开放词汇 | text + context + image | 输出 1-3 个细粒度情绪词，再映射到 7 类标签 |
| 模态扰动 | text-only / text + context / text + context + image | 比较开放词汇结果的标签漂移和情绪词漂移 |

开放词汇映射采用扩展词典与启发式 fallback。审计时发现原始输出中有重复 JSON 字段和截断式 JSON，已增强解析逻辑：严格 JSON 失败时从 raw text 中提取首个 `free_emotions`、`confidence` 和 `reason`，避免把有效回答误判为空。

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
| `is_demo != false` | 0 |

审计中发现并修复了两个问题：

1. 上下文构造最初只按 `dialogue_id` 分组，而 MELD 的 train/dev/test 会重复使用同一 `Dialogue_ID`，导致跨 split 混入上下文。已修复为 `source_split + dialogue_id` 分组，并重跑完整 batch。
2. 部分开放词汇 raw output 不是严格 JSON，但含有有效 `free_emotions`。已增强解析逻辑，并基于同一 batch 输出重新解析。

## 5. 主实验结果

总体性能：

| 方法 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Closed-set Direct Classification | 0.4633 | 0.4384 | 0.4391 |
| Open-vocab + Mapping | 0.3933 | 0.3752 | 0.3759 |

闭集分类各类表现：

| 类别 | Support | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| anger | 43 | 18 | 0.375 | 0.419 | 0.396 |
| disgust | 42 | 6 | 0.600 | 0.143 | 0.231 |
| sadness | 43 | 16 | 0.593 | 0.372 | 0.457 |
| joy | 44 | 35 | 0.461 | 0.795 | 0.583 |
| neutral | 44 | 18 | 0.391 | 0.409 | 0.400 |
| surprise | 42 | 32 | 0.457 | 0.762 | 0.571 |
| fear | 42 | 14 | 0.609 | 0.333 | 0.431 |

开放词汇映射各类表现：

| 类别 | Support | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| anger | 43 | 17 | 0.362 | 0.395 | 0.378 |
| disgust | 42 | 4 | 1.000 | 0.095 | 0.174 |
| sadness | 43 | 18 | 0.667 | 0.419 | 0.514 |
| joy | 44 | 17 | 0.436 | 0.386 | 0.410 |
| neutral | 44 | 18 | 0.333 | 0.409 | 0.367 |
| surprise | 42 | 32 | 0.333 | 0.762 | 0.464 |
| fear | 42 | 12 | 0.364 | 0.286 | 0.320 |

闭集方法整体优于开放词汇映射方法，尤其在 Accuracy 和 Macro-F1 上更稳定。开放词汇方法的优势不体现在直接分类性能，而体现在细粒度情绪表达上；其最终分类表现受到标签映射质量影响明显。

## 6. 开放词汇分析

开放词汇统计：

| 指标 | 值 |
|---|---:|
| 总情绪词数 | 601 |
| 去重情绪词数 | 171 |
| 平均每样本情绪词数 | 2.00 |
| Diversity Ratio | 0.2845 |

高频开放情绪词：

| 情绪词 | 频次 |
|---|---:|
| surprise | 40 |
| confusion | 26 |
| frustration | 26 |
| curious | 16 |
| curiosity | 15 |
| concern | 13 |
| apologetic | 13 |
| shock | 13 |
| annoyance | 12 |
| amused | 12 |
| playful | 10 |
| hesitant | 10 |
| concerned | 9 |
| excitement | 9 |
| uncertainty | 8 |

映射方法分布：

| mapping_method | 样本数 |
|---|---:|
| hybrid_dictionary_plus_fallback | 122 |
| dictionary_tie_break | 75 |
| dictionary | 53 |
| heuristic_llm_fallback | 49 |
| hybrid_tie_break | 1 |

开放词汇输出提供了比闭集标签更细的情绪描述，例如 `confusion`、`frustration`、`curiosity`、`apologetic`、`hesitant` 等。但这些细粒度词到 MELD 粗标签的映射存在天然歧义，例如 `confusion` 可对应 surprise，也可能对应 fear 或 neutral；这是开放词汇方法低于闭集分类的主要原因之一。

## 7. 模态扰动分析

扰动实验性能：

| 输入条件 | 样本数 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| text-only | 300 | 0.3733 | 0.3461 | 0.3471 |
| text + context | 300 | 0.3567 | 0.3322 | 0.3328 |
| text + context + image | 300 | 0.3900 | 0.3686 | 0.3692 |

相对 `text + context + image` 的漂移：

| 对比条件 | 可比较样本数 | Label Shift Rate | Emotion Word Shift Rate |
|---|---:|---:|---:|
| text + context | 300 | 0.2167 | 0.6433 |
| text-only | 300 | 0.2733 | 0.7800 |

结果显示，加入图像帧后开放词汇映射性能最高；移除图像或上下文会显著改变生成的细粒度情绪词。标签漂移率低于情绪词漂移率，说明模型生成的细粒度表达对模态变化更敏感，但经过粗标签映射后部分差异被压缩。

## 8. 混淆矩阵

闭集分类混淆矩阵：

| Gold \ Pred | anger | disgust | sadness | joy | neutral | surprise | fear |
|---|---:|---:|---:|---:|---:|---:|---:|
| anger | 18 | 1 | 2 | 5 | 6 | 9 | 2 |
| disgust | 16 | 6 | 2 | 6 | 6 | 5 | 1 |
| sadness | 2 | 2 | 16 | 8 | 6 | 5 | 4 |
| joy | 3 | 0 | 0 | 35 | 2 | 4 | 0 |
| neutral | 2 | 0 | 4 | 14 | 18 | 5 | 1 |
| surprise | 2 | 0 | 0 | 5 | 2 | 32 | 1 |
| fear | 5 | 1 | 3 | 3 | 6 | 10 | 14 |

开放词汇映射混淆矩阵：

| Gold \ Pred | anger | disgust | sadness | joy | neutral | surprise | fear |
|---|---:|---:|---:|---:|---:|---:|---:|
| anger | 17 | 0 | 1 | 3 | 7 | 13 | 2 |
| disgust | 13 | 4 | 1 | 4 | 8 | 11 | 1 |
| sadness | 4 | 0 | 18 | 3 | 5 | 6 | 7 |
| joy | 5 | 0 | 3 | 17 | 8 | 10 | 1 |
| neutral | 3 | 0 | 0 | 6 | 18 | 10 | 7 |
| surprise | 1 | 0 | 1 | 3 | 2 | 32 | 3 |
| fear | 4 | 0 | 3 | 3 | 6 | 14 | 12 |

主要混淆现象：

- `disgust` 容易被预测为 `anger` 或 `surprise`，说明 MELD 中厌恶类语义和强烈负面/惊讶表达边界较模糊；
- `joy` 和 `surprise` 是闭集方法表现最好的类别；
- 开放词汇方法倾向生成 `surprise/confusion/curiosity` 相关词，导致 `surprise` 预测偏多；
- `fear` 和 `sadness` 都包含负向、低唤醒表达，部分样本容易互相混淆。

## 9. 结论与局限

本轮实验完成了真实 MELD 数据下载、视频关键帧抽取、qwen-vl-plus Batch 推理、结果解析、开放词汇映射和扰动分析。闭集分类在直接分类指标上优于开放词汇映射；开放词汇方法提供更丰富的细粒度情绪词，但需要更强的标签映射模块才能稳定提升粗标签分类性能。

局限：

1. 本轮未使用 MELD 音频，因此语音语调、停顿、音量等情绪线索没有进入模型；
2. 开放词汇到 MELD 标签的映射仍以词典和启发式规则为主，尚未单独调用模型做 LLM Mapping；
3. 样本规模为 300，适合课程项目和快速分析，但不足以代表完整 MELD 分布；
4. 视频只抽取 3 个关键帧，无法覆盖完整动作变化。

后续改进方向：

1. 加入音频特征或音频转写特征；
2. 将开放词汇映射单独设计为 Dictionary / LLM / Hybrid 三组可比较实验；
3. 增大样本数并固定随机种子做多轮评估；
4. 对错误样本进行人工归因，区分标签噪声、语境依赖、视觉误导和映射错误。

## 10. 文件位置

核心交付文件：

```text
results/final_for_B/
├── sampled_data.csv
├── closed_set_results.csv
├── open_vocab_results.csv
├── mapping_inputs.csv
├── perturbation_results.csv
└── result_field_description.md
```

其他审计文件：

```text
logs/frame_extraction_summary.json
logs/all_frame_extraction_summary.json
logs/pipeline_state.json
results/raw_outputs/20260528_155301_batch_output.jsonl
```
