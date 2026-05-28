# Prompt 设计与情绪映射执行说明

这份文档用于指导 API 调用、开放情绪词生成、情绪词映射和结果保存。整体目标是：先让多模态大模型根据文本、上下文和视频帧生成开放情绪词，再把这些开放情绪词映射回 MELD 的 7 个标准情绪标签，方便后续计算准确率、F1 和做错误分析。

---

## 1. 整体流程

本项目不需要部署模型，直接调用多模态大模型 API 即可。建议流程如下：

```text
MELD 样本
  ↓
整理 utterance、context、video frame
  ↓
调用多模态大模型 API
  ↓
生成 1-3 个开放情绪词
  ↓
标准化情绪词
  ↓
Dictionary / LLM / Hybrid Mapping
  ↓
得到 MELD 标准标签
  ↓
保存结果，后续评估与分析
```

MELD 标准标签为：

```text
anger, disgust, sadness, joy, neutral, surprise, fear
```

---

## 2. API 调用目标

API 不直接做闭集分类，而是先让模型输出更细粒度的开放情绪词。

例如，不直接让模型从：

```text
anger, sadness, joy, neutral, surprise, fear, disgust
```

中选择一个，而是让模型生成：

```text
annoyed, frustrated, disappointed, relieved, confused, nervous
```

这样可以保留更细的情绪信息，后面再通过映射规则归到 MELD 的标准标签。

---

## 3. 为什么生成 1-3 个开放情绪词

一条对话中的情绪可能不是单一的。比如：

```text
I can't believe you forgot again.
```

它可能同时包含：

```text
annoyed, disappointed, frustrated
```

所以建议生成 1-3 个词，而不是强制只生成 1 个。

但也不建议生成太多，因为词太多会带来噪声。1-3 个是比较平衡的设置。

---

## 4. API Prompt 模板

### 4.1 多模态开放情绪词生成 Prompt

可以用下面这个模板调用 API。

```text
You are an emotion analysis assistant.

Given a dialogue utterance, its conversation context, and visual information from video frames, identify the speaker's fine-grained emotions.

Important requirements:
1. Do not directly choose from fixed MELD labels.
2. Generate 1 to 3 fine-grained emotion words.
3. Emotion words should be concise English words or short phrases.
4. The emotions should reflect the speaker's actual emotional state.
5. Use the dialogue context and visual cues when available.
6. Return JSON only.

Input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}
- Visual information: {image_frames}

Output JSON format:
{
  "free_emotions": ["emotion_1", "emotion_2"],
  "confidence": 0.0,
  "reason": "brief explanation"
}
```

### 4.2 如果只跑文本，不使用视频帧

如果当前样本暂时没有处理视频帧，可以使用文本版本：

```text
You are an emotion analysis assistant.

Given a dialogue utterance and its conversation context, identify the speaker's fine-grained emotions.

Important requirements:
1. Do not directly choose from fixed MELD labels.
2. Generate 1 to 3 fine-grained emotion words.
3. Emotion words should be concise English words or short phrases.
4. The emotions should reflect the speaker's actual emotional state.
5. Use the dialogue context when available.
6. Return JSON only.

Input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}

Output JSON format:
{
  "free_emotions": ["emotion_1", "emotion_2"],
  "confidence": 0.0,
  "reason": "brief explanation"
}
```

---


### 4.3 闭集分类 Prompt

闭集分类 Prompt 用于做 baseline。它和开放词汇方案不同，不要求模型生成 `annoyed`、`frustrated` 这类细粒度情绪词，而是直接从 MELD 的 7 个标准标签中选择一个。

闭集分类的作用是：

```text
给主实验提供对照基线，方便比较：
直接闭集分类
vs.
开放词汇生成 + 标签映射
```

如果时间允许，建议保留闭集分类实验。这样后续可以比较两种方法的 Accuracy、Macro-F1、Weighted-F1 和混淆矩阵。

多模态闭集分类 Prompt：

```text
You are an emotion classification assistant.

Given a dialogue utterance, its conversation context, and visual information from video frames, classify the speaker's emotion into exactly one MELD label.

Allowed labels:
anger, disgust, sadness, joy, neutral, surprise, fear

Important requirements:
1. Choose exactly one label from the allowed labels.
2. Use the utterance as the primary evidence.
3. Use conversation context and visual cues when available.
4. Do not generate fine-grained emotion words.
5. Return JSON only.

Input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}
- Visual information: {image_frames}

Output JSON format:
{
  "predicted_label": "one of the allowed labels",
  "confidence": 0.0,
  "reason": "brief explanation"
}
```

如果暂时不使用视频帧，可以使用文本版闭集分类 Prompt：

```text
You are an emotion classification assistant.

Given a dialogue utterance and its conversation context, classify the speaker's emotion into exactly one MELD label.

Allowed labels:
anger, disgust, sadness, joy, neutral, surprise, fear

Important requirements:
1. Choose exactly one label from the allowed labels.
2. Use the utterance as the primary evidence.
3. Use conversation context when available.
4. Do not generate fine-grained emotion words.
5. Return JSON only.

Input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}

Output JSON format:
{
  "predicted_label": "one of the allowed labels",
  "confidence": 0.0,
  "reason": "brief explanation"
}
```

闭集分类结果建议单独保存为：

```text
closed_set_results.csv
```

建议字段：

```text
sample_id
utterance
context
gold_label
predicted_label
confidence
reason
is_correct
```

---

### 4.4 模态扰动实验 Prompt

模态扰动实验用于分析不同输入信息对模型判断的影响。简单来说，就是同一个样本，用不同输入条件跑多次，观察模型输出是否发生变化。

建议至少设置三组：

| 实验设置 | 输入内容 | 目的 |
|---|---|---|
| `text-only` | 只输入当前 utterance | 看单句文本能否判断情绪 |
| `text + context` | 输入 utterance 和上下文 | 看上下文是否能帮助判断 |
| `text + context + image` | 输入 utterance、上下文和视频帧 | 看视觉模态是否带来额外信息 |

如果时间充足，可以增加：

| 实验设置 | 输入内容 | 目的 |
|---|---|---|
| `text + image` | 输入当前 utterance 和视频帧 | 看图像是否补充文本信息 |
| `image-only` | 只输入视频帧 | 看纯视觉信息是否足以判断情绪 |

模态扰动实验不是主实验必须步骤，但如果报告中需要写“模态扰动分析”或“消融实验”，建议保留。最小版本只需要跑：

```text
text-only
text + context
text + context + image
```

为了保证公平，模态扰动实验建议使用统一任务，只改变输入内容。推荐使用开放词汇版本：

```text
You are an emotion analysis assistant.

Given the available information, identify the speaker's fine-grained emotions.

Important requirements:
1. Generate 1 to 3 fine-grained emotion words.
2. Do not directly choose from fixed MELD labels.
3. Use only the information provided in the current input condition.
4. If some information is missing, do not assume it.
5. Return JSON only.

Input condition:
{condition_name}

Available input:
- Speaker: {speaker}
- Utterance: {utterance_or_empty}
- Conversation context: {context_or_empty}
- Visual information: {image_frames_or_empty}

Output JSON format:
{
  "input_condition": "{condition_name}",
  "free_emotions": ["emotion_1", "emotion_2"],
  "confidence": 0.0,
  "reason": "brief explanation"
}
```

三种常用输入条件可以这样填写：

#### 4.4.1 text-only

```text
Input condition:
text-only

Available input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: none
- Visual information: none
```

#### 4.4.2 text + context

```text
Input condition:
text + context

Available input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}
- Visual information: none
```

#### 4.4.3 text + context + image

```text
Input condition:
text + context + image

Available input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}
- Visual information: {image_frames}
```

模态扰动实验输出后，仍然需要走后续的标准化和映射流程：

```text
free_emotions
  ↓
标准化
  ↓
Dictionary / LLM / Hybrid Mapping
  ↓
final_mapped_label
```

模态扰动实验结果建议单独保存为：

```text
modality_ablation_results.csv
```

建议字段：

```text
sample_id
input_condition
utterance
context
image_frame_paths
gold_label
free_emotions_raw
free_emotions_norm
final_mapped_label
confidence
reason
is_correct
```


## 5. API 输出格式要求

API 的输出必须尽量保持 JSON 格式，方便后续自动解析。

推荐输出格式：

```json
{
  "free_emotions": ["annoyed", "frustrated"],
  "confidence": 0.86,
  "reason": "The speaker sounds irritated because the same problem happened again."
}
```

字段说明：

| 字段 | 含义 |
|---|---|
| `free_emotions` | 模型生成的开放情绪词，数量为 1-3 个 |
| `confidence` | 模型对情绪判断的置信度，范围 0-1 |
| `reason` | 简短解释，方便后续做错误分析 |

注意：不要让模型直接输出 `mapped_label`，因为映射规则应由后续程序统一完成。

---

## 6. 开放情绪词标准化

模型输出的词可能有大小写、短语、修饰词等差异，映射前需要先标准化。

标准化规则：

```text
Annoyed → annoyed
very happy → happy
a bit nervous → nervous
shocked/surprised → shocked, surprised
feeling frustrated → frustrated
```

建议处理步骤：

1. 转小写；
2. 去掉 `very`, `slightly`, `a bit`, `feeling` 等修饰词；
3. 拆分 `/`、`,`、`and` 连接的多个词；
4. 去除重复词；
5. 保留原始输出，方便检查。

---

## 7. 情绪映射方案

映射的目标是把开放情绪词转换为 MELD 标准标签。

例如：

```text
annoyed → anger
frustrated → anger
relieved → joy
nervous → fear
```

本项目推荐使用 Hybrid Mapping，即：

```text
先用 Dictionary Mapping
遇到未知词、歧义词或标签冲突时，再用 LLM Mapping
```

---

## 8. 核心情绪词典

词典不用覆盖所有开放词汇，只需要覆盖高频、明确、容易归类的词。低频词、未知词和歧义词交给 LLM 判断。

推荐初始词典如下：

```json
{
  "anger": [
    "angry", "mad", "annoyed", "irritated", "frustrated",
    "furious", "impatient", "resentful", "outraged"
  ],
  "sadness": [
    "sad", "upset", "disappointed", "hurt", "lonely",
    "regretful", "depressed", "heartbroken", "miserable"
  ],
  "joy": [
    "happy", "glad", "excited", "amused", "pleased",
    "relieved", "grateful", "cheerful", "delighted"
  ],
  "fear": [
    "afraid", "scared", "nervous", "anxious", "worried",
    "terrified", "insecure", "panicked"
  ],
  "surprise": [
    "surprised", "shocked", "amazed", "astonished",
    "startled", "unexpected"
  ],
  "disgust": [
    "disgusted", "repulsed", "grossed out", "disturbed",
    "offended"
  ],
  "neutral": [
    "neutral", "calm", "indifferent", "serious",
    "casual", "factual", "matter-of-fact", "plain"
  ]
}
```

---

## 9. 歧义词处理

有些词不建议直接放入固定词典，因为它们要结合上下文判断。

例如：

```text
confused, awkward, uncomfortable, sarcastic, hesitant, serious
```

这些词可能对应不同标签：

| 开放词 | 可能标签 |
|---|---|
| `confused` | surprise / neutral / fear |
| `awkward` | neutral / sadness / disgust |
| `uncomfortable` | disgust / fear / sadness |
| `sarcastic` | anger / neutral |
| `serious` | neutral / sadness / anger |

建议把这些词放入 `ambiguous_words` 列表：

```json
{
  "ambiguous_words": [
    "confused", "awkward", "uncomfortable",
    "sarcastic", "hesitant", "serious"
  ]
}
```

如果开放情绪词在这个列表中，不直接用词典映射，而是交给 LLM 结合原句和上下文判断。

---

## 10. 多个开放词如何得到一个最终标签

每个开放词先单独映射，再决定最终标签。

### 情况一：所有词映射结果一致

```text
annoyed → anger
frustrated → anger
irritated → anger
```

最终结果：

```text
mapped_label = anger
```

### 情况二：多个词映射结果不一致，但有多数票

```text
annoyed → anger
frustrated → anger
disappointed → sadness
```

统计：

```text
anger: 2
sadness: 1
```

最终结果：

```text
mapped_label = anger
```

### 情况三：出现平票

```text
annoyed → anger
disappointed → sadness
```

此时不要随便选，调用 LLM Mapping，根据原句和上下文判断最终标签。

### 情况四：出现未知词或歧义词

例如：

```text
awkward
sarcastic
uncomfortable
```

这些词交给 LLM Mapping。

---

## 11. LLM Mapping Prompt

当词典无法处理时，使用下面的 prompt 进行映射。

```text
You are mapping fine-grained emotion words to MELD emotion labels.

Allowed MELD labels:
anger, disgust, sadness, joy, neutral, surprise, fear

Input:
- Speaker: {speaker}
- Utterance: {utterance}
- Conversation context: {context}
- Fine-grained emotion words: {free_emotions}

Task:
Choose exactly one MELD label that best matches the speaker's emotion.
Use the utterance and context to resolve ambiguous emotion words.

Return JSON only:
{
  "mapped_label": "one of the allowed labels",
  "confidence": 0.0,
  "reason": "brief explanation"
}
```

示例：

输入：

```text
Utterance: I can't believe you forgot again.
Fine-grained emotion words: ["annoyed", "disappointed"]
```

输出：

```json
{
  "mapped_label": "anger",
  "confidence": 0.78,
  "reason": "The speaker expresses irritation toward repeated behavior."
}
```

---

## 12. Hybrid Mapping 最终规则

最终采用以下规则：

```text
1. 对 free_emotions 做标准化。
2. 对每个情绪词查核心词典。
3. 如果词在 ambiguous_words 中，交给 LLM Mapping。
4. 如果词典能覆盖所有词，并且映射结果一致，直接输出该标签。
5. 如果词典结果出现多数票，输出多数票标签。
6. 如果出现平票、未知词或歧义词，调用 LLM Mapping。
7. 每个样本最终只保留一个 MELD 标准标签。
```

---

## 13. 需要保存的结果字段

为了后续评估和分析，API 调用结果不要只保存最终标签。建议每条样本保存以下字段：

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本编号 |
| `dialogue_id` | 对话编号 |
| `utterance_id` | 句子编号 |
| `speaker` | 说话人 |
| `utterance` | 当前句子 |
| `context` | 上下文 |
| `gold_label` | MELD 原始真实标签 |
| `image_frame_paths` | 使用的视频帧路径，没有则为空 |
| `prompt_version` | 使用的 prompt 版本 |
| `model_name` | 调用的模型名称 |
| `raw_output` | API 原始输出 |
| `free_emotions_raw` | 原始开放情绪词 |
| `free_emotions_norm` | 标准化后的开放情绪词 |
| `dictionary_mapping` | 词典映射结果 |
| `llm_mapping` | LLM 映射结果，没有则为空 |
| `final_mapped_label` | 最终映射到的 MELD 标签 |
| `mapping_method` | `dictionary` / `llm` / `hybrid` |
| `mapping_confidence` | 映射置信度 |
| `is_correct` | 是否与 `gold_label` 一致 |

示例：

```json
{
  "sample_id": "dia001_utt003",
  "dialogue_id": "dia001",
  "utterance_id": "utt003",
  "speaker": "Speaker A",
  "utterance": "I can't believe you forgot again.",
  "context": "The speaker had already reminded the listener before.",
  "gold_label": "anger",
  "image_frame_paths": ["frames/dia001_utt003_01.jpg"],
  "prompt_version": "open_emotion_v1",
  "model_name": "gpt-4o",
  "raw_output": {
    "free_emotions": ["annoyed", "frustrated"],
    "confidence": 0.86,
    "reason": "The speaker sounds irritated by repeated behavior."
  },
  "free_emotions_raw": ["annoyed", "frustrated"],
  "free_emotions_norm": ["annoyed", "frustrated"],
  "dictionary_mapping": {
    "annoyed": "anger",
    "frustrated": "anger"
  },
  "llm_mapping": null,
  "final_mapped_label": "anger",
  "mapping_method": "dictionary",
  "mapping_confidence": 0.90,
  "is_correct": true
}
```

---

## 14. 异常情况处理

### 14.1 API 没有返回合法 JSON

处理方式：

1. 保存 `raw_output`；
2. 尝试用正则或简单解析提取 `free_emotions`；
3. 如果仍然失败，标记为 `parse_failed`；
4. 后续可以重新调用该样本。

### 14.2 模型输出超过 3 个情绪词

处理方式：

只保留前 3 个，或者保留置信度最高的 3 个。

### 14.3 模型输出 MELD 标准标签

如果模型输出的是：

```text
anger
```

可以保留，但标记为开放词较弱。后续仍然映射为：

```text
anger → anger
```

### 14.4 词典查不到

处理方式：

调用 LLM Mapping。

### 14.5 词典结果冲突

处理方式：

优先多数投票；如果平票，调用 LLM Mapping。

---

## 15. 建议先跑小样本

正式批量运行前，建议先跑 50-100 条样本。

检查以下内容：

```text
1. API 是否稳定返回 JSON
2. free_emotions 是否基本合理
3. 情绪词是否经常超过 3 个
4. 词典覆盖率是否太低
5. 歧义词是否过多
6. final_mapped_label 是否能正常生成
```

如果发现高频未知词，可以人工检查后加入核心词典。

例如统计发现：

```text
embarrassed: 12
jealous: 8
guilty: 7
```

可以根据语义和样本上下文决定是否加入词典，或者继续交给 LLM Mapping。

---

## 16. 最终交付给后续分析的数据

建议最终保存两个文件：

### 16.1 原始 API 结果文件

文件名：

```text
api_outputs_raw.jsonl
```

用途：保留所有 API 原始输出，方便检查和复现。

### 16.2 映射后结果文件

文件名：

```text
mapped_results.csv
```

用途：后续直接计算 Accuracy、Macro-F1、Weighted-F1、混淆矩阵、词云和错误案例。

`mapped_results.csv` 至少需要包含：

```text
sample_id
utterance
context
gold_label
free_emotions_norm
final_mapped_label
mapping_method
mapping_confidence
is_correct
```

---

## 17. 简要总结

实际执行时可以按下面的最简流程操作：

```text
1. 用统一 prompt 调用 API。
2. 每条样本生成 1-3 个开放情绪词。
3. 保存 API 原始输出。
4. 标准化开放情绪词。
5. 先查核心词典。
6. 遇到未知词、歧义词、标签冲突时调用 LLM Mapping。
7. 每条样本得到一个 final_mapped_label。
8. 保存 mapped_results.csv，供后续统计分析使用。
```
