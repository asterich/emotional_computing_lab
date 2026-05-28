# 基于多模态大模型的开放词汇情感识别与标签映射分析

## 1. 选题

### 1.1 项目名称

**基于多模态大模型的开放词汇情感识别与标签映射分析**

### 1.2 所属方向

**基于多模态大模型的情感识别**

该项目属于情感计算课程中的多模态情感识别方向，同时结合大模型在开放词汇情绪理解、自然语言解释生成和多模态推理中的能力。

### 1.3 选题背景

传统情感识别任务通常采用闭集分类方式，即模型只能从预设情绪标签中选择一个结果，例如 `joy`、`anger`、`sadness`、`neutral` 等。该方式便于量化评价，但难以描述真实情绪的复杂性。例如，一个样本可能同时包含“尴尬”“紧张”“轻微愤怒”等细粒度情绪，而闭集标签只能将其压缩成单一类别。

开放词汇情感识别允许模型自由生成更细粒度的情绪词或情绪短语，例如 `annoyed`、`relieved`、`confused`、`embarrassed`、`slightly nervous` 等。该方式更符合真实人类情绪表达，也更适合多模态大模型的生成式能力。

本项目参考 AffectGPT / OV-MER 的基本思想，设计一个轻量级开放词汇多模态情感识别流程，在可控数据规模下完成闭集分类、开放词汇生成、标签映射、模态扰动、错误归因和可视化分析。

### 1.4 选题意义

该项目的意义主要体现在以下方面：

1. 从传统闭集情绪分类扩展到开放词汇情绪理解；
2. 利用多模态大模型处理文本、上下文和视觉帧信息；
3. 通过标签映射方法解决开放词汇输出难以直接评价的问题；
4. 从分类性能、情绪词多样性、模态扰动和错误归因等角度进行综合分析；
5. 在较短周期内形成一个可运行、可评价、可展示的情感计算项目。

---

## 2. 项目目标

### 2.1 总体目标

构建一个基于多模态大模型的开放词汇情感识别实验系统，使模型能够根据文本、图像帧和上下文信息生成细粒度情绪词，并将开放词汇情绪结果映射到标准情绪标签，最终完成分类性能评价和多角度分析。

### 2.2 具体目标

1. 实现闭集情感分类 baseline；
2. 实现开放词汇情绪生成方法；
3. 设计并比较不同标签映射方法；
4. 分析开放情绪词的多样性和细粒度表达能力；
5. 设计模态扰动实验，分析情绪预测漂移现象；
6. 绘制混淆矩阵，分析情绪类别之间的混淆关系；
7. 进行错误类型归因，分析模型失效原因；
8. 生成开放情绪词词云图；
9. 形成完整实验报告、项目 README、展示 PPT 和课堂汇报材料。

---

## 3. 工作内容

### 3.1 整体技术路线

项目整体流程如下：

```text
MELD 数据集样本
        ↓
文本、说话人、上下文、视频关键帧提取
        ↓
多模态大模型推理
        ↓
闭集情绪分类 / 开放词汇情绪生成
        ↓
开放情绪词到标准标签的映射
        ↓
分类性能评价
        ↓
开放情绪词多样性分析
        ↓
模态扰动下的情绪漂移分析
        ↓
混淆矩阵与错误类型归因
        ↓
词云图、统计图、案例展示
```

### 3.2 工作模块划分

| 模块 | 内容 | 输出 |
|---|---|---|
| 数据处理 | 数据集下载、样本抽取、视频抽帧、上下文构造 | 处理后的实验数据 |
| 闭集分类 | 使用固定情绪标签进行情感分类 | 闭集 baseline 结果 |
| 开放词汇生成 | 生成细粒度开放情绪词 | 开放情绪词与情绪描述 |
| 标签映射 | 将开放情绪词映射到标准标签 | 映射标签与映射结果统计 |
| 性能评价 | 计算 Accuracy、Macro-F1、Weighted-F1 | 性能对比表 |
| 多样性分析 | 统计开放情绪词数量、频率和分布 | 情绪词统计表、词云图 |
| 模态扰动 | 删除上下文、移除图像、遮蔽情绪词等 | 漂移分析结果 |
| 错误归因 | 分析错误样本原因 | 错误类型统计表 |
| 可视化 | 绘制混淆矩阵、词云图、柱状图 | 项目图表 |
| 文档展示 | 完成报告、README、PPT 和汇报 | 最终提交材料 |

---

## 4. 核心实验设计

## 4.1 不同方法分类性能比较

### 4.1.1 实验目的

比较闭集分类方法、开放词汇生成方法以及不同标签映射方法在情感识别任务中的表现。

### 4.1.2 方法设置

| 方法编号 | 方法名称 | 方法说明 |
|---|---|---|
| M1 | Closed-set Direct Classification | 直接要求模型从固定情绪标签中选择一个结果 |
| M2 | Open-vocab + Dictionary Mapping | 模型先生成开放情绪词，再通过人工词典映射到标准标签 |
| M3 | Open-vocab + LLM Mapping | 模型先生成开放情绪词，再由大模型进行语义映射 |
| M4 | Open-vocab + Hybrid Mapping | 优先使用词典映射，词典无法覆盖时使用大模型语义映射 |

### 4.1.3 闭集分类 Prompt

```text
Given the dialogue context, current utterance, speaker information, and visual frame,
choose exactly one emotion from the following labels:

anger, disgust, sadness, joy, neutral, surprise, fear.

Return JSON only:
{
  "label": "...",
  "confidence": 0.0-1.0,
  "reason": "..."
}
```

### 4.1.4 开放词汇情绪生成 Prompt

```text
Given the dialogue context, current utterance, speaker information, and visual frame,
infer the speaker's emotional state.

Do not restrict the answer to predefined emotion labels.
Generate 1 to 3 fine-grained emotion words or phrases.

Then provide a short explanation.

Return JSON only:
{
  "free_emotions": ["...", "..."],
  "emotion_description": "...",
  "confidence": 0.0-1.0,
  "reason": "..."
}
```

### 4.1.5 标签映射方式

#### 4.1.5.1 Dictionary Mapping

构建人工情绪词典，将开放情绪词映射到标准标签。

| 标准标签 | 典型开放情绪词 |
|---|---|
| anger | angry, annoyed, irritated, frustrated, furious, mad |
| disgust | disgusted, uncomfortable, contemptuous, repulsed |
| sadness | sad, disappointed, lonely, hurt, regretful, depressed |
| joy | happy, amused, excited, relieved, pleased, delighted |
| neutral | calm, indifferent, factual, plain, emotionless |
| surprise | surprised, shocked, amazed, startled, confused |
| fear | afraid, nervous, anxious, worried, scared |

#### 4.1.5.2 LLM Mapping

将开放情绪词、情绪描述和标准标签输入大模型，由大模型完成语义映射。

```text
Map the following fine-grained emotions to exactly one coarse emotion label.

Fine-grained emotions:
["annoyed", "impatient"]

Description:
"The speaker seems irritated by the situation."

Coarse labels:
anger, disgust, sadness, joy, neutral, surprise, fear

Return JSON:
{
  "mapped_label": "...",
  "mapping_reason": "..."
}
```

#### 4.1.5.3 Hybrid Mapping

Hybrid Mapping 作为项目中的主要改进方法。

规则如下：

1. 若开放情绪词存在于人工词典中，则直接采用词典映射结果；
2. 若多个情绪词映射到不同标签，则采用多数投票；
3. 若词典无法覆盖，则调用大模型进行语义映射；
4. 若大模型映射结果置信度较低，则标记为 `uncertain`，并在错误分析中单独统计。

### 4.1.6 评价指标

| 指标 | 含义 |
|---|---|
| Accuracy | 总体分类准确率 |
| Macro-F1 | 各类别 F1 的算术平均，适合观察少数类表现 |
| Weighted-F1 | 按类别样本数加权后的 F1 |
| Per-class F1 | 每个情绪类别单独计算 F1 |
| Mapping Coverage | 词典映射能够覆盖的开放情绪词比例 |
| Mapping Error Rate | 标签映射错误比例 |

### 4.1.7 预期结果表

| 方法 | Accuracy | Macro-F1 | Weighted-F1 | Mapping Coverage |
|---|---:|---:|---:|---:|
| Closed-set Direct Classification | - | - | - | - |
| Open-vocab + Dictionary Mapping | - | - | - | - |
| Open-vocab + LLM Mapping | - | - | - | - |
| Open-vocab + Hybrid Mapping | - | - | - | - |

---

## 4.2 开放情绪词多样性分析

### 4.2.1 实验目的

分析开放词汇情感识别相较于闭集分类是否能够提供更细粒度、更丰富的情绪表达。

闭集分类只能输出固定类别，例如：

```text
joy
```

开放词汇方法可以输出更细的情绪状态，例如：

```text
amused, relieved, excited, slightly nervous
```

### 4.2.2 分析内容

开放情绪词多样性分析主要包括以下内容：

1. 统计模型生成的不同开放情绪词数量；
2. 统计每个样本平均生成的情绪词数量；
3. 分析每个标准情绪标签下的高频开放情绪词；
4. 分析开放情绪词中的长尾现象；
5. 比较不同输入模态下开放情绪词的变化；
6. 比较正确样本与错误样本中的情绪词分布差异。

### 4.2.3 多样性指标

| 指标 | 说明 |
|---|---|
| Unique Emotion Count | 所有样本中不同开放情绪词的总数 |
| Average Emotion Number | 每个样本平均生成的开放情绪词数量 |
| Emotion Diversity Ratio | 不同情绪词数量 / 情绪词总数 |
| Long-tail Emotion Ratio | 低频开放情绪词占全部情绪词的比例 |
| Top Emotion Words | 出现频率最高的开放情绪词 |

### 4.2.4 分析表示例

| Gold Label | 高频开放情绪词 |
|---|---|
| joy | happy, amused, excited, relieved, pleased |
| anger | annoyed, frustrated, irritated, angry |
| sadness | disappointed, sad, hurt, lonely |
| surprise | shocked, confused, startled, surprised |
| fear | nervous, anxious, worried, afraid |
| disgust | uncomfortable, disgusted, repulsed |
| neutral | calm, neutral, indifferent, factual |

### 4.2.5 分析价值

该部分能够体现开放词汇情感识别的优势。即使开放词汇方法在粗粒度分类准确率上未必显著超过闭集分类，它仍然可以生成更加细致的情绪解释，从而提升结果的可解释性和人类可读性。

---

## 4.3 模态扰动下的漂移分析

### 4.3.1 实验目的

分析多模态大模型在情感识别过程中是否真正利用了文本、视觉和上下文信息，并观察不同模态缺失或被扰动时，模型预测结果是否发生变化。

### 4.3.2 输入模态设置

| 实验组 | 输入内容 |
|---|---|
| T | 当前 utterance 文本 |
| T + C | 当前文本 + 对话上下文 |
| T + I | 当前文本 + 视频关键帧 |
| T + C + I | 当前文本 + 对话上下文 + 视频关键帧 |

其中：

- `T` 表示文本信息；
- `C` 表示上下文信息；
- `I` 表示图像信息，即视频片段中抽取的关键帧。

### 4.3.3 扰动方式

| 扰动方式 | 具体操作 | 分析目的 |
|---|---|---|
| Remove Context | 删除前后文，只保留当前 utterance | 判断上下文是否影响情绪识别 |
| Remove Image | 删除视觉帧，只保留文本信息 | 判断视觉模态是否提供额外情绪线索 |
| Mask Emotion Words | 遮蔽文本中的明显情绪词 | 判断模型是否过度依赖情绪关键词 |
| Replace Image | 使用无关图像替换原始关键帧 | 判断模型是否容易受到错误视觉信息干扰 |
| Image Blur | 对图像进行模糊处理 | 判断视觉细节变化是否影响预测 |

### 4.3.4 漂移指标

#### Label Shift Rate

```text
Label Shift Rate = 扰动后预测标签发生变化的样本数 / 样本总数
```

该指标用于衡量粗粒度情绪标签是否因扰动而变化。

#### Fine-grained Emotion Shift Rate

```text
Fine-grained Emotion Shift Rate = 扰动后开放情绪词发生变化的样本数 / 样本总数
```

该指标用于衡量细粒度开放情绪词是否因扰动而变化。

#### Confidence Drop

```text
Confidence Drop = 原始输入置信度 - 扰动后输入置信度
```

该指标用于衡量模型对扰动样本的不确定性变化。

### 4.3.5 预期结果表

| 输入设置 | Accuracy | Macro-F1 | Label Shift Rate | Emotion Word Shift Rate | Avg Confidence |
|---|---:|---:|---:|---:|---:|
| Original: T + C + I | - | - | - | - | - |
| Remove Context | - | - | - | - | - |
| Remove Image | - | - | - | - | - |
| Mask Emotion Words | - | - | - | - | - |
| Replace Image | - | - | - | - | - |
| Image Blur | - | - | - | - | - |

### 4.3.6 分析价值

该部分能够回答以下问题：

1. 多模态大模型是否主要依赖文本关键词完成情感识别；
2. 视觉模态是否能够改变模型对情绪的判断；
3. 上下文是否能帮助模型识别讽刺、反语和隐含情绪；
4. 开放情绪词是否比粗粒度标签更容易发生漂移；
5. 模态扰动是否会导致模型置信度下降。

---

## 4.4 混淆矩阵与错误类型归因

### 4.4.1 混淆矩阵分析

分类结果将使用混淆矩阵进行可视化，分析不同情绪类别之间的混淆关系。

重点关注以下混淆情况：

| 易混淆类别 | 可能原因 |
|---|---|
| sadness vs neutral | 低强度悲伤容易被识别为中性 |
| anger vs disgust | 愤怒和厌恶在表情与语言上可能存在重叠 |
| surprise vs fear | 惊讶与恐惧都可能包含高唤醒状态 |
| joy vs neutral | 轻微积极情绪容易被识别为中性 |
| fear vs sadness | 焦虑、担忧等情绪可能处于恐惧和悲伤之间 |

### 4.4.2 错误案例表

| 样本编号 | Gold Label | Closed-set Result | Open Emotions | Mapped Label | 错误类型 |
|---|---|---|---|---|---|
| 001 | anger | neutral | annoyed, impatient | anger | 闭集误判，开放词汇修正 |
| 002 | sadness | neutral | disappointed | sadness | 开放词汇更细 |
| 003 | surprise | fear | nervous, shocked | fear | 标签映射错误 |
| 004 | neutral | joy | amused | joy | 过度解读表情 |
| 005 | disgust | anger | irritated, uncomfortable | anger | 粗粒度标签边界模糊 |

### 4.4.3 错误类型归因体系

项目将错误类型划分为以下几类：

| 错误类型 | 说明 |
|---|---|
| Label Granularity Mismatch | 开放情绪词比数据集标签更细，难以准确映射到单一标准标签 |
| Emotion Ambiguity | 样本本身情绪模糊，人工标签和模型判断均存在合理性 |
| Multi-emotion Mixture | 一个样本同时包含多种情绪，闭集标签难以表达 |
| Text-dominant Bias | 模型过度依赖文本关键词，忽略视觉或上下文信息 |
| Visual Hallucination | 模型编造或过度解读图像中的表情、动作、姿态 |
| Context Ignorance | 模型忽略前后文，导致反语、讽刺或语境情绪识别错误 |
| Mapping Error | 开放情绪词合理，但映射到标准标签时出现错误 |
| Low-intensity Emotion Error | 低强度情绪被识别为 neutral 或其他相近类别 |

### 4.4.4 分析价值

混淆矩阵和错误类型归因能够体现项目不是单纯追求准确率，而是进一步分析模型为什么出错、在哪些情绪类别上出错、开放词汇方法是否能够缓解闭集分类的局限。

---

## 4.5 词云图生成

### 4.5.1 实验目的

通过词云图直观展示开放词汇情绪识别生成的细粒度情绪词分布。

### 4.5.2 词云图类型

项目计划生成以下词云图：

| 图类型 | 内容 |
|---|---|
| 全局开放情绪词词云 | 所有样本生成的开放情绪词 |
| joy 类词云 | Gold label 为 joy 的样本中生成的开放情绪词 |
| anger 类词云 | Gold label 为 anger 的样本中生成的开放情绪词 |
| sadness 类词云 | Gold label 为 sadness 的样本中生成的开放情绪词 |
| neutral 类词云 | Gold label 为 neutral 的样本中生成的开放情绪词 |
| 错误样本词云 | 预测错误样本中生成的开放情绪词 |

### 4.5.3 可视化目标

词云图用于展示：

1. 开放情绪词的频率分布；
2. 不同标准情绪类别下的细粒度表达差异；
3. 模型是否存在高频词偏置；
4. 错误样本中是否集中出现某些模糊情绪词；
5. 开放词汇方法相较于闭集标签的表达丰富性。

---

## 5. 数据集和模型选择

## 5.1 数据集选择

### 5.1.1 主数据集：MELD

项目选择 **MELD** 作为主要实验数据集。

MELD 是一个多模态对话情感识别数据集，包含文本、音频和视频信息。该数据集以对话片段为基本单位，每个 utterance 均包含对应的情绪标签。

### 5.1.2 选择原因

| 原因 | 说明 |
|---|---|
| 多模态信息完整 | 包含文本、视觉和音频信息，适合多模态情感识别 |
| 情绪标签明确 | 包含 anger、disgust、sadness、joy、neutral、surprise、fear 七类情绪 |
| 适合开放词汇分析 | 离散情绪标签便于与开放情绪词进行映射 |
| 数据处理成本可控 | 可从视频中抽取关键帧，结合文本和上下文进行推理 |
| 评价方式成熟 | 可使用 Accuracy、Macro-F1、Weighted-F1 和混淆矩阵进行评价 |

### 5.1.3 样本规模

考虑到项目周期和接口调用成本，计划从 MELD dev/test 集中抽取约 **300 条样本** 进行实验。

抽样策略如下：

1. 优先保证七类情绪均有覆盖；
2. 尽量保持类别分布相对均衡；
3. 对 `fear`、`disgust` 等少数类进行适当补充；
4. 每条样本保留当前 utterance、说话人、前后文和视频关键帧。

### 5.1.4 数据字段设计

整理后的数据格式如下：

| 字段名 | 说明 |
|---|---|
| sample_id | 样本编号 |
| dialogue_id | 对话编号 |
| utterance_id | 话语编号 |
| speaker | 说话人 |
| context | 前若干轮对话上下文 |
| utterance | 当前待识别话语 |
| gold_label | 标准情绪标签 |
| video_path | 视频片段路径 |
| frame_path | 抽取的视频关键帧路径 |
| prompt_type | 使用的 prompt 类型 |
| model_output | 大模型原始输出 |
| parsed_label | 解析后的预测标签 |
| free_emotions | 开放词汇情绪词 |
| mapped_label | 映射后的标准标签 |
| confidence | 模型置信度 |
| reason | 模型解释 |

---

## 5.2 模型选择

### 5.2.1 主模型

项目使用支持图文输入的多模态大模型作为主模型，完成文本、上下文和图像帧联合推理。

可选模型包括：

| 模型 | 使用方式 | 说明 |
|---|---|---|
| Qwen2.5-VL | 本地部署或 API 调用 | 中文和英文能力较好，适合图文情感理解 |
| GPT-4o / GPT-4.1 系列 | API 调用 | 多模态理解能力强，输出稳定 |
| Gemini 系列 | API 调用 | 长上下文和图像理解能力较强 |
| InternVL 系列 | 本地部署 | 开源多模态模型，适合本地实验 |

### 5.2.2 推荐实现方式

在项目周期较短的情况下，推荐采用以下实现方式：

1. 使用图文多模态大模型处理 `文本 + 上下文 + 视频关键帧`；
2. 暂不进行模型训练或微调；
3. 使用 prompt engineering 完成闭集分类和开放词汇生成；
4. 使用脚本自动解析 JSON 输出；
5. 使用 Python 完成指标统计和图表绘制。

### 5.2.3 备选模型方案

如主模型部署或调用受限，可采用以下备选方案：

1. 使用图像描述模型先将关键帧转写为视觉描述；
2. 将文本、上下文和视觉描述输入文本大模型；
3. 使用文本大模型完成近似多模态情感识别。

备选流程如下：

```text
视频关键帧
    ↓
图像描述模型生成视觉描述
    ↓
文本 + 上下文 + 视觉描述
    ↓
文本大模型进行情感识别
```

### 5.2.4 输出格式约束

所有模型输出统一使用 JSON 格式，便于后续自动解析和统计。

示例输出：

```json
{
  "free_emotions": ["annoyed", "impatient"],
  "mapped_label": "anger",
  "confidence": 0.82,
  "textual_evidence": "The utterance expresses dissatisfaction.",
  "visual_evidence": "The speaker appears tense.",
  "reason": "The text and visual cue jointly suggest anger."
}
```

---

## 6. 实验结果统计与绘图内容

## 6.1 统计内容

项目计划统计以下结果：

1. 不同方法的 Accuracy、Macro-F1、Weighted-F1；
2. 不同映射方法的覆盖率和错误率；
3. 各类别 Precision、Recall、F1；
4. 开放情绪词总数；
5. 高频开放情绪词；
6. 不同类别下的开放情绪词分布；
7. 模态扰动前后的标签漂移率；
8. 模态扰动前后的开放情绪词漂移率；
9. 模型置信度变化；
10. 错误类型分布。

## 6.2 绘图内容

| 图表 | 说明 |
|---|---|
| 方法性能对比柱状图 | 比较不同方法 Accuracy、Macro-F1、Weighted-F1 |
| 混淆矩阵 | 展示各情绪类别之间的误判关系 |
| 高频情绪词柱状图 | 展示出现频率最高的开放情绪词 |
| 开放情绪词词云图 | 展示开放情绪词整体分布 |
| 各类别词云图 | 展示不同 gold label 下的开放情绪词差异 |
| 扰动实验结果柱状图 | 展示不同扰动方式下的性能变化 |
| Label Shift Rate 图 | 展示模态扰动导致的标签漂移比例 |
| 错误类型饼图 / 柱状图 | 展示不同错误原因的占比 |

---

## 7. 项目实现计划

## 7.1 数据处理

1. 下载并整理 MELD 数据集；
2. 读取 utterance、speaker、dialogue_id、emotion label 等字段；
3. 为每个样本构造上下文文本；
4. 从视频片段中抽取关键帧；
5. 生成统一格式的实验 CSV / JSON 文件。

## 7.2 模型推理

1. 构造闭集分类 prompt；
2. 构造开放词汇情绪生成 prompt；
3. 构造标签映射 prompt；
4. 批量调用多模态大模型；
5. 保存模型原始输出；
6. 解析 JSON 结果并处理异常输出。

## 7.3 标签映射

1. 构建人工开放情绪词典；
2. 实现 Dictionary Mapping；
3. 实现 LLM Mapping；
4. 实现 Hybrid Mapping；
5. 统计映射覆盖率和映射错误率。

## 7.4 评价与分析

1. 计算 Accuracy、Macro-F1、Weighted-F1；
2. 计算每类 Precision、Recall、F1；
3. 绘制混淆矩阵；
4. 统计开放情绪词频；
5. 计算多样性指标；
6. 完成模态扰动实验；
7. 计算 Label Shift Rate 和 Emotion Word Shift Rate；
8. 进行错误案例归因。

## 7.5 可视化与展示

1. 绘制方法性能对比图；
2. 绘制混淆矩阵；
3. 绘制词云图；
4. 绘制扰动实验柱状图；
5. 整理典型案例；
6. 制作 PPT；
7. 完成课堂汇报材料。

---

## 8. 分工方式

项目由三名成员完成，整体分工如下：

- 成员 A 和成员 B 共同负责代码实现、实验执行、数据统计、绘图和 PPT 制作；
- 成员 C 负责课堂汇报和项目报告撰写；
- 三名成员共同参与选题讨论、方法设计、结果审核和最终材料检查。

## 8.1 成员 A：数据处理、模型调用与实验执行

### 主要职责

1. 数据集下载与目录结构整理；
2. 样本抽取与类别分布统计；
3. 视频关键帧抽取脚本编写；
4. 上下文构造与数据字段统一；
5. 闭集分类实验脚本编写；
6. 开放词汇生成实验脚本编写；
7. 多模态大模型接口调用；
8. 原始结果保存与异常输出处理。

### 交付内容

- 数据处理脚本；
- 视频抽帧脚本；
- 模型调用脚本；
- 原始实验结果文件；
- 清洗后的结果文件。

## 8.2 成员 B：标签映射、数据统计、绘图与 PPT 制作

### 主要职责

1. 实现 Dictionary Mapping、LLM Mapping 和 Hybrid Mapping；
2. 计算 Accuracy、Macro-F1、Weighted-F1；
3. 计算 Per-class Precision、Recall、F1；
4. 绘制混淆矩阵；
5. 统计开放情绪词频；
6. 计算开放情绪词多样性指标；
7. 完成模态扰动下的漂移分析；
8. 绘制词云图、柱状图、饼图等可视化结果；
9. 负责项目展示 PPT 制作。

### 交付内容

- 标签映射代码；
- 指标统计脚本；
- 混淆矩阵；
- 词云图；
- 扰动分析图；
- 方法性能对比图；
- 项目展示 PPT。

## 8.3 成员 C：报告撰写与课堂汇报

### 主要职责

1. 完成项目报告撰写；
2. 整理研究背景与相关工作；
3. 整理实验方法与技术路线；
4. 汇总实验结果和分析结论；
5. 撰写错误案例归因部分；
6. 整理项目 README；
7. 完成课堂展示讲稿；
8. 负责上台 presentation。

### 交付内容

- 项目报告；
- 项目 README；
- 汇报讲稿；
- 课堂展示汇报。

---

## 9. 预期成果

## 9.1 代码成果

项目预计形成以下代码文件：

```text
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sampled_data.json
├── frames/
│   └── extracted_frames/
├── prompts/
│   ├── closed_set_prompt.txt
│   ├── open_vocab_prompt.txt
│   └── mapping_prompt.txt
├── scripts/
│   ├── sample_dataset.py
│   ├── extract_frames.py
│   ├── run_closed_set.py
│   ├── run_open_vocab.py
│   ├── emotion_mapping.py
│   ├── perturbation_experiment.py
│   ├── evaluate.py
│   └── visualize.py
├── results/
│   ├── raw_outputs/
│   ├── parsed_outputs/
│   ├── metrics.csv
│   └── error_cases.csv
├── figures/
│   ├── confusion_matrix.png
│   ├── performance_comparison.png
│   ├── emotion_wordcloud.png
│   ├── perturbation_results.png
│   └── error_type_distribution.png
├── report/
│   └── final_report.pdf
├── presentation/
│   └── final_presentation.pptx
└── README.md
```

## 9.2 实验成果

1. 闭集情绪分类结果；
2. 开放词汇情绪生成结果；
3. 不同标签映射方法对比结果；
4. 开放情绪词多样性统计；
5. 模态扰动下的情绪漂移分析；
6. 混淆矩阵；
7. 错误类型归因；
8. 开放情绪词词云图；
9. 典型案例展示。

## 9.3 文档成果

1. 项目 README；
2. 实验报告；
3. PPT；
4. 汇报讲稿；
5. 结果图表与案例分析材料。

---

## 10. 项目创新点

## 10.1 开放词汇情绪生成

项目不局限于传统闭集情绪分类，而是让多模态大模型生成更加细粒度的开放情绪词，提升情感识别结果的表达能力。

## 10.2 Hybrid Emotion Mapping

项目设计 Hybrid Emotion Mapping 方法，将人工词典映射和大模型语义映射结合，使开放词汇情绪输出能够对齐到标准情绪标签，并支持分类指标评价。

## 10.3 情绪词多样性分析

项目不仅计算分类准确率，还进一步分析开放情绪词的数量、频率、多样性和长尾分布，从而体现开放词汇方法的解释价值。

## 10.4 模态扰动漂移分析

项目通过删除上下文、移除图像、遮蔽情绪词、替换图像等方式，分析模型预测结果和开放情绪词在不同模态扰动下的变化。

## 10.5 错误类型归因

项目对错误样本进行细粒度归因，分析模型在标签粒度、文本偏置、视觉幻觉、上下文忽略和映射错误等方面的局限。

---

## 11. 预期结论

本项目预期得出以下结论：

1. 开放词汇情感识别能够生成比闭集分类更细粒度的情绪表达；
2. 开放词汇方法不一定在粗粒度 Accuracy 上显著优于闭集分类，但在可解释性和情绪表达丰富性方面具有优势；
3. Hybrid Emotion Mapping 能够在覆盖率和稳定性之间取得较好平衡；
4. 模态扰动会导致部分样本出现标签漂移和开放情绪词漂移，说明多模态信息会影响模型情绪判断；
5. 文本仍然可能是多模态大模型情绪判断中的主导信息来源；
6. 视觉模态和上下文信息在讽刺、反语、低强度情绪和多情绪混合样本中具有补充作用；
7. 模型错误主要来源于标签粒度不匹配、情绪模糊、多情绪混合、视觉过度解读和标签映射错误。

---

## 12. 项目周期安排

| 时间 | 工作内容 |
|---|---|
| Day 1 | 数据集下载、样本抽取、视频抽帧、数据格式整理 |
| Day 2 | 闭集分类 prompt、开放词汇 prompt、模型调用脚本实现 |
| Day 3 | 批量实验、标签映射、结果解析、异常样本清洗 |
| Day 4 | 指标计算、混淆矩阵、词云图、扰动实验、错误案例分析 |
| Day 5 | 报告撰写、README 整理、PPT 制作、课堂 pre 准备 |

---

## 13. README 摘要

### 13.1 项目简介

本项目面向多模态情感识别任务，构建了一个基于多模态大模型的开放词汇情感识别实验流程。项目以 MELD 数据集为主要实验数据，使用文本、上下文和视频关键帧作为输入，比较闭集分类、开放词汇生成和不同标签映射方法的性能差异。

### 13.2 核心功能

- 闭集情绪分类；
- 开放词汇情绪生成；
- Dictionary Mapping；
- LLM Mapping；
- Hybrid Mapping；
- 情绪词多样性分析；
- 模态扰动漂移分析；
- 混淆矩阵绘制；
- 错误类型归因；
- 词云图生成。

### 13.3 项目输出

- 分类性能对比表；
- 开放情绪词统计表；
- 混淆矩阵；
- 词云图；
- 模态扰动分析图；
- 错误类型归因表；
- 项目报告；
- 展示 PPT。

### 13.4 关键词

```text
Multimodal Emotion Recognition
Open-vocabulary Emotion Recognition
AffectGPT
OV-MER
Large Multimodal Model
Emotion Mapping
MELD
Emotion Word Diversity
Modality Perturbation
Error Attribution
```
