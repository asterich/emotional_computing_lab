# 项目分工与交付安排

## 1. 项目基本信息

**项目方向：** 基于多模态大模型的开放词汇情感识别与标签映射分析  
**项目成员：** A、B、C  
**阶段目标：** 在本周六前完成代码、实验、结果统计、图表、PPT 、大致讲稿，周日完成汇报准备，下周一进行课程展示。

> 时间节点按当前安排记录：  
> - 本周六：2026-05-30  
> - 周日：2026-05-31  
> - 下周一：2026-06-01

---

## 2. 总体分工原则

本项目采用“代码实验与结果生产—统计可视化与展示材料—报告撰写与课堂汇报”三线协作方式。

- **A：负责项目仓库、代码框架、数据集、实验运行与结果交付。**
- **B：负责 prompt 设计、映射方案设计、结果统计、图表绘制与 PPT 制作。**
- **C：负责最终报告、课堂 pre、讲稿完善，以及确认提交要求。**

AB 的任务需在**本周六前完成**，并将交付内容发送至项目群。周日由 C 进行 pre 准备、讲稿完善。

---

## 3. 成员 A 分工：仓库、代码框架、数据集与实验运行

### 3.1 主要职责

A 负责项目的工程实现和实验执行，确保项目有清晰的代码结构、可复现实验流程和完整实验结果。

具体任务包括：

1. **建立项目仓库**
   - 创建 GitHub / Gitee / 本地项目仓库。
   - 初始化项目目录结构。
   - 编写基础 `README.md` 或运行说明草稿。
   - 维护代码文件、数据路径、结果路径的清晰组织。

2. **构建代码框架并撰写代码**
   - 编写数据读取与样本抽取脚本。
   - 编写视频关键帧抽取脚本。
   - 编写闭集分类实验脚本。
   - 编写开放词汇情绪生成实验脚本。
   - 编写批量调用模型的运行脚本。
   - 编写结果保存与 JSON 解析脚本。
   - 保证实验结果可被 B 直接读取和统计。

3. **下载并整理数据集**
   - 下载 MELD 数据集或最终确定的数据集。
   - 整理文本、标签、视频路径等字段。
   - 抽取约 300 条实验样本。
   - 尽量保证七类情绪样本均有覆盖。
   - 生成统一格式的 `sampled_data.json` 或 `sampled_data.csv`。

4. **运行实验**
   - 根据 B 提供的 prompt 执行闭集分类实验。
   - 根据 B 提供的开放词汇 prompt 执行开放词汇生成实验。
   - 根据 B 提供的扰动实验设置运行相关实验。
   - 保存模型原始输出和解析后的结构化结果。

5. **规范保存实验结果**
   - 将实验输出放在合适的、独立的路径下。
   - 不同实验结果应分目录保存，避免混杂。
   - 结果文件命名应包含方法名、输入模态和日期信息。
   - 将最终结果交付给 B 进行统计和绘图。

### 3.2 建议目录结构

```text
project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── sampled_data.csv
├── frames/
│   └── extracted_frames/
├── prompts/
│   ├── closed_set_prompt.txt
│   ├── open_vocab_prompt.txt
│   ├── dictionary_mapping_prompt.txt
│   ├── llm_mapping_prompt.txt
│   └── hybrid_mapping_prompt.txt
├── scripts/
│   ├── sample_dataset.py
│   ├── extract_frames.py
│   ├── run_closed_set.py
│   ├── run_open_vocab.py
│   ├── run_perturbation.py
│   ├── parse_outputs.py
│   └── utils.py
├── results/
│   ├── raw_outputs/
│   │   ├── closed_set/
│   │   ├── open_vocab/
│   │   └── perturbation/
│   ├── parsed_outputs/
│   │   ├── closed_set_results.csv
│   │   ├── open_vocab_results.csv
│   │   └── perturbation_results.csv
│   └── final_for_B/
│       ├── sampled_data.csv
│       ├── closed_set_results.csv
│       ├── open_vocab_results.csv
│       ├── mapping_inputs.csv
│       └── perturbation_results.csv
└── logs/
    └── run_logs.txt
```

### 3.3 交付给 B 的内容

A 需要在**本周四前**将以下内容打包或提交到群里：

```text
results/final_for_B/
├── sampled_data.csv
├── closed_set_results.csv
├── open_vocab_results.csv
├── mapping_inputs.csv
├── perturbation_results.csv
└── result_field_description.md
```

其中 `result_field_description.md` 需要说明每个字段的含义，例如：

| 字段 | 含义 |
|---|---|
| sample_id | 样本编号 |
| gold_label | 原始情绪标签 |
| closed_pred | 闭集分类预测标签 |
| free_emotions | 开放词汇情绪词 |
| mapped_label | 映射后的标准标签 |
| confidence | 模型置信度 |
| prompt_type | 使用的 prompt 类型 |
| perturbation_type | 扰动类型 |
| raw_output | 模型原始输出 |

### 3.4 A 的完成标准

A 的任务完成标准如下：

- 仓库结构清晰，代码可运行。
- 数据集已下载并整理。
- 样本文件字段完整。
- 闭集分类结果已生成。
- 开放词汇结果已生成。
- 扰动实验结果已生成。
- 原始结果和解析结果均已保存。
- 结果文件已放在独立路径下并交付给 B。

---

## 4. 成员 B 分工：Prompt、映射方案、统计绘图与 PPT

### 4.1 主要职责

B 负责实验设计层面的 prompt、标签映射方案、统计分析、可视化图表和展示 PPT。B 的 prompt 和映射方案需要**尽早**发给 A，以保证 A 能够按统一实验设计运行代码。

具体任务包括：

1. **设计所有 prompt**
   - 闭集分类 prompt。
   - 开放词汇情绪生成 prompt。
   - 模态扰动实验 prompt。
   - 标签映射 prompt。
   - 输出格式约束 prompt。

2. **设计三种标签映射方案**
   - 基于词表的 Dictionary Mapping。
   - 基于 LLM 的 LLM Mapping。
   - 结合词表与 LLM 的 Hybrid Mapping。

3. **尽早将 prompt 和映射方案发给 A**
   - prompt 应尽早定稿并发至项目群。
   - prompt 文件建议同时提交 `.txt` 和 `.md` 两种格式。
   - 每个 prompt 需要说明使用场景和输入字段。
   - 映射词表需要以 CSV / JSON / Markdown 表格形式交付。

4. **整理所有实验结果并进行统计**
   - 接收 A 交付的实验结果。
   - 清洗异常输出（如有需要）。
   - 统计不同方法的 Accuracy、Macro-F1、Weighted-F1。
   - 统计各类别 Precision、Recall、F1。
   - 统计开放情绪词数量、词频和多样性指标。
   - 统计扰动实验中的 Label Shift Rate 与 Emotion Word Shift Rate。
   - 整理错误案例表，分析原因。

5. **绘制图表**
   - 方法性能对比柱状图。
   - 混淆矩阵。
   - 高频开放情绪词柱状图。
   - 开放情绪词词云图。
   - 模态扰动漂移分析图。
   - 错误类型分布图。

6. **制作 PPT**
   - PPT 简明清晰，不罗列大段文字。
   - 每页聚焦一个核心信息。
   - 多插入图表、流程图和样例结果。
   - 重点展示实验设计、结果图表、典型案例和结论。
   - 配套文稿，C 周日进行完善。

### 4.2 Prompt 交付清单

B 需要尽早交付以下 prompt 文件：

```text
prompts/
├── closed_set_prompt.md
├── open_vocab_prompt.md
├── perturbation_prompt.md
├── dictionary_mapping_prompt.md
├── llm_mapping_prompt.md
├── hybrid_mapping_prompt.md
└── prompt_usage_guide.md
```

每个 prompt 文件应包含：

- 使用目的；
- 输入字段；
- 输出 JSON 格式；
- 标签范围；
- 示例输入；
- 示例输出。

### 4.3 三种映射方案要求

#### 4.3.1 Dictionary Mapping

基于人工词表进行开放情绪词到标准标签的映射。

示例：

| 标准标签 | 细粒度情绪词示例 |
|---|---|
| anger | angry, annoyed, irritated, frustrated, furious |
| disgust | disgusted, uncomfortable, contemptuous, repulsed |
| sadness | sad, disappointed, hurt, lonely, regretful |
| joy | happy, amused, excited, relieved, pleased |
| neutral | calm, neutral, indifferent, factual, emotionless |
| surprise | surprised, shocked, amazed, startled, confused |
| fear | afraid, nervous, anxious, worried, scared |

#### 4.3.2 LLM Mapping

基于大模型语义理解完成开放情绪词到标准标签的映射。

要求：

- 输入开放情绪词、情绪描述和标准标签集合。
- 输出唯一标准标签。
- 输出映射理由。
- 输出置信度。

#### 4.3.3 Hybrid Mapping

结合 Dictionary Mapping 和 LLM Mapping。

规则建议：

1. 若开放情绪词能被词表覆盖，则优先采用词表映射。
2. 若多个开放情绪词映射到同一标签，则直接输出该标签。
3. 若多个开放情绪词映射到不同标签，则使用多数投票。
4. 若词表无法覆盖或投票冲突，则调用 LLM Mapping。
5. 若 LLM Mapping 结果置信度较低，则标记为 uncertain，供错误分析使用。

### 4.4 统计与绘图交付内容

B 需要在**本周六前**完成以下结果文件和图表：

```text
analysis/
├── metrics_summary.csv
├── per_class_metrics.csv
├── emotion_word_frequency.csv
├── emotion_diversity_stats.csv
├── perturbation_shift_stats.csv
├── error_cases.csv
└── error_type_summary.csv

figures/
├── performance_comparison.png
├── confusion_matrix.png
├── emotion_word_frequency.png
├── emotion_wordcloud_global.png
├── perturbation_shift.png
└── error_type_distribution.png

presentation/
├── final_presentation.pptx
└── presentation_script.md
```

### 4.5 PPT 制作要求

PPT 建议结构如下：

1. 项目背景与任务定义；
2. 方法框架图；
3. 数据集与样本构造；
4. 闭集分类与开放词汇生成方法；
5. 三种映射方案；
6. 不同方法分类性能对比；
7. 开放情绪词多样性分析；
8. 模态扰动漂移分析；
9. 混淆矩阵与错误类型归因；
10. 典型样例展示；
11. 结论与不足。

PPT 内容要求：

- 不堆叠大段文字。
- 使用图表替代长句说明。
- 样例结果应包含输入、模型输出、映射结果和简短分析。
- 关键结论要突出显示。
- 配套简短文稿。

### 4.6 B 的完成标准

B 的任务完成标准如下：

- 所有 prompt 已设计并及时发给 A。
- 三种映射方案已完成。
- 实验结果已统计完成。
- 所有核心指标已生成。
- 图表已绘制完成。
- PPT 已完成。
- PPT 文稿已完成。
- 所有材料在本周六前发送至项目群。

---

## 5. 成员 C 分工：报告撰写、课堂 Pre 与提交要求确认

### 5.1 主要职责

C 负责最终报告撰写、课堂 pre、讲稿完善和提交要求确认。C 需要在周日集中完成展示内容梳理和讲稿打磨，并在下周一完成课堂汇报。

具体任务包括：

1. **负责下周一进行 pre**
   - 熟悉项目背景、方法流程和核心实验结果。
   - 结合 B 制作的 PPT 完成课堂展示。
   - 重点讲清楚项目的任务定义、方法设计、实验发现和结论。
   - 准备回答老师关于数据集、模型、映射方法、实验指标和错误分析的问题。

2. **撰写最终报告**
   - 完成项目背景与研究意义。
   - 整理相关工作与 AffectGPT / OV-MER 思路说明。
   - 描述数据集、模型选择和实验流程。
   - 说明闭集分类、开放词汇生成和标签映射方法。
   - 汇总 B 统计的实验结果和图表。
   - 撰写开放情绪词多样性分析。
   - 撰写模态扰动漂移分析。
   - 撰写混淆矩阵与错误类型归因。
   - 总结项目结论、不足和后续改进方向。

3. **确认最终提交要求**
   - 向课程负责人或助教确认最终是否需要提交代码。
   - 确认代码是否需要打包、提交仓库链接或现场展示。
   - 确认 pre 是否需要展示代码、demo 或运行结果。
   - 确认最终提交材料包括哪些内容，例如报告、PPT、代码、实验结果、README 等。
   - 将确认结果及时发送至项目群，方便 A 和 B 补充材料。

4. **周日进行 pre 准备和讲稿完善**
   - 检查 PPT 逻辑是否连贯。
   - 在 **B** 交付的汇报讲稿基础上进行润色。
   - 熟悉每页 PPT 的讲解重点。
   - 对实验结果和典型案例进行口头解释准备。
   - 进行至少一次完整模拟汇报。
   - 说明每位成员具体贡献

### 5.2 C 需要向老师或助教确认的问题

C 需要尽早确认以下问题：

1. 最终是否需要提交代码？
2. 若需要提交代码，提交方式是压缩包、GitHub/Gitee 链接还是课程平台上传？
4. 是否需要提交模型调用日志或实验结果原始文件？
5. 课堂 pre 是否需要现场展示代码？ demo？
8. 最终报告是否有格式要求、页数要求或模板？

### 5.3 C 的完成标准

C 的任务完成标准如下：

- 已确认最终提交要求。
- 已确认 pre 是否需要展示代码或 demo。
- 已熟悉项目核心方法与结果。
- 已完成周日 pre 准备。
- 已在下周一完成课堂展示。
- 已完成最终报告。

---

## 6. 时间安排与交付节点

| 时间节点 | 负责人 | 任务 | 交付内容 |
|---|---|---|---|
| 项目启动后尽早 | B | 设计 prompt 和映射方案并发给 A | prompt 文件、映射词表、映射规则 |
| 本周中前 | A | 建仓库、整理数据、完成代码框架 | 仓库链接、目录结构、初版脚本 |
| 本周中后 | A | 跑闭集分类和开放词汇实验 | 原始输出、解析结果 |
| 本周中后 | A / B | 跑映射和扰动实验 | 映射结果、扰动实验结果 |
| 本周六前 | A | 交付完整实验结果给 B | `results/final_for_B/` |
| 本周六前 | B | 完成统计、绘图和 PPT 初稿 | 图表、统计表、PPT、文稿 |
| 本周六前 | A / B | 将所有材料发至项目群 | 代码、结果、图表、PPT |
| 周日 | C | 完成 pre 准备和讲稿完善 |  |
| 报告ddl前 | C | 完成报告撰写 |  |
| 下周一 | C | 课堂 pre | 汇报展示 |

## 注意

1. B 的 prompt 和映射方案需要尽早交付给 A，否则 A 无法统一实验设置。

2. A 的实验结果需要在**本周四晚前**交付给 B，否则 B 无法完成统计、绘图和 PPT。

3. B 的 PPT 和文稿需要在**本周六前**交付给 C，否则 C 周日无法充分准备 pre。

4. C 需要尽早确认提交要求，否则 A 和 B 可能遗漏代码、demo 或结果文件。

5. 所有交付内容需要在群里或仓库同步，避免只存放在个人电脑中。

   

