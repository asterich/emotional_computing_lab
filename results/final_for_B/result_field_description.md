# Result Field Description

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本编号。 |
| `dialogue_id` | MELD 对话编号或 demo 对话编号。 |
| `utterance_id` | 当前话语在对话中的编号。 |
| `speaker` | 说话人。 |
| `utterance` | 当前待识别话语。 |
| `context` | 前若干轮对话上下文。 |
| `gold_label` | MELD 七类标准标签。 |
| `prompt_type` | closed_set / open_vocab / perturbation。 |
| `input_condition` | 输入模态条件，如 text-only、text + context、text + context + image。 |
| `model_name` | 调用或模拟的模型名称。 |
| `custom_id` | Batch 请求唯一编号，格式为 experiment:sample_id:condition。 |
| `raw_output` | 模型返回的原始文本内容。 |
| `predicted_label` | 闭集分类预测标签。 |
| `free_emotions` | 开放词汇原始情绪词。 |
| `free_emotions_norm` | 标准化后的开放情绪词。 |
| `mapped_label` | 开放词汇映射后的标准标签。 |
| `mapping_method` | dictionary / hybrid fallback 等映射方法。 |
| `confidence` | 模型输出置信度。 |
| `reason` | 模型输出解释。 |
| `is_correct` | 预测或映射标签是否等于 gold_label。 |
| `is_demo` | true 表示缺少真实数据/API 完成结果时由 demo fallback 生成。 |
