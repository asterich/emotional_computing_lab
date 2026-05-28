from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import LABELS, Paths
from .csvio import write_csv
from .mapping import dictionary_map, is_allowed_label, normalize_emotions


COMMON_FIELDS = [
    "sample_id",
    "dialogue_id",
    "utterance_id",
    "speaker",
    "utterance",
    "context",
    "gold_label",
    "prompt_type",
    "input_condition",
    "model_name",
    "custom_id",
    "raw_output",
    "confidence",
    "reason",
    "is_demo",
]

CLOSED_FIELDS = COMMON_FIELDS + ["predicted_label", "is_correct"]
OPEN_FIELDS = COMMON_FIELDS + [
    "free_emotions",
    "free_emotions_norm",
    "mapped_label",
    "mapping_method",
    "mapping_confidence",
    "mapping_reason",
    "is_correct",
]
MAPPING_INPUT_FIELDS = [
    "sample_id",
    "utterance",
    "context",
    "gold_label",
    "free_emotions",
    "free_emotions_norm",
    "mapped_label",
    "mapping_method",
    "mapping_confidence",
    "mapping_reason",
    "is_demo",
]


def parse_and_export(
    paths: Paths,
    samples: List[Dict[str, object]],
    output_rows: Iterable[Dict[str, object]],
    model: str,
) -> Dict[str, Path]:
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    closed_rows: List[Dict[str, object]] = []
    open_rows: List[Dict[str, object]] = []
    perturb_rows: List[Dict[str, object]] = []

    for raw in output_rows:
        custom_id = str(raw.get("custom_id", ""))
        if not custom_id:
            continue
        try:
            experiment, sample_id, condition = custom_id.split(":", 2)
        except ValueError:
            continue
        sample = samples_by_id.get(sample_id)
        if not sample:
            continue
        content_text = extract_content(raw)
        parsed = parse_json_text(content_text)
        base = base_row(sample, experiment, decode_condition(condition), model, custom_id, content_text, raw)
        if experiment == "closed_set":
            closed_rows.append(parse_closed(base, parsed))
        elif experiment == "open_vocab":
            open_rows.append(parse_open(base, parsed))
        elif experiment == "perturbation":
            perturb_rows.append(parse_open(base, parsed))

    paths_map = {
        "closed": paths.parsed_outputs / "closed_set_results.csv",
        "open": paths.parsed_outputs / "open_vocab_results.csv",
        "mapping": paths.parsed_outputs / "mapping_inputs.csv",
        "perturbation": paths.parsed_outputs / "perturbation_results.csv",
        "field_description": paths.final_for_b / "result_field_description.md",
    }

    write_csv(paths_map["closed"], closed_rows, CLOSED_FIELDS)
    write_csv(paths_map["open"], open_rows, OPEN_FIELDS)
    write_csv(paths_map["mapping"], mapping_inputs(open_rows), MAPPING_INPUT_FIELDS)
    write_csv(paths_map["perturbation"], perturb_rows, OPEN_FIELDS)
    write_shift_stats(paths.parsed_outputs / "perturbation_shift_stats.csv", perturb_rows)

    for key, path in paths_map.items():
        if key == "field_description":
            continue
        target = paths.final_for_b / path.name
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    write_field_description(paths_map["field_description"])
    return paths_map


def extract_content(raw: Dict[str, object]) -> str:
    response = raw.get("response")
    if isinstance(response, dict):
        body = response.get("body")
        if isinstance(body, dict):
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict):
                        return str(message.get("content", ""))
    if raw.get("raw_output"):
        return str(raw["raw_output"])
    return json.dumps(raw, ensure_ascii=False)


def parse_json_text(text: str) -> Dict[str, object]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


def base_row(
    sample: Dict[str, object],
    experiment: str,
    condition: str,
    model: str,
    custom_id: str,
    raw_output: str,
    raw: Dict[str, object],
) -> Dict[str, object]:
    condition_text = condition
    return {
        "sample_id": sample.get("sample_id", ""),
        "dialogue_id": sample.get("dialogue_id", ""),
        "utterance_id": sample.get("utterance_id", ""),
        "speaker": sample.get("speaker", ""),
        "utterance": sample.get("utterance", ""),
        "context": sample.get("context", ""),
        "gold_label": sample.get("gold_label", ""),
        "prompt_type": experiment,
        "input_condition": condition_text,
        "model_name": model,
        "custom_id": custom_id,
        "raw_output": one_line(raw_output),
        "confidence": "",
        "reason": "",
        "is_demo": str(raw.get("is_demo", False)).lower(),
    }


def decode_condition(value: str) -> str:
    return value.replace("_plus_", " + ").replace("text_only", "text-only").replace("_", " ")


def one_line(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n")


def parse_closed(base: Dict[str, object], parsed: Dict[str, object]) -> Dict[str, object]:
    label = str(parsed.get("predicted_label") or parsed.get("label") or "").strip().lower()
    if not is_allowed_label(label):
        label = "neutral"
    base.update(
        {
            "predicted_label": label,
            "confidence": parsed.get("confidence", ""),
            "reason": parsed.get("reason", ""),
            "is_correct": str(label == str(base["gold_label"])).lower(),
        }
    )
    return base


def parse_open(base: Dict[str, object], parsed: Dict[str, object]) -> Dict[str, object]:
    free = parsed.get("free_emotions") or parsed.get("emotions") or []
    if not free:
        parsed = {**extract_open_fields(str(base.get("raw_output", ""))), **parsed}
        free = parsed.get("free_emotions") or parsed.get("emotions") or []
    normalized = normalize_emotions(free)
    mapped, method, mapping_reason, mapping_confidence = dictionary_map(
        normalized, utterance=str(base.get("utterance", "")), context=str(base.get("context", ""))
    )
    base.update(
        {
            "free_emotions": free if isinstance(free, list) else str(free),
            "free_emotions_norm": normalized,
            "mapped_label": mapped,
            "mapping_method": method,
            "mapping_confidence": mapping_confidence,
            "mapping_reason": mapping_reason,
            "confidence": parsed.get("confidence", ""),
            "reason": parsed.get("reason", ""),
            "is_correct": str(mapped == str(base["gold_label"])).lower(),
        }
    )
    return base


def extract_open_fields(text: str) -> Dict[str, object]:
    fields: Dict[str, object] = {}
    emotions_match = re.search(r'"(?:free_emotions|emotions)"\s*:\s*(\[[^\]]*\])', text, flags=re.S)
    if emotions_match:
        raw_list = emotions_match.group(1)
        try:
            value = json.loads(raw_list)
            if isinstance(value, list):
                fields["free_emotions"] = value
        except json.JSONDecodeError:
            quoted = re.findall(r'"([^"]+)"', raw_list)
            if quoted:
                fields["free_emotions"] = quoted
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if confidence_match:
        fields["confidence"] = confidence_match.group(1)
    reason_match = re.search(r'"reason"\s*:\s*("(?:\\.|[^"\\])*")', text, flags=re.S)
    if reason_match:
        try:
            fields["reason"] = json.loads(reason_match.group(1))
        except json.JSONDecodeError:
            pass
    return fields


def mapping_inputs(open_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "sample_id": row.get("sample_id", ""),
            "utterance": row.get("utterance", ""),
            "context": row.get("context", ""),
            "gold_label": row.get("gold_label", ""),
            "free_emotions": row.get("free_emotions", ""),
            "free_emotions_norm": row.get("free_emotions_norm", ""),
            "mapped_label": row.get("mapped_label", ""),
            "mapping_method": row.get("mapping_method", ""),
            "mapping_confidence": row.get("mapping_confidence", ""),
            "mapping_reason": row.get("mapping_reason", ""),
            "is_demo": row.get("is_demo", ""),
        }
        for row in open_rows
    ]


def write_shift_stats(path: Path, rows: List[Dict[str, object]]) -> None:
    by_sample: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), {})[str(row["input_condition"])] = row
    stats = []
    for condition in sorted({str(row["input_condition"]) for row in rows}):
        condition_rows = [row for row in rows if str(row["input_condition"]) == condition]
        if not condition_rows:
            continue
        correct = sum(1 for row in condition_rows if str(row.get("is_correct")) == "true")
        shifts = 0
        emotion_shifts = 0
        comparable = 0
        for sample_id, condition_map in by_sample.items():
            baseline = condition_map.get("text + context + image") or condition_map.get("text + context")
            current = condition_map.get(condition)
            if not baseline or not current or current is baseline:
                continue
            comparable += 1
            if baseline.get("mapped_label") != current.get("mapped_label"):
                shifts += 1
            if baseline.get("free_emotions_norm") != current.get("free_emotions_norm"):
                emotion_shifts += 1
        stats.append(
            {
                "input_condition": condition,
                "sample_count": len(condition_rows),
                "accuracy": round(correct / len(condition_rows), 4),
                "label_shift_rate": round(shifts / comparable, 4) if comparable else "",
                "emotion_word_shift_rate": round(emotion_shifts / comparable, 4) if comparable else "",
            }
        )
    write_csv(path, stats, ["input_condition", "sample_count", "accuracy", "label_shift_rate", "emotion_word_shift_rate"])


def write_field_description(path: Path) -> None:
    descriptions: List[Tuple[str, str]] = [
        ("sample_id", "样本编号。"),
        ("dialogue_id", "MELD 对话编号或 demo 对话编号。"),
        ("utterance_id", "当前话语在对话中的编号。"),
        ("speaker", "说话人。"),
        ("utterance", "当前待识别话语。"),
        ("context", "前若干轮对话上下文。"),
        ("gold_label", "MELD 七类标准标签。"),
        ("prompt_type", "closed_set / open_vocab / perturbation。"),
        ("input_condition", "输入模态条件，如 text-only、text + context、text + context + image。"),
        ("model_name", "调用或模拟的模型名称。"),
        ("custom_id", "Batch 请求唯一编号，格式为 experiment:sample_id:condition。"),
        ("raw_output", "模型返回的原始文本内容。"),
        ("predicted_label", "闭集分类预测标签。"),
        ("free_emotions", "开放词汇原始情绪词。"),
        ("free_emotions_norm", "标准化后的开放情绪词。"),
        ("mapped_label", "开放词汇映射后的标准标签。"),
        ("mapping_method", "dictionary / hybrid fallback 等映射方法。"),
        ("confidence", "模型输出置信度。"),
        ("reason", "模型输出解释。"),
        ("is_correct", "预测或映射标签是否等于 gold_label。"),
        ("is_demo", "true 表示缺少真实数据/API 完成结果时由 demo fallback 生成。"),
    ]
    body = ["# Result Field Description", "", "| 字段 | 含义 |", "|---|---|"]
    body.extend(f"| `{name}` | {desc} |" for name, desc in descriptions)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
