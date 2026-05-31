from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .batch import BatchClient, read_jsonl, write_jsonl
from .config import LABELS, Paths, api_config
from .csvio import read_csv, write_csv
from .mapping import WORD_TO_LABEL, normalize_emotions
from .parse import extract_content, parse_json_text
from .prompts import Message, batch_line


STRATEGY_RESULT_FIELDS = [
    "source_experiment",
    "mapping_strategy",
    "sample_id",
    "dialogue_id",
    "utterance_id",
    "speaker",
    "utterance",
    "context",
    "gold_label",
    "input_condition",
    "free_emotions",
    "free_emotions_norm",
    "mapped_label",
    "mapping_covered",
    "mapping_confidence",
    "mapping_reason",
    "is_correct",
]

STRATEGY_METRIC_FIELDS = [
    "source_experiment",
    "input_condition",
    "mapping_strategy",
    "n",
    "covered",
    "coverage_rate",
    "accuracy",
    "mapping_error_rate",
    "macro_f1",
    "weighted_f1",
]


def run_mapping_strategy_experiments(root: Path, poll_seconds: int = 600, run_id: str | None = None) -> Dict[str, object]:
    paths = Paths.from_root(root)
    paths.ensure()
    config = api_config(root)
    if not config.enabled:
        raise RuntimeError("Qwen/DashScope API config is missing.")

    open_rows = read_csv(paths.parsed_outputs / "open_vocab_results.csv")
    perturb_rows = read_csv(paths.parsed_outputs / "perturbation_results.csv")
    base_rows = [strategy_base_row("open_vocab", row) for row in open_rows]
    base_rows.extend(strategy_base_row("perturbation", row) for row in perturb_rows)
    if not base_rows:
        raise RuntimeError("No open-vocabulary rows found. Run or fetch the main experiment first.")

    run_name = run_id or read_run_id(paths) or "latest"
    request_path = paths.raw_outputs / "batch_inputs" / f"{run_name}_mapping_strategy_llm_requests.jsonl"
    output_path = paths.raw_outputs / f"{run_name}_mapping_strategy_llm_output.jsonl"
    state_path = paths.logs / "mapping_strategy_state.json"
    requests = [
        batch_line(f"strategy_llm:{index:04d}", config.model, strategy_llm_messages(row))
        for index, row in enumerate(base_rows)
    ]
    write_jsonl(request_path, requests)
    status = BatchClient(config).submit_and_wait(request_path, output_path, state_path, poll_seconds=poll_seconds)
    if status not in {"completed", "succeeded", "success"}:
        summary = {
            "batch_status": status,
            "base_rows": len(base_rows),
            "applied_llm_rows": 0,
            "request_path": str(request_path.relative_to(root)),
            "output_path": str(output_path.relative_to(root)),
        }
        write_summary(paths, summary)
        return summary

    raw_rows = read_jsonl(output_path)
    write_jsonl(paths.raw_outputs / "mapping_strategies" / f"{run_name}_mapping_strategy_llm_raw.jsonl", raw_rows)
    write_jsonl(paths.raw_outputs / "mapping_strategies" / "latest_mapping_strategy_llm_raw.jsonl", raw_rows)
    llm_results = parse_llm_strategy_outputs(raw_rows)
    strategy_rows = build_strategy_rows(base_rows, llm_results)
    metric_rows = build_metric_rows(strategy_rows)
    write_outputs(paths, strategy_rows, metric_rows)
    summary = {
        "batch_status": status,
        "base_rows": len(base_rows),
        "output_rows": len(raw_rows),
        "strategy_rows": len(strategy_rows),
        "applied_llm_rows": sum(1 for item in llm_results.values() if item["covered"]),
        "invalid_llm_rows": sum(1 for item in llm_results.values() if not item["covered"]),
        "request_path": str(request_path.relative_to(root)),
        "output_path": str(output_path.relative_to(root)),
        "result_csv": str((paths.parsed_outputs / "mapping_strategy_results.csv").relative_to(root)),
        "metrics_csv": str((paths.parsed_outputs / "mapping_strategy_metrics.csv").relative_to(root)),
    }
    write_summary(paths, summary)
    update_last_run_summary(paths, summary)
    return summary


def strategy_base_row(source: str, row: Dict[str, str]) -> Dict[str, str]:
    return {
        "source_experiment": source,
        "sample_id": row.get("sample_id", ""),
        "dialogue_id": row.get("dialogue_id", ""),
        "utterance_id": row.get("utterance_id", ""),
        "speaker": row.get("speaker", ""),
        "utterance": row.get("utterance", ""),
        "context": row.get("context", ""),
        "gold_label": row.get("gold_label", ""),
        "input_condition": row.get("input_condition", ""),
        "free_emotions": row.get("free_emotions", ""),
        "free_emotions_norm": row.get("free_emotions_norm", ""),
    }


def dictionary_only_map(row: Dict[str, str]) -> Dict[str, object]:
    labels = []
    words = normalize_emotions(row.get("free_emotions_norm", "").replace("|", ","))
    for word in words:
        if word in WORD_TO_LABEL:
            labels.append(WORD_TO_LABEL[word])
    if not labels:
        return {
            "mapped_label": "",
            "covered": False,
            "confidence": 0.0,
            "reason": "No normalized emotion word matched the dictionary.",
        }
    counts = Counter(labels)
    top = counts.most_common()
    reason = "Dictionary majority vote." if len(top) == 1 or top[0][1] > top[1][1] else "Dictionary tie; used first generated mapped emotion."
    return {
        "mapped_label": top[0][0],
        "covered": True,
        "confidence": min(0.95, 0.65 + top[0][1] * 0.1),
        "reason": reason,
    }


def strategy_llm_messages(row: Dict[str, str]) -> List[Message]:
    prompt = "\n".join(
        [
            "Map fine-grained emotion words to exactly one MELD coarse emotion label.",
            f"Allowed labels: {', '.join(LABELS)}.",
            "The mapped_label value MUST be exactly one allowed label.",
            "If the fine-grained emotion is outside the label set, choose the closest coarse label.",
            "Useful hints: desperation/urgency/concern -> fear; mocking/frustration/disapproval -> anger; embarrassment/awkwardness -> disgust or sadness; sympathy/apology/regret -> sadness; anticipation/skepticism/confusion -> surprise.",
            "",
            f"Speaker: {row.get('speaker', '')}",
            f"Utterance: {row.get('utterance', '')}",
            f"Conversation context: {row.get('context', '') or 'none'}",
            f"Fine-grained emotions: {row.get('free_emotions_norm', '') or row.get('free_emotions', '') or 'none'}",
            "",
            'Return JSON only: {"mapped_label":"one allowed label","confidence":0.0,"reason":"brief mapping reason"}',
        ]
    )
    return [
        {"role": "system", "content": "You are an emotion mapping assistant. Return JSON only."},
        {"role": "user", "content": prompt},
    ]


def parse_llm_strategy_outputs(raw_rows: Iterable[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
    results: Dict[int, Dict[str, object]] = {}
    for raw in raw_rows:
        custom_id = str(raw.get("custom_id", ""))
        try:
            _, index_value = custom_id.split(":", 1)
            index = int(index_value)
        except ValueError:
            continue
        parsed = parse_json_text(extract_content(raw))
        label = str(parsed.get("mapped_label") or parsed.get("label") or "").strip().lower()
        covered = label in LABELS
        results[index] = {
            "mapped_label": label if covered else "",
            "covered": covered,
            "confidence": parsed.get("confidence", ""),
            "reason": parsed.get("mapping_reason") or parsed.get("reason") or ("Invalid LLM mapping label." if not covered else ""),
        }
    return results


def build_strategy_rows(base_rows: List[Dict[str, str]], llm_results: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index, base in enumerate(base_rows):
        dictionary = dictionary_only_map(base)
        llm = llm_results.get(
            index,
            {"mapped_label": "", "covered": False, "confidence": "", "reason": "Missing LLM mapping output."},
        )
        hybrid = dictionary if dictionary["covered"] else llm
        rows.append(strategy_row(base, "dictionary", dictionary))
        rows.append(strategy_row(base, "llm", llm))
        rows.append(strategy_row(base, "hybrid", hybrid))
    return rows


def strategy_row(base: Dict[str, str], strategy: str, result: Dict[str, object]) -> Dict[str, object]:
    label = str(result.get("mapped_label", ""))
    covered = bool(result.get("covered", False))
    return {
        **base,
        "mapping_strategy": strategy,
        "mapped_label": label if covered else "unmapped",
        "mapping_covered": str(covered).lower(),
        "mapping_confidence": result.get("confidence", ""),
        "mapping_reason": result.get("reason", ""),
        "is_correct": str(covered and label == base.get("gold_label", "")).lower(),
    }


def build_metric_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_rows = []
    groups = sorted({(str(row["source_experiment"]), str(row["input_condition"]), str(row["mapping_strategy"])) for row in rows})
    for source, condition, strategy in groups:
        group_rows = [
            row
            for row in rows
            if row["source_experiment"] == source and row["input_condition"] == condition and row["mapping_strategy"] == strategy
        ]
        metric_rows.append(metric_row(source, condition, strategy, group_rows))
    return metric_rows


def metric_row(source: str, condition: str, strategy: str, rows: List[Dict[str, object]]) -> Dict[str, object]:
    n = len(rows)
    covered = sum(1 for row in rows if str(row.get("mapping_covered")) == "true")
    correct = sum(1 for row in rows if str(row.get("is_correct")) == "true")
    macro_f1, weighted_f1 = f1_scores(rows)
    accuracy = correct / n if n else 0.0
    return {
        "source_experiment": source,
        "input_condition": condition,
        "mapping_strategy": strategy,
        "n": n,
        "covered": covered,
        "coverage_rate": round(covered / n, 6) if n else 0.0,
        "accuracy": round(accuracy, 6),
        "mapping_error_rate": round(1.0 - accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
    }


def f1_scores(rows: List[Dict[str, object]]) -> Tuple[float, float]:
    per_label = []
    for label in LABELS:
        tp = sum(1 for row in rows if row.get("gold_label") == label and row.get("mapped_label") == label)
        fp = sum(1 for row in rows if row.get("gold_label") != label and row.get("mapped_label") == label)
        fn = sum(1 for row in rows if row.get("gold_label") == label and row.get("mapped_label") != label)
        support = sum(1 for row in rows if row.get("gold_label") == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append((support, f1))
    total = sum(support for support, _ in per_label)
    macro_f1 = sum(f1 for _, f1 in per_label) / len(per_label) if per_label else 0.0
    weighted_f1 = sum(support * f1 for support, f1 in per_label) / total if total else 0.0
    return macro_f1, weighted_f1


def write_outputs(paths: Paths, strategy_rows: List[Dict[str, object]], metric_rows: List[Dict[str, object]]) -> None:
    result_path = paths.parsed_outputs / "mapping_strategy_results.csv"
    metric_path = paths.parsed_outputs / "mapping_strategy_metrics.csv"
    write_csv(result_path, strategy_rows, STRATEGY_RESULT_FIELDS)
    write_csv(metric_path, metric_rows, STRATEGY_METRIC_FIELDS)
    for path in [result_path, metric_path]:
        target = paths.final_for_b / path.name
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def write_summary(paths: Paths, summary: Dict[str, object]) -> None:
    (paths.logs / "mapping_strategy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def update_last_run_summary(paths: Paths, summary: Dict[str, object]) -> None:
    summary_path = paths.logs / "last_run_summary.json"
    if not summary_path.exists():
        return
    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_summary["mapping_strategy_experiments"] = summary
    summary_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")


def read_run_id(paths: Paths) -> str:
    summary_path = paths.logs / "last_run_summary.json"
    if not summary_path.exists():
        return ""
    return str(json.loads(summary_path.read_text(encoding="utf-8")).get("run_id", ""))
