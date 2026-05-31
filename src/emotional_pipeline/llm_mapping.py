from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from .batch import BatchClient, read_jsonl, write_jsonl
from .config import LABELS, Paths, api_config
from .csvio import read_csv, write_csv
from .parse import MAPPING_INPUT_FIELDS, OPEN_FIELDS, extract_content, mapping_inputs, parse_json_text, write_shift_stats
from .prompts import Message, batch_line


LLM_MAPPING_METHOD = "llm_mapping_fallback"
LLM_MAPPING_CANDIDATES = {"heuristic_llm_fallback", "needs_llm_mapping"}


def run_llm_mapping_fallbacks(root: Path, poll_seconds: int = 300, run_id: str | None = None) -> Dict[str, object]:
    paths = Paths.from_root(root)
    paths.ensure()
    config = api_config(root)
    if not config.enabled:
        raise RuntimeError("Qwen/DashScope API config is missing.")

    open_path = paths.parsed_outputs / "open_vocab_results.csv"
    perturb_path = paths.parsed_outputs / "perturbation_results.csv"
    if not open_path.exists() or not perturb_path.exists():
        raise RuntimeError("Parsed open/perturbation results are missing; run or fetch the main batch first.")

    open_rows = read_csv(open_path)
    perturb_rows = read_csv(perturb_path)
    candidates = collect_candidates(open_rows, perturb_rows)
    if not candidates:
        summary = {
            "candidate_count": 0,
            "applied_count": 0,
            "batch_status": "not_needed",
            "final_mapping_methods": mapping_method_counts(open_rows, perturb_rows),
        }
        write_mapping_summary(paths, summary)
        update_last_run_summary(paths, summary)
        return summary

    run_name = run_id or read_run_id(paths) or "latest"
    request_path = paths.raw_outputs / "batch_inputs" / f"{run_name}_llm_mapping_requests.jsonl"
    output_path = paths.raw_outputs / f"{run_name}_llm_mapping_output.jsonl"
    state_path = paths.logs / "llm_mapping_state.json"
    requests = [
        batch_line(f"llm_mapping:{index:04d}", config.model, llm_mapping_messages(candidate["row"]))
        for index, candidate in enumerate(candidates)
    ]
    write_jsonl(request_path, requests)

    status = BatchClient(config).submit_and_wait(request_path, output_path, state_path, poll_seconds=poll_seconds)
    if status not in {"completed", "succeeded", "success"}:
        summary = {
            "candidate_count": len(candidates),
            "applied_count": 0,
            "batch_status": status,
            "request_path": str(request_path.relative_to(root)),
            "output_path": str(output_path.relative_to(root)),
        }
        write_mapping_summary(paths, summary)
        return summary

    raw_rows = read_jsonl(output_path)
    write_llm_mapping_raw_outputs(paths, run_name, raw_rows)
    applied, invalid = apply_mapping_outputs(candidates, raw_rows)
    write_result_tables(paths, open_rows, perturb_rows)
    summary = {
        "candidate_count": len(candidates),
        "output_rows": len(raw_rows),
        "applied_count": applied,
        "invalid_count": invalid,
        "batch_status": status,
        "model": config.model,
        "request_path": str(request_path.relative_to(root)),
        "output_path": str(output_path.relative_to(root)),
        "final_mapping_methods": mapping_method_counts(open_rows, perturb_rows),
    }
    write_mapping_summary(paths, summary)
    update_last_run_summary(paths, summary)
    return summary


def collect_candidates(open_rows: List[Dict[str, str]], perturb_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for source, rows in [("open_vocab", open_rows), ("perturbation", perturb_rows)]:
        for row in rows:
            if row.get("mapping_method") in LLM_MAPPING_CANDIDATES:
                candidates.append({"source": source, "row": row})
    return candidates


def mapping_method_counts(open_rows: List[Dict[str, str]], perturb_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    open_counts = Counter(row.get("mapping_method", "") for row in open_rows)
    perturb_counts = Counter(row.get("mapping_method", "") for row in perturb_rows)
    return {
        "open_vocab": dict(open_counts),
        "perturbation": dict(perturb_counts),
        "total_llm_mapping_fallback": open_counts.get(LLM_MAPPING_METHOD, 0) + perturb_counts.get(LLM_MAPPING_METHOD, 0),
        "remaining_llm_candidates": sum(open_counts.get(method, 0) + perturb_counts.get(method, 0) for method in LLM_MAPPING_CANDIDATES),
    }


def llm_mapping_messages(row: Dict[str, str]) -> List[Message]:
    free_emotions = row.get("free_emotions_norm") or row.get("free_emotions") or "none"
    prompt = "\n".join(
        [
            "Map the fine-grained emotion words to exactly one MELD coarse emotion label.",
            f"Allowed labels: {', '.join(LABELS)}.",
            "The value of mapped_label MUST be exactly one of the allowed labels.",
            "Do not return fine-grained words such as mocking, urgency, desperation, embarrassment, or sympathy.",
            "If the fine-grained emotion is outside the allowed set, choose the closest allowed coarse label.",
            "Use the utterance and context only as semantic evidence. Do not invent labels.",
            "Useful mapping hints: desperation/urgency/concern -> fear; mocking/frustration/disapproval -> anger; embarrassment/awkwardness -> disgust or sadness; sympathy/apology/regret -> sadness; anticipation/skepticism -> surprise.",
            "",
            f"Speaker: {row.get('speaker', '')}",
            f"Utterance: {row.get('utterance', '')}",
            f"Conversation context: {row.get('context', '') or 'none'}",
            f"Fine-grained emotions: {free_emotions}",
            "",
            'Return JSON only: {"mapped_label":"one allowed label","confidence":0.0,"reason":"brief mapping reason"}',
        ]
    )
    return [
        {
            "role": "system",
            "content": "You are an emotion label mapping assistant. Return JSON only.",
        },
        {"role": "user", "content": prompt},
    ]


def apply_mapping_outputs(candidates: List[Dict[str, object]], raw_rows: List[Dict[str, object]]) -> Tuple[int, int]:
    by_id = {str(raw.get("custom_id", "")): raw for raw in raw_rows}
    applied = 0
    invalid = 0
    for index, candidate in enumerate(candidates):
        raw = by_id.get(f"llm_mapping:{index:04d}")
        if not raw:
            invalid += 1
            continue
        parsed = parse_json_text(extract_content(raw))
        label = str(parsed.get("mapped_label") or parsed.get("label") or "").strip().lower()
        if label not in LABELS:
            invalid += 1
            continue
        row = candidate["row"]
        if not isinstance(row, dict):
            invalid += 1
            continue
        row["mapped_label"] = label
        row["mapping_method"] = LLM_MAPPING_METHOD
        row["mapping_confidence"] = str(parsed.get("confidence", ""))
        row["mapping_reason"] = str(parsed.get("mapping_reason") or parsed.get("reason") or "Mapped by LLM fallback.")
        row["is_correct"] = str(label == str(row.get("gold_label", ""))).lower()
        applied += 1
    return applied, invalid


def write_result_tables(paths: Paths, open_rows: List[Dict[str, str]], perturb_rows: List[Dict[str, str]]) -> None:
    write_csv(paths.parsed_outputs / "open_vocab_results.csv", open_rows, OPEN_FIELDS)
    write_csv(paths.parsed_outputs / "mapping_inputs.csv", mapping_inputs(open_rows), MAPPING_INPUT_FIELDS)
    write_csv(paths.parsed_outputs / "perturbation_results.csv", perturb_rows, OPEN_FIELDS)
    write_shift_stats(paths.parsed_outputs / "perturbation_shift_stats.csv", perturb_rows)
    for name in ["open_vocab_results.csv", "mapping_inputs.csv", "perturbation_results.csv"]:
        source = paths.parsed_outputs / name
        target = paths.final_for_b / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def write_llm_mapping_raw_outputs(paths: Paths, run_id: str, rows: List[Dict[str, object]]) -> None:
    output_dir = paths.raw_outputs / "llm_mapping"
    write_jsonl(output_dir / f"{run_id}_llm_mapping_raw.jsonl", rows)
    write_jsonl(output_dir / "latest_llm_mapping_raw.jsonl", rows)


def write_mapping_summary(paths: Paths, summary: Dict[str, object]) -> None:
    (paths.logs / "llm_mapping_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def update_last_run_summary(paths: Paths, llm_mapping_summary: Dict[str, object]) -> None:
    summary_path = paths.logs / "last_run_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["llm_mapping"] = llm_mapping_summary
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def read_run_id(paths: Paths) -> str:
    summary_path = paths.logs / "last_run_summary.json"
    if not summary_path.exists():
        return ""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return str(summary.get("run_id", ""))
