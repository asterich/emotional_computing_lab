from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

from .batch import BatchClient, read_jsonl, write_jsonl
from .config import Paths, api_config, now_run_id
from .data import SAMPLE_FIELDS, load_or_create_samples
from .demo import demo_batch_outputs
from .frames import extract_frames_if_possible
from .llm_mapping import run_llm_mapping_fallbacks
from .parse import parse_and_export
from .prompts import batch_line, closed_set_messages, open_vocab_messages, write_prompt_files
from .csvio import write_csv
from .csvio import read_csv


PERTURBATION_CONDITIONS = ["text-only", "text + context", "text + context + image"]


def run_pipeline(
    root: Path,
    n: int = 300,
    seed: int = 42,
    demo: bool = False,
    allow_fallback: bool = True,
    use_api: bool = True,
    force: bool = False,
    poll_seconds: int = 60,
) -> Dict[str, object]:
    paths = Paths.from_root(root)
    if force:
        clean_artifacts(root)
    paths.ensure()
    write_prompt_files(paths.prompts)

    config = api_config(root)
    samples = load_or_create_samples(paths, n=n, seed=seed, allow_demo=demo or allow_fallback)
    samples, frame_summary = extract_frames_if_possible(samples, paths)
    write_csv(paths.data_processed / "sampled_data.csv", samples, SAMPLE_FIELDS)
    write_csv(paths.final_for_b / "sampled_data.csv", samples, SAMPLE_FIELDS)

    run_id = now_run_id()
    requests = build_batch_requests(samples, config.model, root)
    input_path = paths.raw_outputs / "batch_inputs" / f"{run_id}_all_requests.jsonl"
    output_path = paths.raw_outputs / f"{run_id}_batch_output.jsonl"
    write_jsonl(input_path, requests)

    output_rows: List[Dict[str, object]] = []
    batch_status = "not_used"
    api_error = ""
    if use_api and config.enabled:
        state_path = paths.logs / "pipeline_state.json"
        try:
            status = BatchClient(config).submit_and_wait(input_path, output_path, state_path, poll_seconds=poll_seconds)
            batch_status = status or "not_submitted"
            if output_path.exists():
                output_rows = read_jsonl(output_path)
        except Exception as exc:
            api_error = str(exc)
            batch_status = "api_error"
            if state_path.exists() and "Network error for GET /batches/" in api_error:
                batch_status = "poll_error_batch_submitted"
    elif use_api and not config.enabled:
        batch_status = "missing_api_config"

    if not output_rows:
        if not allow_fallback:
            raise RuntimeError(
                "Batch did not produce output rows during this command. "
                f"status={batch_status}. api_error={api_error}. "
                "If a batch_id exists in logs/pipeline_state.json, rerun "
                "`uv run emotional-pipeline fetch-batch --poll-seconds 300`."
            )
        samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
        output_rows = demo_batch_outputs(requests, samples_by_id, config.model)
        batch_status = f"{batch_status}_demo_fallback"
        write_jsonl(output_path, output_rows)

    write_experiment_raw_outputs(paths, run_id, output_rows)
    exported = parse_and_export(paths, samples, output_rows, config.model)
    llm_mapping_summary = {}
    if use_api and config.enabled and output_rows:
        try:
            llm_mapping_summary = run_llm_mapping_fallbacks(root, poll_seconds=poll_seconds, run_id=run_id)
        except Exception as exc:
            llm_mapping_summary = {"batch_status": "error", "error": str(exc)}
    summary = {
        "run_id": run_id,
        "sample_count": len(samples),
        "request_count": len(requests),
        "model": config.model,
        "api_base_url": mask_url(config.base_url),
        "batch_status": batch_status,
        "api_error": api_error,
        "batch_input": str(input_path.relative_to(root)),
        "batch_output": str(output_path.relative_to(root)),
        "frame_extraction": frame_summary,
        "llm_mapping": llm_mapping_summary,
        "final_for_B": str(paths.final_for_b.relative_to(root)),
        "exports": {key: str(path.relative_to(root)) for key, path in exported.items()},
    }
    (paths.logs / "last_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_batch_requests(samples: List[Dict[str, object]], model: str, root: Path) -> List[Dict[str, object]]:
    requests: List[Dict[str, object]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        main_condition = "text + context + image" if sample.get("frame_paths") else "text + context"
        requests.append(
            batch_line(f"closed_set:{sample_id}:{condition_key(main_condition)}", model, closed_set_messages(sample, main_condition, root=root))
        )
        requests.append(
            batch_line(f"open_vocab:{sample_id}:{condition_key(main_condition)}", model, open_vocab_messages(sample, main_condition, root=root))
        )
        for condition in PERTURBATION_CONDITIONS:
            requests.append(
                batch_line(f"perturbation:{sample_id}:{condition_key(condition)}", model, open_vocab_messages(sample, condition, root=root))
            )
    return requests


def condition_key(value: str) -> str:
    return value.replace(" + ", "_plus_").replace("-", "_").replace(" ", "_")


def write_experiment_raw_outputs(paths: Paths, run_id: str, rows: List[Dict[str, object]]) -> None:
    buckets: Dict[str, List[Dict[str, object]]] = {"closed_set": [], "open_vocab": [], "perturbation": []}
    for row in rows:
        custom_id = str(row.get("custom_id", ""))
        experiment = custom_id.split(":", 1)[0]
        if experiment in buckets:
            buckets[experiment].append(row)
    for experiment, values in buckets.items():
        write_jsonl(paths.raw_outputs / experiment / f"{run_id}_{experiment}_raw.jsonl", values)
        write_jsonl(paths.raw_outputs / experiment / f"latest_{experiment}_raw.jsonl", values)


def status(root: Path) -> Dict[str, object]:
    paths = Paths.from_root(root)
    summary_path = paths.logs / "last_run_summary.json"
    state_path = paths.logs / "pipeline_state.json"
    return {
        "last_run_summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None,
        "batch_state": json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None,
        "frame_extraction": json.loads((paths.logs / "frame_extraction_summary.json").read_text(encoding="utf-8"))
        if (paths.logs / "frame_extraction_summary.json").exists()
        else None,
        "final_for_B_exists": paths.final_for_b.exists(),
        "final_for_B_files": sorted(path.name for path in paths.final_for_b.glob("*")) if paths.final_for_b.exists() else [],
    }


def fetch_pending_batch(root: Path, poll_seconds: int = 300) -> Dict[str, object]:
    paths = Paths.from_root(root)
    config = api_config(root)
    state_path = paths.logs / "pipeline_state.json"
    summary_path = paths.logs / "last_run_summary.json"
    if not config.enabled:
        raise RuntimeError("Qwen/DashScope API config is missing.")
    if not state_path.exists():
        raise RuntimeError("No batch state found. Run `emotional-pipeline run` first.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch_id = state.get("batch_id")
    if not batch_id:
        raise RuntimeError("No batch_id found in logs/pipeline_state.json.")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        output_path = root / summary.get("batch_output", state.get("output_path", "results/raw_outputs/latest_batch_output.jsonl"))
    else:
        summary = {}
        output_path = Path(str(state.get("output_path", paths.raw_outputs / "latest_batch_output.jsonl")))
        if not output_path.is_absolute():
            output_path = root / output_path

    status_value = BatchClient(config).wait_for_batch(str(batch_id), output_path, state_path, poll_seconds=poll_seconds)
    completed_statuses = {"completed", "succeeded", "success"}
    if status_value not in completed_statuses:
        return {
            "batch_id": batch_id,
            "batch_status": status_value,
            "downloaded": False,
            "message": "Batch has not completed yet; existing fallback output was left unchanged.",
        }

    output_rows = read_jsonl(output_path) if output_path.exists() else []
    if not output_rows:
        return {
            "batch_id": batch_id,
            "batch_status": status_value,
            "downloaded": False,
            "message": "Batch completed but no output rows were downloaded.",
        }

    samples_path = paths.data_processed / "sampled_data.csv"
    if not samples_path.exists():
        raise RuntimeError("Missing data/processed/sampled_data.csv; cannot parse downloaded batch output.")
    samples = read_csv(samples_path)
    run_id = str(summary.get("run_id") or output_path.name.replace("_batch_output.jsonl", ""))
    write_experiment_raw_outputs(paths, run_id, output_rows)
    exported = parse_and_export(paths, samples, output_rows, config.model)
    try:
        llm_mapping_summary = run_llm_mapping_fallbacks(root, poll_seconds=poll_seconds, run_id=run_id)
    except Exception as exc:
        llm_mapping_summary = {"batch_status": "error", "error": str(exc)}
    frame_summary_path = paths.logs / "frame_extraction_summary.json"
    updated = {
        **summary,
        "run_id": run_id,
        "sample_count": len(samples),
        "request_count": len(output_rows),
        "model": config.model,
        "api_base_url": mask_url(config.base_url),
        "batch_status": status_value,
        "batch_output": str(output_path.relative_to(root)),
        "frame_extraction": json.loads(frame_summary_path.read_text(encoding="utf-8"))
        if frame_summary_path.exists()
        else None,
        "llm_mapping": llm_mapping_summary,
        "final_for_B": str(paths.final_for_b.relative_to(root)),
        "downloaded_real_batch": True,
        "exports": {key: str(path.relative_to(root)) for key, path in exported.items()},
    }
    summary_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "batch_id": batch_id,
        "batch_status": status_value,
        "downloaded": True,
        "output_rows": len(output_rows),
        "batch_output": str(output_path.relative_to(root)),
        "final_for_B": str(paths.final_for_b.relative_to(root)),
    }


def clean_artifacts(root: Path) -> None:
    paths = Paths.from_root(root)
    for path in [paths.data_processed, paths.frames, paths.results, paths.logs, paths.prompts]:
        if path.exists():
            shutil.rmtree(path)


def mask_url(value: str) -> str:
    return value.split("?")[0]
