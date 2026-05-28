from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Union

from .config import LABELS


def user_input(sample: Dict[str, object], condition: str) -> str:
    speaker = sample.get("speaker", "")
    utterance = sample.get("utterance", "")
    context = sample.get("context", "")
    frames = sample.get("frame_paths", "") or "none"

    if condition == "text-only":
        context = "none"
        frames = "none"
    elif condition == "text + context":
        frames = "none"
    elif condition == "text + context + image":
        frames = frames or "none"

    return "\n".join(
        [
            f"Input condition: {condition}",
            "Available input:",
            f"- Speaker: {speaker}",
            f"- Utterance: {utterance}",
            f"- Conversation context: {context or 'none'}",
            f"- Visual information: {frames}",
        ]
    )


Message = Dict[str, Union[str, List[Dict[str, object]]]]


def closed_set_messages(sample: Dict[str, object], condition: str = "text + context + image", root: Path | None = None) -> List[Message]:
    prompt = (
        "Classify the speaker's emotion into exactly one MELD label.\n"
        f"Allowed labels: {', '.join(LABELS)}.\n"
        "Use the utterance as primary evidence and use context or visual information when available.\n\n"
        f"{user_input(sample, condition)}\n\n"
        'Output JSON format: {"predicted_label":"one allowed label","confidence":0.0,"reason":"brief explanation"}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an emotion classification assistant. "
                "Return JSON only with predicted_label, confidence, and reason."
            ),
        },
        {
            "role": "user",
            "content": multimodal_content(prompt, sample, condition, root),
        },
    ]


def open_vocab_messages(sample: Dict[str, object], condition: str = "text + context + image", root: Path | None = None) -> List[Message]:
    prompt = (
        "Identify the speaker's fine-grained emotions.\n"
        "Important requirements:\n"
        "1. Do not directly choose from fixed MELD labels.\n"
        "2. Generate 1 to 3 concise English emotion words or short phrases.\n"
        "3. Use only the information provided in the current input condition.\n"
        "4. If information is missing, do not assume it.\n\n"
        f"{user_input(sample, condition)}\n\n"
        'Output JSON format: {"free_emotions":["emotion_1","emotion_2"],"confidence":0.0,"reason":"brief explanation"}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an emotion analysis assistant. "
                "Return JSON only with free_emotions, confidence, and reason."
            ),
        },
        {
            "role": "user",
            "content": multimodal_content(prompt, sample, condition, root),
        },
    ]


def multimodal_content(prompt: str, sample: Dict[str, object], condition: str, root: Path | None) -> Union[str, List[Dict[str, object]]]:
    if "image" not in condition or root is None:
        return prompt
    image_parts = []
    for path in frame_paths(sample, root)[:3]:
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    if not image_parts:
        return prompt
    return [{"type": "text", "text": prompt}, *image_parts]


def frame_paths(sample: Dict[str, object], root: Path) -> List[Path]:
    raw = str(sample.get("frame_paths", ""))
    values = [value for value in raw.split("|") if value]
    paths = []
    for value in values:
        path = Path(value)
        paths.append(path if path.is_absolute() else root / path)
    return paths


def request_body(model: str, messages: List[Message]) -> Dict[str, object]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }


def batch_line(custom_id: str, model: str, messages: List[Message]) -> Dict[str, object]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": request_body(model, messages),
    }


def write_prompt_files(prompt_dir) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "closed_set_prompt.md": closed_set_messages(
            {"speaker": "{speaker}", "utterance": "{utterance}", "context": "{context}", "frame_paths": "{image_frames}"}
        )[1]["content"],
        "open_vocab_prompt.md": open_vocab_messages(
            {"speaker": "{speaker}", "utterance": "{utterance}", "context": "{context}", "frame_paths": "{image_frames}"}
        )[1]["content"],
        "perturbation_prompt.md": open_vocab_messages(
            {"speaker": "{speaker}", "utterance": "{utterance}", "context": "{context}", "frame_paths": "{image_frames}"},
            condition="{condition_name}",
        )[1]["content"],
        "allowed_labels.json": json.dumps({"labels": LABELS}, indent=2),
    }
    for name, content in files.items():
        (prompt_dir / name).write_text(str(content) + "\n", encoding="utf-8")
