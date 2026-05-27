from __future__ import annotations

import json
from typing import Dict, List

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


def closed_set_messages(sample: Dict[str, object], condition: str = "text + context + image") -> List[Dict[str, str]]:
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
            "content": (
                "Classify the speaker's emotion into exactly one MELD label.\n"
                f"Allowed labels: {', '.join(LABELS)}.\n"
                "Use the utterance as primary evidence and use context or visual information when available.\n\n"
                f"{user_input(sample, condition)}\n\n"
                'Output JSON format: {"predicted_label":"one allowed label","confidence":0.0,"reason":"brief explanation"}'
            ),
        },
    ]


def open_vocab_messages(sample: Dict[str, object], condition: str = "text + context + image") -> List[Dict[str, str]]:
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
            "content": (
                "Identify the speaker's fine-grained emotions.\n"
                "Important requirements:\n"
                "1. Do not directly choose from fixed MELD labels.\n"
                "2. Generate 1 to 3 concise English emotion words or short phrases.\n"
                "3. Use only the information provided in the current input condition.\n"
                "4. If information is missing, do not assume it.\n\n"
                f"{user_input(sample, condition)}\n\n"
                'Output JSON format: {"free_emotions":["emotion_1","emotion_2"],"confidence":0.0,"reason":"brief explanation"}'
            ),
        },
    ]


def request_body(model: str, messages: List[Dict[str, str]]) -> Dict[str, object]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }


def batch_line(custom_id: str, model: str, messages: List[Dict[str, str]]) -> Dict[str, object]:
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
