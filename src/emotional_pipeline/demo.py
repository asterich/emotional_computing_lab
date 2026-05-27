from __future__ import annotations

import json
from typing import Dict, Iterable, List

from .mapping import EMOTION_DICT


DEMO_REASON = {
    "anger": "The utterance expresses blame or frustration.",
    "disgust": "The wording indicates aversion toward the situation.",
    "sadness": "The speaker describes disappointment or loss.",
    "joy": "The utterance conveys positive excitement.",
    "neutral": "The utterance is mainly informational.",
    "surprise": "The speaker reacts to unexpected information.",
    "fear": "The speaker shows worry about a possible negative outcome.",
}


def demo_batch_outputs(requests: Iterable[Dict[str, object]], samples_by_id: Dict[str, Dict[str, object]], model: str) -> List[Dict[str, object]]:
    rows = []
    for request in requests:
        custom_id = str(request["custom_id"])
        experiment, sample_id, condition = custom_id.split(":", 2)
        sample = samples_by_id[sample_id]
        content = demo_content(experiment, sample, decode_condition(condition))
        rows.append(
            {
                "id": f"demo-{custom_id}",
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "request_id": f"demo-{custom_id}",
                    "body": {
                        "model": model,
                        "choices": [{"message": {"role": "assistant", "content": json.dumps(content, ensure_ascii=False)}}],
                    },
                },
                "error": None,
                "is_demo": True,
            }
        )
    return rows


def demo_content(experiment: str, sample: Dict[str, object], condition: str) -> Dict[str, object]:
    label = str(sample.get("gold_label", "neutral"))
    shifted = label
    if experiment == "perturbation" and condition == "text-only":
        if label in {"sadness", "fear"}:
            shifted = "neutral"
        elif label == "disgust":
            shifted = "anger"
    words = demo_words(shifted)
    if experiment == "closed_set":
        return {
            "predicted_label": shifted,
            "confidence": 0.82 if shifted == label else 0.58,
            "reason": DEMO_REASON.get(shifted, "The available evidence supports this label."),
        }
    return {
        "input_condition": condition,
        "free_emotions": words,
        "confidence": 0.84 if shifted == label else 0.57,
        "reason": DEMO_REASON.get(shifted, "The available evidence supports these emotion words."),
    }


def demo_words(label: str) -> List[str]:
    words = EMOTION_DICT.get(label, EMOTION_DICT["neutral"])
    return words[:2]


def decode_condition(value: str) -> str:
    return value.replace("_plus_", " + ").replace("text_only", "text-only").replace("_", " ")
