from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from .config import LABELS


EMOTION_DICT = {
    "anger": ["angry", "mad", "annoyed", "irritated", "frustrated", "furious", "impatient", "resentful", "outraged"],
    "disgust": ["disgusted", "repulsed", "grossed out", "disturbed", "offended", "gross"],
    "sadness": ["sad", "upset", "disappointed", "hurt", "lonely", "regretful", "depressed", "heartbroken", "miserable"],
    "joy": ["happy", "glad", "excited", "amused", "pleased", "relieved", "grateful", "cheerful", "delighted"],
    "neutral": ["neutral", "calm", "indifferent", "serious", "casual", "factual", "matter-of-fact", "plain"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "startled", "unexpected"],
    "fear": ["afraid", "scared", "nervous", "anxious", "worried", "terrified", "insecure", "panicked"],
}

AMBIGUOUS_WORDS = {"confused", "awkward", "uncomfortable", "sarcastic", "hesitant", "serious"}

WORD_TO_LABEL = {
    word: label
    for label, words in EMOTION_DICT.items()
    for word in words
}


def normalize_emotions(values: object) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_items = [values]
    else:
        raw_items = [str(item) for item in values]

    normalized: List[str] = []
    for item in raw_items:
        text = item.lower().strip()
        text = re.sub(r"\b(very|slightly|somewhat|feeling|a bit|kind of|rather)\b", "", text)
        text = text.replace("/", ",").replace(" and ", ",")
        for piece in text.split(","):
            word = re.sub(r"[^a-z\-\s]", "", piece).strip()
            word = re.sub(r"\s+", " ", word)
            if word and word not in normalized:
                normalized.append(word)
    return normalized[:5]


def dictionary_map(free_emotions: List[str], utterance: str = "", context: str = "") -> Tuple[str, str, str, float]:
    labels = []
    unresolved = []
    for word in free_emotions:
        if word in AMBIGUOUS_WORDS:
            unresolved.append(word)
        elif word in WORD_TO_LABEL:
            labels.append(WORD_TO_LABEL[word])
        else:
            fallback = lexical_fallback(word, utterance, context)
            if fallback:
                labels.append(fallback)
            else:
                unresolved.append(word)

    if labels:
        counts = Counter(labels)
        top = counts.most_common()
        if len(top) == 1 or top[0][1] > top[1][1]:
            method = "dictionary" if not unresolved else "hybrid_dictionary_plus_fallback"
            return top[0][0], method, "Dictionary majority vote over normalized emotion words.", min(0.95, 0.65 + top[0][1] * 0.1)

    if unresolved:
        inferred = infer_from_text(utterance, context)
        return inferred, "heuristic_llm_fallback", f"Unresolved words: {', '.join(unresolved)}; used text-level fallback.", 0.55

    return "neutral", "default_neutral", "No mappable emotion words found.", 0.4


def lexical_fallback(word: str, utterance: str, context: str) -> str:
    compact = word.replace("-", " ")
    if any(token in compact for token in ["worry", "anx", "scare", "panic"]):
        return "fear"
    if any(token in compact for token in ["annoy", "anger", "mad", "irritat"]):
        return "anger"
    if any(token in compact for token in ["sad", "disappoint", "hurt"]):
        return "sadness"
    if any(token in compact for token in ["happy", "joy", "glad", "amus", "excit"]):
        return "joy"
    if any(token in compact for token in ["shock", "surpris", "startl"]):
        return "surprise"
    if any(token in compact for token in ["disgust", "gross", "repuls"]):
        return "disgust"
    return ""


def infer_from_text(utterance: str, context: str) -> str:
    text = f"{utterance} {context}".lower()
    keyword_labels = [
        ("fear", ["worried", "scared", "afraid", "deadline", "miss"]),
        ("anger", ["can't believe", "promised", "again", "angry"]),
        ("disgust", ["gross", "touch that", "disgust"]),
        ("sadness", ["miss", "thought this was going to work", "sad"]),
        ("joy", ["best news", "happy", "great", "job"]),
        ("surprise", ["wait", "oh my god", "coming", "unexpected"]),
    ]
    for label, keywords in keyword_labels:
        if any(keyword in text for keyword in keywords):
            return label
    return "neutral"


def is_allowed_label(label: str) -> bool:
    return label in LABELS
