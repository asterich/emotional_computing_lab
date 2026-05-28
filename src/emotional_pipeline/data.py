from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import LABELS, Paths
from .csvio import read_csv, write_csv


SAMPLE_FIELDS = [
    "sample_id",
    "dialogue_id",
    "utterance_id",
    "speaker",
    "utterance",
    "context",
    "gold_label",
    "video_path",
    "frame_paths",
    "source_split",
    "is_demo",
]


DEMO_ROWS = [
    ("anger", "Ross", "I can't believe you did that again.", "Rachel: I thought it was fine."),
    ("disgust", "Monica", "That is absolutely gross.", "Chandler: I found it behind the fridge."),
    ("sadness", "Rachel", "I really thought this was going to work out.", "Monica: Are you okay?"),
    ("joy", "Chandler", "This is the best news I've heard all week.", "Joey: Wait until you hear the rest."),
    ("neutral", "Phoebe", "The meeting starts at seven.", "Ross: Do we need to bring anything?"),
    ("surprise", "Joey", "Wait, you got the job?", "Rachel: They called me this morning."),
    ("fear", "Monica", "I'm worried we might miss the deadline.", "Chandler: We still have tonight."),
    ("anger", "Rachel", "You promised you would not tell anyone.", "Ross: I only told one person."),
    ("disgust", "Chandler", "I do not want to touch that thing.", "Joey: It moved a little."),
    ("sadness", "Ross", "I miss how things used to be.", "Phoebe: That sounds hard."),
    ("joy", "Phoebe", "I am so happy for you.", "Monica: It finally happened."),
    ("neutral", "Ross", "I left the keys on the table.", "Rachel: Thanks."),
    ("surprise", "Monica", "Oh my god, I did not see that coming.", "Chandler: Nobody did."),
    ("fear", "Joey", "I am scared this audition will go badly.", "Rachel: You practiced a lot."),
]


def load_or_create_samples(paths: Paths, n: int, seed: int, allow_demo: bool) -> List[Dict[str, object]]:
    raw_csvs = find_meld_csvs(paths.data_raw)
    if raw_csvs:
        rows = sample_meld(raw_csvs, paths, n=n, seed=seed)
    elif allow_demo:
        rows = create_demo_samples(n=n)
    else:
        raise FileNotFoundError(
            "No MELD CSV found under data/raw. Put MELD CSV files there or rerun with --demo."
        )
    write_csv(paths.data_processed / "sampled_data.csv", rows, SAMPLE_FIELDS)
    write_csv(paths.final_for_b / "sampled_data.csv", rows, SAMPLE_FIELDS)
    return rows


def find_meld_csvs(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        return []
    candidates = sorted(raw_dir.rglob("*.csv"))
    scored = []
    for path in candidates:
        name = path.name.lower()
        score = 0
        if "sent_emo" in name or "meld" in name:
            score += 5
        if "dev" in name or "test" in name:
            score += 2
        if "train" in name:
            score += 1
        if "sent_emo" in name or {"dialogue_id", "utterance_id", "emotion"}.issubset(csv_header_keys(path)):
            scored.append((score, path))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for score, path in scored if score >= 2] or [scored[0][1]]


def csv_header_keys(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            line = handle.readline()
    except OSError:
        return set()
    return {part.strip().lower().replace(" ", "_") for part in line.split(",")}


def sample_meld(csv_paths: List[Path], paths: Paths, n: int, seed: int) -> List[Dict[str, object]]:
    raw_rows = []
    for path in csv_paths:
        raw_rows.extend(normalize_meld_row(row, paths, path) for row in read_csv(path))
    normalized = raw_rows
    normalized = [row for row in normalized if row["gold_label"] in LABELS and row["utterance"]]
    by_label: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in normalized:
        by_label[str(row["gold_label"])].append(row)

    rng = random.Random(seed)
    for rows in by_label.values():
        rng.shuffle(rows)

    selected: List[Dict[str, object]] = []
    per_label = max(1, n // len(LABELS))
    for label in LABELS:
        selected.extend(by_label.get(label, [])[:per_label])

    remaining = [row for row in normalized if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, n - len(selected))])
    selected = selected[:n]
    selected.sort(key=lambda row: (str(row["source_split"]), str(row["dialogue_id"]), int_or_zero(row["utterance_id"])))

    context_by_dialogue: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in normalized:
        context_by_dialogue[dialogue_key(row)].append(row)
    for rows in context_by_dialogue.values():
        rows.sort(key=lambda row: int_or_zero(row["utterance_id"]))

    selected_ids = {id(row) for row in selected}
    final_rows = []
    index = 1
    for row in selected:
        dialogue_rows = context_by_dialogue[dialogue_key(row)]
        row["context"] = build_context(dialogue_rows, int_or_zero(row["utterance_id"]))
        row["sample_id"] = f"S{index:04d}"
        final_rows.append(row)
        selected_ids.discard(id(row))
        index += 1
    return final_rows


def normalize_meld_row(row: Dict[str, str], paths: Paths, source_path: Path) -> Dict[str, object]:
    lookup = {key.lower().replace(" ", "_"): value for key, value in row.items()}
    dialogue_id = pick(lookup, ["dialogue_id", "dialogueid", "dialogue"])
    utterance_id = pick(lookup, ["utterance_id", "utteranceid", "utt_id"])
    speaker = pick(lookup, ["speaker"])
    utterance = pick(lookup, ["utterance", "text", "sentence"])
    label = pick(lookup, ["emotion", "label", "sentiment"]).lower().strip()
    split = infer_split(source_path)
    video_path = find_video_path(paths.data_raw, split, dialogue_id, utterance_id)
    return {
        "sample_id": "",
        "dialogue_id": dialogue_id,
        "utterance_id": utterance_id,
        "speaker": speaker,
        "utterance": utterance,
        "context": "",
        "gold_label": label,
        "video_path": str(video_path) if video_path else "",
        "frame_paths": "",
        "source_split": split,
        "is_demo": "false",
    }


def dialogue_key(row: Dict[str, object]) -> Tuple[str, str]:
    return str(row.get("source_split", "")), str(row.get("dialogue_id", ""))


def pick(row: Dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        if row.get(name) is not None:
            return str(row[name]).strip()
    return ""


def infer_split(path: Path) -> str:
    lower = str(path).lower()
    if "train" in lower:
        return "train"
    if "dev" in lower:
        return "dev"
    if "test" in lower:
        return "test"
    return "unknown"


def find_video_path(raw_dir: Path, split: str, dialogue_id: str, utterance_id: str) -> Optional[Path]:
    if not dialogue_id or not utterance_id:
        return None
    names = [
        f"dia{dialogue_id}_utt{utterance_id}.mp4",
        f"dia{dialogue_id}_utt{utterance_id}.avi",
    ]
    split_roots = {
        "dev": ["dev_splits_complete"],
        "test": ["output_repeated_splits_test", "test_splits_complete"],
        "train": ["train_splits", "train_splits_complete"],
    }
    search_roots = []
    for dirname in split_roots.get(split, []):
        search_roots.extend(raw_dir.rglob(dirname))
    search_roots.append(raw_dir)
    for root in search_roots:
        for name in names:
            matches = list(root.rglob(name)) if root.exists() else []
            if matches:
                return matches[0]
    return None


def build_context(dialogue_rows: List[Dict[str, object]], utterance_id: int, window: int = 3) -> str:
    previous = [row for row in dialogue_rows if int_or_zero(row["utterance_id"]) < utterance_id]
    previous = previous[-window:]
    return " ".join(f'{row["speaker"]}: {row["utterance"]}' for row in previous)


def create_demo_samples(n: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index in range(n):
        label, speaker, utterance, context = DEMO_ROWS[index % len(DEMO_ROWS)]
        rows.append(
            {
                "sample_id": f"S{index + 1:04d}",
                "dialogue_id": f"demo_{index // 7 + 1}",
                "utterance_id": str(index % 7),
                "speaker": speaker,
                "utterance": utterance,
                "context": context,
                "gold_label": label,
                "video_path": "",
                "frame_paths": "",
                "source_split": "demo",
                "is_demo": "true",
            }
        )
    return rows


def int_or_zero(value: object) -> int:
    try:
        return int(str(value))
    except ValueError:
        return 0
