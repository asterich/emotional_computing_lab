from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .config import Paths, relative_or_empty


def extract_frames_if_possible(samples: List[Dict[str, object]], paths: Paths) -> List[Dict[str, object]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return samples

    for sample in samples:
        video_value = str(sample.get("video_path", ""))
        if not video_value:
            continue
        video_path = paths.root / video_value if not Path(video_value).is_absolute() else Path(video_value)
        if not video_path.exists():
            continue
        sample_dir = paths.frames / str(sample["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        extracted = extract_keyframes(cv2, video_path, sample_dir)
        sample["frame_paths"] = relative_or_empty(paths.root, extracted)
    return samples


def extract_keyframes(cv2, video_path: Path, output_dir: Path, max_frames: int = 3) -> List[Path]:
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        capture.release()
        return []
    indices = sorted(set([max(0, int(total * ratio)) for ratio in [0.25, 0.5, 0.75]][:max_frames]))
    paths: List[Path] = []
    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = capture.read()
        if ok:
            out = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            paths.append(out)
    capture.release()
    return paths
