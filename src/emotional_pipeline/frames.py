from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Paths, relative_or_empty
from .csvio import write_csv


REPORT_FIELDS = ["sample_id", "video_path", "frame_paths", "frame_count", "status", "message"]
ALL_VIDEO_REPORT_FIELDS = ["split", "video_id", "video_path", "frame_paths", "frame_count", "status", "message"]


def extract_frames_if_possible(samples: List[Dict[str, object]], paths: Paths) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    report_rows: List[Dict[str, object]] = []
    try:
        import cv2  # type: ignore
    except Exception as exc:
        for sample in samples:
            report_rows.append(
                report_row(sample, "", [], "opencv_missing", f"OpenCV is not installed or cannot be imported: {exc}")
            )
        return samples, write_report(paths, report_rows)

    for sample in samples:
        video_value = str(sample.get("video_path", ""))
        if not video_value:
            report_rows.append(report_row(sample, "", [], "no_video_path", "Sample has no associated video path."))
            continue
        video_path = paths.root / video_value if not Path(video_value).is_absolute() else Path(video_value)
        if not video_path.exists():
            report_rows.append(report_row(sample, str(video_path), [], "video_missing", "Video path does not exist."))
            continue
        sample_dir = paths.frames / str(sample["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        extracted = extract_keyframes(cv2, video_path, sample_dir)
        sample["frame_paths"] = relative_or_empty(paths.root, extracted)
        status = "extracted" if extracted else "extract_failed"
        message = "Keyframes extracted." if extracted else "Video opened but no frames were written."
        report_rows.append(report_row(sample, str(video_path), extracted, status, message))
    return samples, write_report(paths, report_rows)


def extract_keyframes(cv2, video_path: Path, output_dir: Path, max_frames: int = 3) -> List[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return []
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
            frame = resize_frame(frame, max_dim=512)
            out = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            paths.append(out)
    capture.release()
    return paths


def extract_all_meld_video_frames(
    paths: Paths,
    output_dir: Optional[Path] = None,
    max_frames: int = 3,
    force: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    paths.ensure()
    output_root = output_dir or (paths.root / "frames" / "all_extracted_frames")
    output_root.mkdir(parents=True, exist_ok=True)
    videos = discover_meld_videos(paths.data_raw)
    if limit is not None:
        videos = videos[:limit]

    report_rows: List[Dict[str, object]] = []
    try:
        import cv2  # type: ignore
    except Exception as exc:
        for video_path in videos:
            split = infer_video_split(video_path)
            report_rows.append(
                all_video_report_row(
                    split,
                    video_path,
                    [],
                    "opencv_missing",
                    f"OpenCV is not installed or cannot be imported: {exc}",
                    paths.root,
                )
            )
        return write_all_video_report(paths, report_rows, len(videos), output_root)

    for video_path in videos:
        split = infer_video_split(video_path)
        video_output_dir = output_root / split / video_path.stem
        if force and video_output_dir.exists():
            shutil.rmtree(video_output_dir)
        video_output_dir.mkdir(parents=True, exist_ok=True)

        existing_frames = sorted(video_output_dir.glob("*.jpg"))
        if existing_frames and not force:
            report_rows.append(
                all_video_report_row(
                    split,
                    video_path,
                    existing_frames,
                    "cached",
                    "Existing keyframes reused.",
                    paths.root,
                )
            )
            continue

        extracted = extract_keyframes(cv2, video_path, video_output_dir, max_frames=max_frames)
        status = "extracted" if extracted else "extract_failed"
        message = "Keyframes extracted." if extracted else "Video opened but no frames were written."
        report_rows.append(all_video_report_row(split, video_path, extracted, status, message, paths.root))

    return write_all_video_report(paths, report_rows, len(videos), output_root)


def discover_meld_videos(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        return []
    known_dirs = ["train_splits", "dev_splits_complete", "output_repeated_splits_test", "test_splits_complete"]
    videos = set()
    for dirname in known_dirs:
        for split_root in raw_dir.rglob(dirname):
            if split_root.is_dir():
                videos.update(path for path in split_root.rglob("*.mp4") if is_real_video_file(path))
                videos.update(path for path in split_root.rglob("*.avi") if is_real_video_file(path))
    return sorted(videos, key=lambda path: str(path))


def is_real_video_file(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and path.stat().st_size > 1024


def infer_video_split(video_path: Path) -> str:
    parts = {part.lower() for part in video_path.parts}
    if "train_splits" in parts or "train_splits_complete" in parts:
        return "train"
    if "dev_splits_complete" in parts:
        return "dev"
    if "output_repeated_splits_test" in parts or "test_splits_complete" in parts:
        return "test"
    return "unknown"


def all_video_report_row(
    split: str,
    video_path: Path,
    frame_paths: List[Path],
    status: str,
    message: str,
    root: Path,
) -> Dict[str, object]:
    return {
        "split": split,
        "video_id": video_path.stem,
        "video_path": str(video_path),
        "frame_paths": relative_or_empty(root, frame_paths),
        "frame_count": len(frame_paths),
        "status": status,
        "message": message,
    }


def write_all_video_report(
    paths: Paths,
    rows: List[Dict[str, object]],
    video_count: int,
    output_root: Path,
) -> Dict[str, object]:
    csv_path = paths.logs / "all_frame_extraction_report.csv"
    json_path = paths.logs / "all_frame_extraction_summary.json"
    write_csv(csv_path, rows, ALL_VIDEO_REPORT_FIELDS)
    usable_statuses = {"extracted", "cached"}
    extracted_videos = sum(1 for row in rows if row["status"] in usable_statuses)
    summary = {
        "video_count": video_count,
        "processed_videos": len(rows),
        "extracted_videos": extracted_videos,
        "total_frames": sum(int(row["frame_count"]) for row in rows),
        "statuses": status_counts(rows),
        "output_dir": str(output_root.relative_to(paths.root)),
        "report_csv": str(csv_path.relative_to(paths.root)),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def resize_frame(frame, max_dim: int):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    import cv2  # type: ignore

    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def report_row(
    sample: Dict[str, object],
    video_path: str,
    frame_paths: List[Path],
    status: str,
    message: str,
) -> Dict[str, object]:
    return {
        "sample_id": sample.get("sample_id", ""),
        "video_path": video_path or sample.get("video_path", ""),
        "frame_paths": sample.get("frame_paths", "") or "|".join(str(path) for path in frame_paths),
        "frame_count": len(frame_paths),
        "status": status,
        "message": message,
    }


def write_report(paths: Paths, rows: List[Dict[str, object]]) -> Dict[str, object]:
    csv_path = paths.logs / "frame_extraction_report.csv"
    json_path = paths.logs / "frame_extraction_summary.json"
    write_csv(csv_path, rows, REPORT_FIELDS)
    extracted_samples = sum(1 for row in rows if row["status"] == "extracted")
    summary = {
        "sample_count": len(rows),
        "extracted_samples": extracted_samples,
        "total_frames": sum(int(row["frame_count"]) for row in rows),
        "statuses": status_counts(rows),
        "report_csv": str(csv_path.relative_to(paths.root)),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def status_counts(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts
