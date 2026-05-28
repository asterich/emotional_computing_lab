from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Paths
from .frames import extract_all_meld_video_frames
from .pipeline import clean_artifacts, fetch_pending_batch, run_pipeline, status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="emotional-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the full data, batch, parse, and export pipeline.")
    run.add_argument("--n", type=int, default=300, help="Number of samples to use.")
    run.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    run.add_argument("--demo", action="store_true", help="Allow demo samples when MELD raw data is absent.")
    run.add_argument("--no-fallback", action="store_true", help="Fail instead of writing demo fallback outputs.")
    run.add_argument("--no-api", action="store_true", help="Skip API submission and generate demo fallback outputs.")
    run.add_argument("--force", action="store_true", help="Remove generated artifacts before running.")
    run.add_argument("--poll-seconds", type=int, default=60, help="How long to poll an async batch before fallback.")

    subparsers.add_parser("status", help="Show last run and batch status.")

    fetch = subparsers.add_parser("fetch-batch", help="Poll and download the pending Qwen Batch output.")
    fetch.add_argument("--poll-seconds", type=int, default=300, help="How long to poll the existing batch.")

    extract_frames = subparsers.add_parser("extract-all-frames", help="Extract keyframes from every MELD video.")
    extract_frames.add_argument("--max-frames", type=int, default=3, help="Maximum keyframes per video.")
    extract_frames.add_argument("--force", action="store_true", help="Regenerate existing extracted frames.")
    extract_frames.add_argument("--limit", type=int, default=None, help="Optional debug limit.")

    clean = subparsers.add_parser("clean", help="Remove generated artifacts.")
    clean.add_argument("--artifacts-only", action="store_true", help="Required safety flag.")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path.cwd()

    if args.command == "run":
        summary = run_pipeline(
            root=root,
            n=args.n,
            seed=args.seed,
            demo=args.demo,
            allow_fallback=not args.no_fallback,
            use_api=not args.no_api,
            force=args.force,
            poll_seconds=args.poll_seconds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "status":
        print(json.dumps(status(root), ensure_ascii=False, indent=2))
        return 0

    if args.command == "fetch-batch":
        print(json.dumps(fetch_pending_batch(root, poll_seconds=args.poll_seconds), ensure_ascii=False, indent=2))
        return 0

    if args.command == "extract-all-frames":
        summary = extract_all_meld_video_frames(
            Paths.from_root(root),
            max_frames=args.max_frames,
            force=args.force,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "clean":
        if not args.artifacts_only:
            parser.error("clean requires --artifacts-only")
        clean_artifacts(root)
        print("Generated artifacts removed.")
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
