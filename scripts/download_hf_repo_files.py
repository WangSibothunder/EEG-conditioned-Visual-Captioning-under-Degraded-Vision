from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def list_files(repo_id: str, repo_type: str, include_prefixes: list[str]) -> list[tuple[str, int]]:
    api = HfApi()
    files: list[tuple[str, int]] = []
    for info in api.list_repo_tree(repo_id, repo_type=repo_type, recursive=True, expand=True):
        path = getattr(info, "path", "")
        size = getattr(info, "size", None)
        if not path or size is None:
            continue
        if include_prefixes and not any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in include_prefixes):
            continue
        files.append((path, int(size)))
    return sorted(files)


def retry_sleep(attempt: int, base_seconds: float) -> None:
    time.sleep(min(base_seconds * attempt, 300))


def list_files_with_retries(
    repo_id: str,
    repo_type: str,
    include_prefixes: list[str],
    retries: int,
    retry_sleep_seconds: float,
) -> list[tuple[str, int]]:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return list_files(repo_id, repo_type, include_prefixes)
        except Exception as exc:  # Network/API errors are common on large HF repos.
            last_exc = exc
            print(
                json.dumps(
                    {
                        "event": "list_error",
                        "attempt": attempt,
                        "retries": retries,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            if attempt < retries:
                retry_sleep(attempt, retry_sleep_seconds)
    assert last_exc is not None
    raise last_exc


def download_one_with_retries(
    *,
    repo_id: str,
    repo_type: str,
    path: str,
    local_dir: Path,
    cache_dir: str | None,
    retries: int,
    retry_sleep_seconds: float,
) -> Path:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=path,
                    local_dir=str(local_dir),
                    cache_dir=cache_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            )
        except Exception as exc:
            last_exc = exc
            print(
                json.dumps(
                    {
                        "event": "download_retry",
                        "path": path,
                        "attempt": attempt,
                        "retries": retries,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            if attempt < retries:
                retry_sleep(attempt, retry_sleep_seconds)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Hugging Face repo files one by one with resumable status logs.")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--status-jsonl", required=True)
    parser.add_argument("--include-prefix", action="append", default=[])
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--list-retries", type=int, default=5)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=20.0)
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_jsonl)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    files = list_files_with_retries(
        args.repo,
        args.repo_type,
        args.include_prefix,
        args.list_retries,
        args.retry_sleep_seconds,
    )
    if args.max_files is not None:
        files = files[: args.max_files]

    print(json.dumps({"event": "file_list", "count": len(files), "bytes": sum(size for _, size in files)}, ensure_ascii=True), flush=True)

    for index, (path, expected_size) in enumerate(files, start=1):
        out_path = local_dir / path
        if out_path.exists() and out_path.stat().st_size == expected_size:
            record = {
                "event": "skip_existing",
                "index": index,
                "count": len(files),
                "path": path,
                "expected_size": expected_size,
                "actual_size": out_path.stat().st_size,
            }
            print(json.dumps(record, ensure_ascii=True), flush=True)
            status_path.open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=True) + "\n")
            continue

        start = time.time()
        try:
            downloaded = download_one_with_retries(
                repo_id=args.repo,
                repo_type=args.repo_type,
                path=path,
                local_dir=local_dir,
                cache_dir=args.cache_dir,
                retries=args.download_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            actual_size = downloaded.stat().st_size
            ok = actual_size == expected_size
            record = {
                "event": "downloaded" if ok else "size_mismatch",
                "index": index,
                "count": len(files),
                "path": path,
                "expected_size": expected_size,
                "actual_size": actual_size,
                "seconds": round(time.time() - start, 3),
            }
        except Exception as exc:
            record = {
                "event": "error",
                "index": index,
                "count": len(files),
                "path": path,
                "expected_size": expected_size,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "seconds": round(time.time() - start, 3),
            }
        print(json.dumps(record, ensure_ascii=True), flush=True)
        status_path.open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=True) + "\n")
        if record["event"] == "error":
            raise SystemExit(1)
        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
