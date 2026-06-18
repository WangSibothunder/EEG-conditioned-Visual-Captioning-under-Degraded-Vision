from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset snapshot.")
    parser.add_argument("--repo", required=True, help="Dataset repo id, e.g. owner/name.")
    parser.add_argument("--dest", required=True, help="Local destination directory.")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.dest,
    )
    print(path)


if __name__ == "__main__":
    main()
