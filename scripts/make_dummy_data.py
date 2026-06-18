from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collate import eeg_vision_collate
from src.data.dataset import EEGVisionCaptionDataset
from src.data.dummy_data import create_dummy_data


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_simple_yaml(Path(path))
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, config)]

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", maxsplit=1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if ":" not in stripped:
                raise ValueError(f"{path}:{line_number} expected 'key: value' syntax")
            key, value = stripped.split(":", maxsplit=1)
            key = key.strip()
            value = value.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value:
                parent[key] = _parse_scalar(value)
            else:
                nested: dict[str, Any] = {}
                parent[key] = nested
                stack.append((indent, nested))

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic dummy EEG/image data.")
    parser.add_argument("--config", default="configs/debug.yaml", help="YAML config path.")
    parser.add_argument("--num-train", type=int, default=None, help="Number of train rows.")
    parser.add_argument("--num-val", type=int, default=None, help="Number of val rows.")
    parser.add_argument("--smoke-test", action="store_true", help="Load one dataloader batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    data_config = config.get("data", {})

    root = Path(data_config.get("root", "data"))
    num_train = args.num_train if args.num_train is not None else 8
    num_val = args.num_val if args.num_val is not None else 2
    seed = int(config.get("seed", 0))
    image_size = int(data_config.get("image_size", 224))
    eeg_channels = int(data_config.get("eeg_channels", 64))
    eeg_timesteps = int(data_config.get("eeg_timesteps", 250))

    paths = create_dummy_data(
        root=root,
        num_train=num_train,
        num_val=num_val,
        seed=seed,
        image_size=image_size,
        eeg_channels=eeg_channels,
        eeg_timesteps=eeg_timesteps,
    )

    print(f"wrote {num_train} train rows to {paths['train_manifest']}")
    print(f"wrote {num_val} val rows to {paths['val_manifest']}")
    print(f"wrote images to {paths['image_dir']}")
    print(f"wrote eeg arrays to {paths['eeg_dir']}")

    if args.smoke_test:
        dataset = EEGVisionCaptionDataset(
            paths["train_manifest"],
            image_size=image_size,
            eeg_shape=(eeg_channels, eeg_timesteps),
        )
        loader = DataLoader(dataset, batch_size=min(2, len(dataset)), collate_fn=eeg_vision_collate)
        batch = next(iter(loader))
        print(
            "smoke batch: "
            f"image={tuple(batch['image'].shape)} "
            f"eeg={tuple(batch['eeg'].shape)} "
            f"labels={tuple(batch['label'].shape)}"
        )


if __name__ == "__main__":
    main()
