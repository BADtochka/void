from __future__ import annotations

import argparse
import fnmatch
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from faster_whisper.utils import _MODELS
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

DOWNLOAD_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
)


def resolve_model(model_name_or_path: str) -> str:
    """Return a local model path, downloading with visible progress if necessary."""
    local_path = Path(model_name_or_path).expanduser()
    if local_path.exists():
        return str(local_path)

    repo_id = _MODELS.get(model_name_or_path)
    if repo_id is None:
        if "/" not in model_name_or_path:
            choices = ", ".join(_MODELS)
            raise ValueError(f"Unknown Whisper model {model_name_or_path!r}. Available: {choices}")
        repo_id = model_name_or_path

    logger.info("Preparing Whisper model %s from %s", model_name_or_path, repo_id)
    model_path = snapshot_download(
        repo_id,
        allow_patterns=list(DOWNLOAD_PATTERNS),
        tqdm_class=tqdm,
    )
    logger.info("Whisper model is ready: %s", model_path)
    return model_path


def _download_size(model_info: Any) -> int:
    total = 0
    for file in model_info.siblings:
        if any(fnmatch.fnmatch(file.rfilename, pattern) for pattern in DOWNLOAD_PATTERNS):
            total += file.size or 0
    return total


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    raise AssertionError("unreachable")


def available_models() -> list[tuple[str, str, int | None]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    for name, repo_id in _MODELS.items():
        aliases[repo_id].append(name)

    api = HfApi()
    sizes: dict[str, int | None] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(api.model_info, repo_id, files_metadata=True): repo_id
            for repo_id in aliases
        }
        for future in as_completed(futures):
            repo_id = futures[future]
            try:
                sizes[repo_id] = _download_size(future.result())
            except Exception as error:
                logger.warning("Could not get size for %s: %s", repo_id, error)
                sizes[repo_id] = None

    return [
        (", ".join(names), repo_id, sizes[repo_id])
        for repo_id, names in aliases.items()
    ]


def print_available_models() -> None:
    rows = available_models()
    name_width = max(len("MODEL"), *(len(name) for name, _, _ in rows))
    size_width = max(len("DOWNLOAD"), 10)
    print(f"{'MODEL':<{name_width}}  {'DOWNLOAD':>{size_width}}  REPOSITORY")
    print(f"{'-' * name_width}  {'-' * size_width}  {'-' * 42}")
    for names, repo_id, size in rows:
        display_size = _human_size(size) if size is not None else "unknown"
        print(f"{names:<{name_width}}  {display_size:>{size_width}}  {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List or download faster-whisper models")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list available models and their download sizes")
    download_parser = subparsers.add_parser("download", help="download a model with progress")
    download_parser.add_argument("model", help="model name, Hugging Face repository, or local path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "list":
        print_available_models()
    else:
        print(resolve_model(args.model))


if __name__ == "__main__":
    main()
