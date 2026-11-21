"""Download artifacts from a Hugging Face dataset and verify their hashes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from pokerag.core.integrity import sha256_file

HF_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"

DEFAULT_FILES = ("INTEGRITY.sha256", "LICENSE")
DEFAULT_PREFIXES = ("manifests/", "chunks/", "reports/")
DEFAULT_EXTRA = ("indexes.tar.gz", "documents.parquet")


@dataclass(frozen=True, slots=True)
class DownloadResult:
    path: Path
    sha256: str
    repo_commit: str | None
    skipped: bool


def resolve_url(dataset: str, path: str, revision: str = "main") -> str:
    return HF_RESOLVE.format(dataset=dataset, revision=revision, path=path)


def select_paths(
    paths: Iterable[str],
    *,
    include_all: bool = False,
    files: tuple[str, ...] = DEFAULT_FILES,
    prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
    extra: tuple[str, ...] = DEFAULT_EXTRA,
) -> list[str]:
    """Choose which dataset files belong to the default artifact set."""

    all_paths = set(paths)
    if include_all:
        return sorted(all_paths)

    selected = {path for path in files if path in all_paths}
    selected.update(path for path in extra if path in all_paths)
    for path in all_paths:
        if path.startswith(prefixes):
            selected.add(path)
    return sorted(selected)


def download_file(
    *,
    dataset: str,
    remote_path: str,
    dest: Path,
    revision: str = "main",
    expected_sha256: str | None = None,
    force: bool = False,
    progress: Progress | None = None,
    timeout: float = 120.0,
) -> DownloadResult:
    """Download one dataset file, verifying the digest before moving it in place.

    An existing destination is reused when its digest matches. Downloads land in
    ``<dest>.part`` first so an interrupted transfer never looks complete.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not force:
        digest = sha256_file(dest)
        if expected_sha256 is None or digest == expected_sha256:
            return DownloadResult(dest, digest, None, skipped=True)
        dest.unlink()

    url = resolve_url(dataset, remote_path, revision)
    part = dest.with_name(dest.name + ".part")
    task = None

    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": "pokerag-m0/0.1"},
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0")) or None
            repo_commit = response.headers.get("x-repo-commit")
            if progress is not None:
                task = progress.add_task(remote_path, total=total)
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    if progress is not None and task is not None:
                        progress.update(task, advance=len(chunk))
            if progress is not None and task is not None:
                progress.remove_task(task)

    digest = sha256_file(part)
    if expected_sha256 is not None and digest != expected_sha256:
        part.unlink(missing_ok=True)
        raise ValueError(
            f"digest mismatch for {remote_path}: expected {expected_sha256}, got {digest}"
        )

    part.replace(dest)
    return DownloadResult(dest, digest, repo_commit, skipped=False)


def default_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )
