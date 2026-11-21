"""Run manifests: what produced a number, and how to reproduce it.

The brief asks for three independent runs per configuration plus a manifest
that records code, prompt, model, index, seed, sampling, tokens and latency.
This module collects that record and writes it next to the per-request trace.
It is deliberately library-only: the runner decides what to call.

Missing values stay ``None``. A run that did not record tokens must not look
like a run that recorded zero tokens.
"""

from __future__ import annotations

import importlib.metadata
import math
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pokerag.core.hashing import sha256_file
from pokerag.core.jsonl import write_json, write_jsonl

PACKAGES = (
    "pokerag",
    "torch",
    "transformers",
    "sentence-transformers",
    "qdrant-client",
    "bm25s",
    "numpy",
    "pandas",
)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def _git(args: list[str], cwd: Path | None) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    """Return ``(commit_sha, dirty)``; both ``None`` outside a git checkout."""
    head = _git(["rev-parse", "HEAD"], cwd)
    if head is None or head.returncode != 0:
        return None, None
    status = _git(["status", "--porcelain"], cwd)
    dirty = bool(status.stdout.strip()) if status is not None else None
    return head.stdout.strip(), dirty


def file_digest(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    return sha256_file(candidate) if candidate.is_file() else None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str | None = None
    dtype: str | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
        }


@dataclass(frozen=True)
class SamplingSpec:
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RequestRecord:
    query_id: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    abstained: bool = False
    cited_chunk_ids: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "latency_ms": round(self.latency_ms, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "abstained": self.abstained,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "error": self.error,
        }


def new_run_id(run_set: str, run_index: int) -> str:
    if run_index < 0:
        raise ValueError("run_index must be >= 0")
    return f"{run_set}-r{run_index:02d}"


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    run_set: str
    run_index: int
    created_at: str
    snapshot: str
    index_version: str
    strategy: str
    mode: str
    code_sha: str | None = None
    code_dirty: bool | None = None
    config_path: str | None = None
    config_sha256: str | None = None
    prompt_path: str | None = None
    prompt_sha256: str | None = None
    qrels_path: str | None = None
    qrels_sha256: str | None = None
    queries_path: str | None = None
    queries_sha256: str | None = None
    model: ModelSpec | None = None
    sampling: SamplingSpec | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_set": self.run_set,
            "run_index": self.run_index,
            "created_at": self.created_at,
            "code_sha": self.code_sha,
            "code_dirty": self.code_dirty,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "prompt_path": self.prompt_path,
            "prompt_sha256": self.prompt_sha256,
            "qrels_path": self.qrels_path,
            "qrels_sha256": self.qrels_sha256,
            "queries_path": self.queries_path,
            "queries_sha256": self.queries_sha256,
            "snapshot": self.snapshot,
            "index_version": self.index_version,
            "strategy": self.strategy,
            "mode": self.mode,
            "model": self.model.to_dict() if self.model else None,
            "sampling": self.sampling.to_dict() if self.sampling else None,
            "environment": self.environment,
            "extra": self.extra,
        }


def build_manifest(
    *,
    run_set: str,
    run_index: int,
    snapshot: str,
    index_version: str,
    strategy: str,
    mode: str,
    config_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    qrels_path: str | Path | None = None,
    queries_path: str | Path | None = None,
    model: ModelSpec | None = None,
    sampling: SamplingSpec | None = None,
    extra: dict[str, Any] | None = None,
    cwd: Path | None = None,
) -> RunManifest:
    code_sha, code_dirty = git_state(cwd)
    return RunManifest(
        run_id=new_run_id(run_set, run_index),
        run_set=run_set,
        run_index=run_index,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        snapshot=snapshot,
        index_version=index_version,
        strategy=strategy,
        mode=mode,
        code_sha=code_sha,
        code_dirty=code_dirty,
        config_path=str(config_path) if config_path else None,
        config_sha256=file_digest(config_path),
        prompt_path=str(prompt_path) if prompt_path else None,
        prompt_sha256=file_digest(prompt_path),
        qrels_path=str(qrels_path) if qrels_path else None,
        qrels_sha256=file_digest(qrels_path),
        queries_path=str(queries_path) if queries_path else None,
        queries_sha256=file_digest(queries_path),
        model=model,
        sampling=sampling,
        environment=environment(),
        extra=extra or {},
    )


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def summarise(records: list[RequestRecord]) -> dict[str, Any]:
    """Aggregate latency, tokens and abstention; errors are counted, not hidden."""
    ok = [record for record in records if record.error is None]
    latencies = [record.latency_ms for record in ok]
    inputs = [record.input_tokens for record in records if record.input_tokens is not None]
    outputs = [record.output_tokens for record in records if record.output_tokens is not None]

    return {
        "requests": len(records),
        "errors": len(records) - len(ok),
        "abstained": sum(1 for record in records if record.abstained),
        "abstain_rate": (
            sum(1 for record in records if record.abstained) / len(records)
            if records
            else None
        ),
        "latency_ms": {
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
        "tokens": {
            "input_total": sum(inputs) if inputs else None,
            "output_total": sum(outputs) if outputs else None,
            "with_input": len(inputs),
            "with_output": len(outputs),
        },
    }


def write_run(
    root: str | Path,
    manifest: RunManifest,
    records: list[RequestRecord],
) -> Path:
    """Write ``manifest.json`` and ``requests.jsonl`` under ``root/run_id``."""
    run_dir = Path(root) / manifest.run_id
    payload = manifest.to_dict()
    payload["summary"] = summarise(records)
    write_json(run_dir / "manifest.json", payload)
    write_jsonl(run_dir / "requests.jsonl", [record.to_dict() for record in records])
    return run_dir
