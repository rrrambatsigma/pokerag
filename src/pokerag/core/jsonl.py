"""Atomic JSON/JSONL helpers.

Idempotency requirement: running a stage twice must leave byte-identical
artefacts. Two consequences are implemented here:

* every write goes to ``<name>.tmp`` and is then moved with ``os.replace``, so a
  crash never leaves a half-written file;
* JSON is serialised with sorted keys and JSONL is written line by line in the
  caller-provided (stable) order, so the bytes depend only on the data.

Append-only audit logs (``fetch_log.jsonl``) are the one exception and use
:func:`append_jsonl` on purpose: they record what happened, not what is true.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent(path: Path) -> Path:
    """Create the parent directory of ``path`` if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary file and an atomic replace."""
    ensure_parent(path)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as stable, pretty UTF-8 JSON."""
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def dump_jsonl(rows: Iterable[Any]) -> tuple[str, int]:
    """Serialise rows to JSONL text, preserving the given order."""
    lines: list[str] = []
    count = 0
    for row in rows:
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        count += 1
    return ("\n".join(lines) + "\n") if lines else "", count


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """Atomically overwrite ``path`` with one JSON object per line.

    Returns the number of rows written. Order is the caller's responsibility;
    stage loaders sort by a stable key (``doc_id``) so re-runs are byte-identical.
    """
    text, count = dump_jsonl(rows)
    atomic_write_text(path, text)
    return count


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one JSON object to an audit log (not part of the content hashes)."""
    ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield objects from a JSONL file, skipping blank lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
