"""Document metadata required by the citation payload.

``Hit`` carries the chunk text and offsets, but the UI also needs the page
title, National Dex number and types. Those live in the frozen
``documents.parquet`` infobox, which is a JSON object with a ``"0"`` list of
rendered infobox rows. The parser is deliberately tolerant: a missing or
malformed row yields an empty field rather than breaking a citation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pokerag.index.config import IndexConfig

TYPES = frozenset(
    {
        "normal",
        "fire",
        "water",
        "electric",
        "grass",
        "ice",
        "fighting",
        "poison",
        "ground",
        "flying",
        "psychic",
        "bug",
        "rock",
        "ghost",
        "dragon",
        "dark",
        "steel",
        "fairy",
    }
)

DEX_PATTERN = re.compile(r"#(\d{3,4})")
TITLE_SUFFIX = " (Pokémon)"


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    title: str
    dex: int | None
    types: tuple[str, ...]
    url: str


def _infobox_lines(raw: object) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, dict):
        return []
    lines = raw.get("0")
    if isinstance(lines, str):
        return [lines]
    if isinstance(lines, list):
        return [str(line) for line in lines]
    return []


def parse_infobox(raw: object) -> tuple[int | None, tuple[str, ...]]:
    """Return ``(dex, types)`` from one infobox JSON value."""
    lines = _infobox_lines(raw)

    dex: int | None = None
    for line in lines:
        match = DEX_PATTERN.search(line)
        if match:
            dex = int(match.group(1))
            break

    types: tuple[str, ...] = ()
    for line in lines:
        parts = [part.strip().lower() for part in line.split("|")]
        if not 1 <= len(parts) <= 2:
            continue
        known = tuple(part for part in parts if part in TYPES)
        # Single-type pages render as "Electric | Unknown"; keep the real type.
        if known and all(part in TYPES or part in {"unknown", ""} for part in parts):
            types = known
            break
    return dex, types


def display_title(title: str) -> str:
    return title.replace(TITLE_SUFFIX, "").strip()


def build_doc_meta(frame: pd.DataFrame) -> dict[str, DocMeta]:
    meta: dict[str, DocMeta] = {}
    for row in frame.itertuples(index=False):
        dex, types = parse_infobox(row.infobox_json)
        meta[str(row.doc_id)] = DocMeta(
            doc_id=str(row.doc_id),
            title=display_title(str(row.title)),
            dex=dex,
            types=types,
            url=str(row.url),
        )
    return meta


def load_doc_meta(cfg: IndexConfig) -> dict[str, DocMeta]:
    path: Path = cfg.snapshot_dir / "documents.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing documents.parquet: {path}")
    frame = pd.read_parquet(
        path, columns=["doc_id", "title", "url", "infobox_json"]
    )
    return build_doc_meta(frame)
