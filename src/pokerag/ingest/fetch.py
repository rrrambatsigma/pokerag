"""Fetch stage: revision-pinned wikitext plus rendered HTML.

Two requests per page are used on purpose:

* wikitext gives the exact revision identity (revid, timestamp, sha1) and is
  the artefact that proves the snapshot is stable;
* rendered HTML gives clean sections and tables without hand-parsing hundreds
  of Bulbapedia templates.

HTML is requested with ``oldid=<revid>`` so both artefacts always describe the
same revision, even if the wiki is edited mid-run.

Idempotency: the page register is an **upsert keyed by doc_id**, rewritten
atomically in ``doc_id`` order, and "already fetched" means "the stored file
still has the recorded hash" rather than "a file exists".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator

from pokerag.core.hashing import sha256_bytes
from pokerag.core.jsonl import atomic_write_text, read_jsonl, write_json, write_jsonl
from pokerag.core.logging import get_logger
from pokerag.ingest.config import CorpusConfig
from pokerag.ingest.models import HtmlRecord, PageRecord, SelectedPage
from pokerag.ingest.select import url_for
from pokerag.ingest.transport import MediaWikiTransport

logger = get_logger("pokerag.ingest.fetch")

REDIRECT_RE = re.compile(r"^\s*#(?:redirect|REDIRECT)\s*\[\[([^\]\|#]+)", re.IGNORECASE)


def file_sha256(path: Path) -> str | None:
    """Return the SHA-256 of a file, or ``None`` when it is missing."""
    if not path.exists():
        return None
    return sha256_bytes(path.read_bytes())


def _chunks(items: list[SelectedPage], size: int) -> Iterator[list[SelectedPage]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _resolve_title(requested: str, payload: dict) -> str:
    """Follow ``normalized`` and ``redirects`` entries to the final title."""
    query = payload.get("query", {})
    mapping: dict[str, str] = {}
    for entry in query.get("normalized", []) or []:
        mapping[entry["from"]] = entry["to"]
    for entry in query.get("redirects", []) or []:
        mapping[entry["from"]] = entry["to"]
    title = requested
    for _ in range(5):
        nxt = mapping.get(title)
        if not nxt or nxt == title:
            break
        title = nxt
    return title


def load_pages(path: Path) -> dict[str, dict]:
    """Load the page register, keeping the newest revision per doc_id.

    Unknown keys from older schemas are dropped so a schema change does not
    require re-fetching the corpus.
    """
    allowed = set(PageRecord.model_fields)
    register: dict[str, dict] = {}
    for row in read_jsonl(path):
        row = {key: value for key, value in row.items() if key in allowed}
        row.setdefault("raw_sha256", "")
        existing = register.get(row["doc_id"])
        if existing is None or int(row["revid"]) > int(existing["revid"]):
            register[row["doc_id"]] = row
    return register


def _write_pages(path: Path, rows: dict[str, dict]) -> None:
    write_jsonl(path, (rows[doc_id] for doc_id in sorted(rows)))


def fetch_wikitext(
    cfg: CorpusConfig,
    transport: MediaWikiTransport,
    selection: list[SelectedPage],
    limit: int | None = None,
    resume: bool = True,
) -> list[PageRecord]:
    """Fetch wikitext for every selected title and return the page register."""
    snap = cfg.snapshot_dir
    raw_dir = snap / "raw" / "wikitext"
    index_path = snap / "pages_index.jsonl"
    error_path = snap / "fetch_errors.jsonl"

    register = load_pages(index_path) if resume else {}
    # Backfill hashes for registers written by older versions: accept the file
    # once, then enforce equality on every later run.
    for row in register.values():
        if not row.get("raw_sha256"):
            actual = file_sha256(Path(row["wikitext_path"]))
            if actual:
                row["raw_sha256"] = actual
    known_titles = {row["requested_title"]: doc_id for doc_id, row in register.items()}
    pending: list[SelectedPage] = []
    for page in selection:
        doc_id = known_titles.get(page.title)
        if doc_id is None:
            pending.append(page)
            continue
        row = register[doc_id]
        if file_sha256(Path(row["wikitext_path"])) != row.get("raw_sha256"):
            pending.append(page)
    if limit is not None:
        pending = pending[:limit]
    logger.info("wikitext: %s pages pending (%s cached)", len(pending), len(register))

    for batch in _chunks(pending, cfg.fetch.batch_titles):
        payload = transport.get_json(
            {
                "action": "query",
                "titles": "|".join(page.title for page in batch),
                "prop": "revisions",
                "rvprop": "ids|timestamp|sha1|size|content",
                "rvslots": "main",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            }
        )
        pages_by_title = {page["title"]: page for page in payload.get("query", {}).get("pages", [])}
        for selected in batch:
            final_title = _resolve_title(selected.title, payload)
            page = pages_by_title.get(final_title)
            if page is None or page.get("missing"):
                logger.warning("missing page: %s", selected.title)
                continue
            revision = page["revisions"][0]
            content = revision["slots"]["main"]["content"]
            redirect_match = REDIRECT_RE.match(content)
            target = redirect_match.group(1).strip() if redirect_match else None

            raw = {
                "requested_title": selected.title,
                "title": page["title"],
                "pageid": page["pageid"],
                "revid": revision["revid"],
                "timestamp": revision["timestamp"],
                "sha1": revision.get("sha1"),
                "size": revision.get("size"),
                "content": content,
            }
            raw_path = raw_dir / f"{page['pageid']}_{revision['revid']}.json"
            write_json(raw_path, raw)
            record = PageRecord(
                doc_id=f"bulba:{page['pageid']:07d}",
                requested_title=selected.title,
                pageid=page["pageid"],
                revid=revision["revid"],
                title=page["title"],
                url=url_for(page["title"]),
                rev_timestamp=revision["timestamp"],
                api_sha1=revision.get("sha1") or "",
                api_size=int(revision.get("size") or len(content)),
                raw_sha256=file_sha256(raw_path) or "",
                wikitext_path=str(raw_path),
                is_redirect=target is not None,
                redirect_target=target,
            )
            register[record.doc_id] = record.model_dump()
        _write_pages(index_path, register)

    _write_pages(index_path, register)
    return [PageRecord.model_validate(register[doc_id]) for doc_id in sorted(register)]


def load_html_index(path: Path) -> dict[str, dict]:
    """Load the HTML index, keeping one row per doc_id."""
    allowed = set(HtmlRecord.model_fields)
    rows: dict[str, dict] = {}
    for row in read_jsonl(path):
        row = {key: value for key, value in row.items() if key in allowed}
        row.setdefault("html_sha256", "")
        rows[row["doc_id"]] = row
    return rows


def _write_html_index(path: Path, rows: dict[str, dict]) -> None:
    write_jsonl(path, (rows[doc_id] for doc_id in sorted(rows)))


def fetch_html(
    cfg: CorpusConfig,
    transport: MediaWikiTransport,
    records: Iterable[PageRecord],
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, str]:
    """Fetch rendered HTML pinned to each page's revision.

    Returns ``doc_id -> html_path`` for every page that has HTML on disk.
    """
    snap = cfg.snapshot_dir
    html_dir = snap / "raw" / "html"
    index_path = snap / "html_index.jsonl"

    index = load_html_index(index_path) if resume else {}
    for row in index.values():
        if not row.get("html_sha256"):
            actual = file_sha256(Path(row["html_path"]))
            if actual:
                row["html_sha256"] = actual
    pending = []
    for record in records:
        row = index.get(record.doc_id)
        if row and file_sha256(Path(row["html_path"])) == row.get("html_sha256"):
            continue
        pending.append(record)
    if limit is not None:
        pending = pending[:limit]
    logger.info("html: %s pages pending (%s cached)", len(pending), len(index))

    for position, record in enumerate(pending, start=1):
        payload = transport.get_json(
            {
                "action": "parse",
                "oldid": record.revid,
                "prop": "text|revid",
                "format": "json",
                "formatversion": "2",
            }
        )
        parsed = payload.get("parse")
        if not parsed or "text" not in parsed:
            logger.warning("no HTML for %s", record.doc_id)
            continue
        html_path = html_dir / f"{record.pageid}_{record.revid}.html"
        atomic_write_text(html_path, parsed["text"])
        entry = HtmlRecord(
            doc_id=record.doc_id,
            html_path=str(html_path),
            html_sha256=file_sha256(html_path) or "",
        )
        index[record.doc_id] = entry.model_dump()
        if position % 20 == 0:
            _write_html_index(index_path, index)
    _write_html_index(index_path, index)
    return {doc_id: row["html_path"] for doc_id, row in sorted(index.items())}
