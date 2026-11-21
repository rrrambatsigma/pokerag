"""Cleaning stage: rendered HTML -> spans of clean text.

The cleaner is deliberately conservative about *content* and aggressive about
*chrome*: navigation boxes, references, edit links, images and maintenance
banners are removed; everything a reader would call "the article" is kept,
including tables (converted to ``a | b`` rows) and the infobox (converted to
``key: value`` lines).

Every emitted section is a character span of the final text. That invariant is
what later keeps qrels valid across chunking strategies.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag

from pokerag.core.hashing import sha256_json, sha256_text
from pokerag.core.jsonl import read_jsonl, write_jsonl
from pokerag.core.logging import get_logger
from pokerag.ingest.config import CorpusConfig
from pokerag.ingest.fetch import load_html_index, load_pages
from pokerag.ingest.models import CleanDoc, LicenseInfo, Section

logger = get_logger("pokerag.ingest.clean")

# Bump this whenever the cleaning rules change. ``run_clean`` skips a document
# whose stored version and config hash still match, so an edited cleaner with an
# unchanged version would silently keep stale documents.
CLEANER_VERSION = "1"

REDIRECT_RE = re.compile(r"^\s*#(?:redirect|REDIRECT)\s*\[\[([^\]\|#]+)", re.IGNORECASE)
DISAMBIG_RE = re.compile(r"\{\{\s*(disambig|dab|disambiguation)\b", re.IGNORECASE)
HEADING_TAGS = {"h2", "h3", "h4", "h5"}

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?\)\]%])")
_SPACE_AFTER_OPEN = re.compile(r"([\(\[])\s+")
_MULTI_LINE = re.compile(r"\n{3,}")

# Lines that only exist in Bulbapedia's prev/next navigation and carry no prose.
JUNK_LINE = re.compile(
    r"^(?:Pok[eé]mon\s*\|.*|Pok[eé]mon\s*|#\d{4}:.*|[-<]*>+|\|\s*[-<]*>+)$"
)


def strip_junk_lines(text: str) -> str:
    """Remove residual navigation lines that survive table filtering."""
    return "\n".join(line for line in text.split("\n") if not JUNK_LINE.match(line.strip()))


def parse_wikitext_flags(wikitext: str) -> tuple[bool, str | None, bool]:
    """Return ``(is_redirect, redirect_target, is_disambiguation)``."""
    match = REDIRECT_RE.match(wikitext)
    return (
        match is not None,
        match.group(1).strip() if match else None,
        bool(DISAMBIG_RE.search(wikitext)),
    )


def normalize_text(text: str) -> str:
    """Collapse whitespace and repair spacing introduced by HTML extraction."""
    text = (
        text.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u2192", "->")
        .replace("\u2190", "<-")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _MULTI_LINE.sub("\n\n", text)
    return text.strip()


def element_text(element: Tag) -> str:
    """Extract readable text from a block element."""
    name = element.name
    if name in ("ul", "ol"):
        items = [normalize_text(li.get_text(" ", strip=True)) for li in element.find_all("li")]
        items = [item for item in items if item]
        return " | ".join(items)
    if name == "dl":
        parts = []
        for child in element.find_all(["dt", "dd"]):
            value = normalize_text(child.get_text(" ", strip=True))
            if value:
                parts.append(f"{value}" if child.name == "dt" else f"- {value}")
        return "\n".join(parts)
    return normalize_text(element.get_text(" ", strip=True))


def table_to_markdown(table: Tag) -> str:
    """Render a table as pipe-separated rows (nested tables flattened)."""
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            normalize_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


NOISE_PATTERNS = (
    r"For Pok[eé]mon GO information",
    r"This article is about",
    r"Project Pok[eé]dex",
    r"^Pok[eé]mon\s*\|",
    r"Images on the Bulbagarden Archives",
    r"See also:",
)


def is_link_farm(element: Tag) -> bool:
    """True for navigation blocks: many links, almost no prose.

    Bulbapedia's bottom navigation is a chain of class-less ``div``s, so link
    density is the only reliable signal. Article prose averages far more than
    12 characters per link; a link farm averages fewer than 8.
    """
    links = element.find_all("a")
    if len(links) < 8:
        return False
    text = normalize_text(element.get_text(" ", strip=True))
    return len(text) / len(links) < 12


def is_garbage_table(table: Tag) -> bool:
    """Detect navigation/pokédex-prev-next tables that carry no article content."""
    classes = table.get("class") or []
    if "navbox" in classes or "infobox" in classes or "toc" in classes:
        return True
    if is_link_farm(table):
        return True
    text = normalize_text(table.get_text(" ", strip=True))
    if len(text) < 25:
        return True
    if any(re.search(pattern, text) for pattern in NOISE_PATTERNS):
        return True
    if "->" in text and "<-" in text and len(text) < 400:
        return True
    if "•" in text and len(text) < 600:
        return True
    return False


def parse_infobox(soup: BeautifulSoup) -> list[str]:
    """Flatten the species infobox into compact ``cell | cell`` fact lines.

    Bulbapedia renders most infobox labels as images, so labels are often not
    recoverable as text. Values (types, abilities, height, weight, egg groups)
    are, and they are what the retriever needs.
    """
    table = soup.select_one("table.infobox")
    if table is None:
        return []
    lines: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            normalize_text(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"], recursive=False)
        ]
        cells = [cell for cell in cells if cell]
        if len(cells) < 2:
            continue
        if "[[" in " ".join(cells) or set(cells) == {"Unknown"}:
            continue
        line = " | ".join(cells)
        if "{{{" in line or line.count("Unknown") > 2 or len(line) > 300:
            continue
        if line not in lines:
            lines.append(line)
        if len(lines) >= 20:
            break
    return lines


def _sections_from_html(
    soup: BeautifulSoup, drop_selectors: list[str], drop_titles: set[str]
) -> tuple[list[tuple[list[str], int, str, str]], dict[str, int]]:
    """Walk the article body and split it into heading-scoped text blocks."""
    root = soup.select_one(".mw-parser-output") or soup.select_one("#mw-content-text") or soup.body
    if root is None:
        return [], {}

    dropped: dict[str, int] = {}
    for selector in drop_selectors:
        elements = root.select(selector)
        dropped[selector] = len(elements)
        for element in elements:
            element.decompose()

    blocks: list[tuple[list[str], int, str, str]] = []
    heading: list[str] = []
    level = 0
    buffer: list[str] = []
    has_text = False

    def flush() -> None:
        nonlocal buffer, has_text
        if not buffer:
            return
        body = normalize_text(strip_junk_lines("\n".join(buffer)))
        buffer = []
        if len(body) < 25:
            has_text = False
            return
        if len(body) < 700 and any(re.search(pattern, body) for pattern in NOISE_PATTERNS):
            has_text = False
            return
        path_lower = " ".join(heading).lower()
        if level == 0:
            kind = "lead"
        elif "trivia" in path_lower:
            kind = "trivia"
        elif not has_text:
            kind = "table"
        else:
            kind = "section"
        blocks.append((list(heading), level, body, kind))
        has_text = False

    for child in root.children:
        if not isinstance(child, Tag):
            continue
        if child.name in HEADING_TAGS:
            flush()
            title = normalize_text(child.get_text(" ", strip=True))
            new_level = int(child.name[1])
            heading = heading[: new_level - 2] + [title]
            level = new_level
            if title in drop_titles:
                heading = []
                level = 0
            continue
        if child.name == "table":
            if is_garbage_table(child):
                continue
            markdown = table_to_markdown(child)
            if markdown:
                buffer.append(markdown)
            continue
        if child.name == "div" and is_link_farm(child):
            dropped["link_farm"] = dropped.get("link_farm", 0) + 1
            continue
        text = element_text(child)
        if text:
            buffer.append(text)
            has_text = True
    flush()
    return blocks, dropped


def clean_html(
    html: str,
    *,
    doc_id: str,
    pageid: int,
    revid: int,
    title: str,
    url: str,
    license: LicenseInfo,
    cfg: CorpusConfig,
    is_redirect: bool = False,
    redirect_target: str | None = None,
    is_disambiguation: bool = False,
) -> CleanDoc:
    """Convert one rendered page into a :class:`CleanDoc` with exact spans."""
    soup = BeautifulSoup(html, "lxml")
    infobox = parse_infobox(soup)
    blocks, dropped = _sections_from_html(
        soup,
        cfg.cleaning.drop_selectors,
        set(cfg.cleaning.drop_section_titles),
    )

    parts: list[str] = []
    sections: list[Section] = []
    position = 0

    def append_block(path: list[str], block_level: int, body: str, kind: str) -> None:
        nonlocal position
        if parts:
            parts.append("\n\n")
            position += 2
        start = position
        parts.append(body)
        position += len(body)
        sections.append(
            Section(
                heading_path=path,
                level=block_level,
                kind=kind,  # type: ignore[arg-type]
                char_start=start,
                char_end=position,
            )
        )

    if infobox and not is_redirect:
        body = "\n".join(infobox)
        append_block(["Infobox"], 1, body, "infobox")

    for path, block_level, body, kind in blocks:
        append_block(path, block_level, body, kind)

    text = "".join(parts)
    n_chars = len(text)
    return CleanDoc(
        doc_id=doc_id,
        pageid=pageid,
        revid=revid,
        title=title,
        url=url,
        text=text,
        content_sha256=sha256_text(text),
        n_chars=n_chars,
        n_sections=len(sections),
        sections=sections,
        infobox=infobox,
        is_redirect=is_redirect,
        redirect_target=redirect_target,
        is_disambiguation=is_disambiguation,
        is_stub=n_chars < cfg.cleaning.min_chars,
        dropped={key: value for key, value in dropped.items() if value},
        license=license,
        cleaner_version=CLEANER_VERSION,
        clean_config_sha=sha256_json(cfg.cleaning.model_dump()),
    )


def run_clean(cfg: CorpusConfig, limit: int | None = None, rebuild: bool = False) -> int:
    """Clean every fetched page, upserting into ``cleaned.jsonl``.

    A document is re-cleaned only when its revision changed, its stored hash is
    missing, or the cleaner version/config changed. Writing is atomic and sorted
    by ``doc_id``, so an unchanged corpus produces an identical file.
    """
    snap = cfg.snapshot_dir
    register = load_pages(snap / "pages_index.jsonl")
    html_index = load_html_index(snap / "html_index.jsonl")
    out_path = snap / "cleaned.jsonl"

    existing = {row["doc_id"]: row for row in read_jsonl(out_path)} if not rebuild else {}
    config_sha = sha256_json(cfg.cleaning.model_dump())

    pending = []
    for doc_id in sorted(register):
        page = register[doc_id]
        if doc_id not in html_index:
            continue
        row = existing.get(doc_id)
        up_to_date = (
            row is not None
            and row.get("cleaner_version") == CLEANER_VERSION
            and row.get("clean_config_sha") == config_sha
            and row.get("revid") == page["revid"]
        )
        if not up_to_date:
            pending.append(page)
    if limit is not None:
        pending = pending[:limit]
    logger.info("clean: %s pages pending (%s up to date)", len(pending), len(register) - len(pending))

    written = 0
    for page in pending:
        wikitext = Path(page["wikitext_path"]).read_text(encoding="utf-8")
        _, _, is_disambiguation = parse_wikitext_flags(wikitext)
        html = Path(html_index[page["doc_id"]]["html_path"]).read_text(encoding="utf-8")
        document = clean_html(
            html,
            doc_id=page["doc_id"],
            pageid=page["pageid"],
            revid=page["revid"],
            title=page["title"],
            url=page["url"],
            license=cfg.license_for(page["revid"]),
            cfg=cfg,
            is_redirect=page.get("is_redirect", False),
            redirect_target=page.get("redirect_target"),
            is_disambiguation=is_disambiguation,
        )
        existing[document.doc_id] = document.model_dump()
        written += 1
        if written % 25 == 0:
            write_jsonl(out_path, (existing[doc_id] for doc_id in sorted(existing)))
            logger.info("cleaned %s/%s", written, len(pending))
    write_jsonl(out_path, (existing[doc_id] for doc_id in sorted(existing)))
    return written
