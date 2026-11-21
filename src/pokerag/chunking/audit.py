"""Chunking audit for ablation A1, measured without retrieval or embeddings.

The index report answers "how many chunks and how long"; this module answers
"how clean are the cuts", which is the part of A1 that must stand on its own:

* boundary integrity — does a chunk start and end at a sentence, at a heading,
  or in the middle of a word/table row;
* section fragmentation — how many chunks one section is cut into, i.e. how
  often the evidence for one fact is split;
* overlap redundancy — how many characters are duplicated by the configured
  overlap, because that is context the generator pays for twice;
* size distribution — per strategy, and per section kind;
* parent/child structure — the spans A7 will expand into.

Everything is derived from ``chunks/*.parquet`` plus ``documents.parquet``; no
chunk is rebuilt, no vector is touched, so the audit cannot change the data it
describes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pokerag.chunking.build import parse_sections
from pokerag.chunking.config import ChunkingConfig
from pokerag.chunking.split import merge_spans
from pokerag.core.hashing import sha256_json
from pokerag.core.jsonl import atomic_write_text, read_json, write_json
from pokerag.core.logging import get_logger

logger = get_logger("pokerag.chunking.audit")

SENTENCE_END = set(".!?:;\"')]")
TOKEN_BUCKETS = (
    ("<200", 0, 200),
    ("200-400", 200, 400),
    ("400-600", 400, 600),
    ("600-800", 600, 800),
    ("800-1024", 800, 1025),
    (">1024", 1025, None),
)


def _previous_nonspace(text: str, index: int) -> int:
    cursor = index - 1
    while cursor >= 0 and text[cursor] in " \n\t":
        cursor -= 1
    return cursor


def starts_at_sentence(text: str, start: int) -> bool:
    """True when the chunk begins at the start of a sentence (or of the text)."""
    cursor = _previous_nonspace(text, start)
    return cursor < 0 or text[cursor] in SENTENCE_END


def ends_at_sentence(text: str, end: int) -> bool:
    """True when the chunk ends after a sentence end (or at the end of the text)."""
    cursor = _previous_nonspace(text, end)
    return cursor < 0 or text[cursor] in SENTENCE_END


def splits_word(text: str, position: int) -> bool:
    """True when the boundary falls between two alphanumeric characters."""
    if position <= 0 or position >= len(text):
        return False
    return text[position - 1].isalnum() and text[position].isalnum()


def table_like_ratio(text: str) -> float:
    """Share of non-letter, non-space characters; high values mean table rows."""
    if not text:
        return 0.0
    noise = sum(1 for char in text if not char.isalpha() and not char.isspace())
    return noise / len(text)


def size_buckets(chunks: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, low, high in TOKEN_BUCKETS:
        values = chunks["n_tokens"]
        mask = values >= low
        if high is not None:
            mask &= values < high
        counts[label] = int(mask.sum())
    return counts


def section_fragmentation(
    documents: pd.DataFrame, chunks: pd.DataFrame
) -> dict[str, float | int]:
    """How many chunks each document section is split into.

    ``intact`` means a section is fully contained in exactly one chunk, which is
    the precondition for citing one chunk as evidence for one fact. ``split`` is
    the opposite: three or more chunks touch the same section.
    """
    by_doc: dict[str, list[tuple[int, int]]] = {}
    for row in chunks.itertuples():
        by_doc.setdefault(row.doc_id, []).append((int(row.char_start), int(row.char_end)))

    counts: list[int] = []
    missing = 0
    for row in documents.itertuples():
        sections = [
            section
            for section in parse_sections(row.sections_json)
            if int(section.get("char_end", 0)) > int(section.get("char_start", 0))
        ]
        spans = by_doc.get(row.doc_id, [])
        for section in sections:
            start, end = int(section["char_start"]), int(section["char_end"])
            touching = sum(1 for chunk_start, chunk_end in spans if min(end, chunk_end) > max(start, chunk_start))
            if touching == 0:
                missing += 1
            counts.append(touching)

    counts.sort()
    return {
        "sections": len(counts),
        "missing": missing,
        "p50": counts[len(counts) // 2] if counts else 0,
        "p95": counts[int(0.95 * (len(counts) - 1))] if counts else 0,
        "max": counts[-1] if counts else 0,
        "intact": round(sum(1 for value in counts if value == 1) / max(1, len(counts)), 4),
        "split_three_or_more": round(sum(1 for value in counts if value >= 3) / max(1, len(counts)), 4),
    }


def overlap_redundancy(documents: pd.DataFrame, chunks: pd.DataFrame) -> dict[str, float | int]:
    """Duplicated characters caused by overlap, relative to total chunk characters."""
    by_doc: dict[str, list[tuple[int, int]]] = {}
    total = 0
    for row in chunks.itertuples():
        start, end = int(row.char_start), int(row.char_end)
        by_doc.setdefault(row.doc_id, []).append((start, end))
        total += max(0, end - start)
    union_total = 0
    for doc_id, spans in by_doc.items():
        for start, end in merge_spans(spans):
            union_total += max(0, end - start)
    duplicated = total - union_total
    return {
        "chunk_chars": total,
        "unique_chars": union_total,
        "duplicated_chars": duplicated,
        "redundancy": round(duplicated / max(1, total), 4),
        "duplicated_tokens_estimate": int(round(duplicated / 4)),
    }


def boundary_integrity(documents: pd.DataFrame, chunks: pd.DataFrame) -> dict[str, float | int]:
    texts = {row.doc_id: row.text for row in documents.itertuples()}
    section_starts: dict[str, set[int]] = {}
    h2_starts: dict[str, list[int]] = {}
    for row in documents.itertuples():
        sections = parse_sections(row.sections_json)
        section_starts[row.doc_id] = {
            int(section["char_start"]) for section in sections if section.get("char_start") is not None
        }
        h2_starts[row.doc_id] = sorted(
            int(section["char_start"])
            for section in sections
            if section.get("level") == 2 and int(section.get("char_end", 0)) > int(section.get("char_start", 0))
        )

    clean_start = clean_end = at_heading = crosses_h2 = table_like = mid_word_start = mid_word_end = 0
    total = 0
    for row in chunks.itertuples():
        text = texts.get(row.doc_id)
        if text is None:
            continue
        start, end = int(row.char_start), int(row.char_end)
        total += 1
        at_heading_here = start in section_starts.get(row.doc_id, set())
        clean_start += starts_at_sentence(text, start) or at_heading_here
        clean_end += ends_at_sentence(text, end)
        at_heading += at_heading_here
        crosses_h2 += sum(1 for boundary in h2_starts.get(row.doc_id, []) if start < boundary < end)
        table_like += table_like_ratio(row.text) > 0.3
        mid_word_start += splits_word(text, start)
        mid_word_end += splits_word(text, end)
    return {
        "chunks": total,
        "clean_start": round(clean_start / max(1, total), 4),
        "clean_end": round(clean_end / max(1, total), 4),
        "starts_at_heading": round(at_heading / max(1, total), 4),
        "mid_word_start": round(mid_word_start / max(1, total), 4),
        "mid_word_end": round(mid_word_end / max(1, total), 4),
        "crosses_h2": crosses_h2,
        "table_like": round(table_like / max(1, total), 4),
    }


def parent_structure(chunks: pd.DataFrame) -> dict[str, float | int]:
    """Parent/child counts and the containment invariant A7 relies on."""
    parents = chunks[chunks["role"] == "parent"]
    children = chunks[chunks["role"] != "parent"]
    spans = {
        row.chunk_id: (int(row.char_start), int(row.char_end)) for row in parents.itertuples()
    }
    per_parent: dict[str, int] = {}
    violations = 0
    for row in children.itertuples():
        parent_id = row.parent_id
        if not isinstance(parent_id, str) or parent_id not in spans:
            continue
        per_parent[parent_id] = per_parent.get(parent_id, 0) + 1
        parent_start, parent_end = spans[parent_id]
        if not (parent_start <= int(row.char_start) and int(row.char_end) <= parent_end):
            violations += 1
    counts = sorted(per_parent.values())
    return {
        "parents": int(len(parents)),
        "children": int(len(children)),
        "children_per_parent_p50": counts[len(counts) // 2] if counts else 0,
        "children_per_parent_p95": counts[int(0.95 * (len(counts) - 1))] if counts else 0,
        "children_per_parent_max": max(counts) if counts else 0,
        "parents_without_child": int(len(parents) - len(per_parent)),
        "containment_violations": violations,
        "parent_tokens": int(parents["n_tokens"].sum()),
    }


def strategy_stats(
    strategy: str, documents: pd.DataFrame, chunks: pd.DataFrame, manifest: dict
) -> dict:
    units = chunks[chunks["role"] != "parent"]
    entry = manifest.get("strategies", {}).get(strategy, {})
    return {
        "strategy": strategy,
        "rows": int(len(chunks)),
        "units": int(len(units)),
        "parents": int(len(chunks) - len(units)),
        "chars_p50": int(units["n_chars"].median()),
        "chars_p95": int(units["n_chars"].quantile(0.95)),
        "chars_max": int(units["n_chars"].max()),
        "tokens_p50": int(units["n_tokens"].median()),
        "tokens_p95": int(units["n_tokens"].quantile(0.95)),
        "tokens_max": int(units["n_tokens"].max()),
        "tokens_total": int(units["n_tokens"].sum()),
        "dropped": int(entry.get("chunks_dropped", 0)),
        "below_min": int(entry.get("chunks_below_min", 0)),
        "coverage": entry.get("coverage_ratio"),
        "file_sha256": entry.get("file_sha256"),
        "size_buckets": size_buckets(units),
        "boundary": boundary_integrity(documents, units),
        "sections": section_fragmentation(documents, units),
        "overlap": overlap_redundancy(documents, units),
        "parents_detail": parent_structure(chunks) if strategy == "hierarchical" else None,
    }


def build_audit(cfg: ChunkingConfig) -> dict:
    """Return the audit for every strategy that has a chunk file."""
    snap = cfg.snapshot_dir
    documents = pd.read_parquet(snap / "documents.parquet", columns=["doc_id", "text", "sections_json"])
    manifest_path = snap / "chunks_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"strategies": {}}
    audit: dict[str, dict] = {}
    for strategy in ("fixed", "heading", "hierarchical"):
        path = cfg.chunks_dir / f"{strategy}.parquet"
        if not path.exists():
            continue
        chunks = pd.read_parquet(path)
        audit[strategy] = strategy_stats(strategy, documents, chunks, manifest)
    return {
        "snapshot": cfg.snapshot,
        "snapshot_id": manifest.get("snapshot_id"),
        "tokenizer": manifest.get("tokenizer", cfg.tokenizer),
        "budget_tokens": manifest.get("budget_tokens", cfg.budget_tokens),
        "min_chars_keep": cfg.min_chars_keep,
        "chunking_config_sha256": sha256_json(cfg.model_dump()),
        "strategies": audit,
    }


def _bucket_row(stats: dict) -> str:
    return " | ".join(str(stats["size_buckets"][label]) for label, _, _ in TOKEN_BUCKETS)


def render_markdown(audit: dict) -> str:
    names = list(audit["strategies"])
    lines = [
        "# A1 — audit chunking (tanpa retrieval)",
        "",
        "> Dibuat otomatis oleh `pokerag-chunk report`; tidak ada chunk yang dibangun",
        "> ulang dan tidak ada vektor yang disentuh. Angka berasal dari",
        "> `chunks/<strategi>.parquet` dan `documents.parquet` pada snapshot ini.",
        "",
        f"- snapshot: `{audit['snapshot']}` — `{str(audit['snapshot_id'])[:16]}`",
        f"- tokenizer: `{audit['tokenizer']}`, budget {audit['budget_tokens']} token,"
        f" `min_chars_keep` {audit['min_chars_keep']} karakter",
        "",
        "## Ukuran dan biaya",
        "",
        "| Strategi | Unit | Parent | char p50 / p95 / max | token p50 / p95 / max | total token | dibuang | < min | cakupan |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = audit["strategies"][name]
        coverage = f"{s['coverage']:.1%}" if isinstance(s["coverage"], (int, float)) else "-"
        lines.append(
            f"| `{name}` | {s['units']} | {s['parents']} | {s['chars_p50']} / {s['chars_p95']} / {s['chars_max']} | "
            f"{s['tokens_p50']} / {s['tokens_p95']} / {s['tokens_max']} | {s['tokens_total']:,} | "
            f"{s['dropped']} | {s['below_min']} | {coverage} |"
        )

    lines += [
        "",
        "## Integritas batas potongan",
        "",
        "| Strategi | mulai bersih | berhenti bersih | mulai di heading | mulai di tengah kata | berhenti di tengah kata | memotong batas h2 | mirip tabel |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        b = audit["strategies"][name]["boundary"]
        lines.append(
            f"| `{name}` | {b['clean_start']:.1%} | {b['clean_end']:.1%} | "
            f"{b['starts_at_heading']:.1%} | {b['mid_word_start']:.1%} | {b['mid_word_end']:.1%} | "
            f"{b['crosses_h2']} | {b['table_like']:.1%} |"
        )

    lines += [
        "",
        "## Fragmentasi section",
        "",
        "| Strategi | section | utuh dalam 1 chunk | pecah ≥3 chunk | p50 | p95 | maks | tidak tersentuh |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = audit["strategies"][name]["sections"]
        lines.append(
            f"| `{name}` | {s['sections']} | {s['intact']:.1%} | {s['split_three_or_more']:.1%} | "
            f"{s['p50']} | {s['p95']} | {s['max']} | {s['missing']} |"
        )

    lines += [
        "",
        "## Redundansi overlap",
        "",
        "| Strategi | karakter chunk | karakter unik | duplikat | redundansi | token duplikat (≈) |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        o = audit["strategies"][name]["overlap"]
        lines.append(
            f"| `{name}` | {o['chunk_chars']:,} | {o['unique_chars']:,} | {o['duplicated_chars']:,} | "
            f"{o['redundancy']:.1%} | {o['duplicated_tokens_estimate']:,} |"
        )

    lines += [
        "",
        "## Distribusi ukuran (token, unit saja)",
        "",
        "| Strategi | " + " | ".join(label for label, _, _ in TOKEN_BUCKETS) + " |",
        "|---|" + "---|" * len(TOKEN_BUCKETS),
    ]
    for name in names:
        lines.append(f"| `{name}` | {_bucket_row(audit['strategies'][name])} |")

    hierarchical = audit["strategies"].get("hierarchical", {}).get("parents_detail")
    if hierarchical:
        lines += [
            "",
            "## Struktur parent/child (`hierarchical`)",
            "",
            f"- parent: {hierarchical['parents']:,} (tidak di-embed), child: {hierarchical['children']:,}",
            f"- child per parent: p50 {hierarchical['children_per_parent_p50']}, "
            f"p95 {hierarchical['children_per_parent_p95']}, maks {hierarchical['children_per_parent_max']}",
            f"- parent tanpa child: {hierarchical['parents_without_child']:,}",
            f"- token di parent: {hierarchical['parent_tokens']:,} (tidak masuk index)",
            f"- pelanggaran invariant child ⊂ parent: **{hierarchical['containment_violations']}**",
        ]

    lines += [
        "",
        "## Definisi",
        "",
        "| Metrik | Definisi |",
        "|---|---|",
        "| Unit | chunk yang di-embed (`role != parent`) |",
        "| Parent | baris `hierarchical` untuk context expansion, tidak di-embed |",
        "| mulai bersih | chunk mulai di `char_start` sebuah section, atau karakter non-spasi sebelumnya adalah `.` `!` `?` `:` `;` atau tanda kutip/penutup |",
        "| berhenti bersih | karakter non-spasi sebelum `char_end` adalah akhir kalimat, atau chunk berakhir di akhir dokumen |",
        "| mulai tepat di heading | `char_start` sama dengan `char_start` sebuah section |",
        "| tengah kata | karakter sebelum dan sesudah batas keduanya huruf/angka |",
        "| memotong batas h2 | ada section level 2 yang mulai di dalam rentang chunk |",
        "| mirip tabel | lebih dari 30% karakter chunk bukan huruf/spasi |",
        "| fragmentasi section | jumlah chunk yang beririsan dengan satu section |",
        "| utuh dalam 1 chunk | section hanya disentuh satu chunk — syarat satu chunk bisa dikutip untuk satu fakta |",
        "| redundansi overlap | (jumlah karakter chunk − gabungan span) / jumlah karakter chunk |",
        "| token duplikat | perkiraan dari redundansi karakter (÷4), bukan hitungan tokenizer |",
        "| cakupan | proporsi teks dokumen yang tercakup minimal satu chunk |",
        "",
        "## Ancaman validitas",
        "",
        "* Deteksi batas kalimat berbasis karakter, bukan parser kalimat; singkatan",
        "  seperti \"vs.\" dapat dihitung sebagai akhir kalimat.",
        "* `mirip tabel` adalah proksi; tabel dengan banyak kata tetap terhitung",
        "  sebagai prosa.",
        "* Token duplikat adalah perkiraan; laporan retrieval nanti memakai token",
        "  nyata dari konteks top-k.",
        "* Section kecil (mis. heading kosong) tetap dihitung; jumlah section tidak",
        "  sama dengan jumlah fakta.",
        "",
        "## Reproduksi",
        "",
        "```bash",
        "pokerag-chunk build",
        "pokerag-chunk report",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_report(cfg: ChunkingConfig, docs_dir: Path | None = None) -> dict:
    """Write the JSON artefact and the committed Markdown report."""
    audit = build_audit(cfg)
    write_json(cfg.snapshot_dir / "chunks_audit.json", audit)
    target = (docs_dir or Path("docs/experiments")) / "A1-chunking.md"
    atomic_write_text(target, render_markdown(audit))
    logger.info("audit chunking: %s", target)
    return audit
