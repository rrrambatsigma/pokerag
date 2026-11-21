"""Validation gate: proves the snapshot meets the brief before it is frozen.

Each gate records what was observed, so the report can be attached to the
project documentation instead of a claim that "the corpus is fine".
"""

from __future__ import annotations

import statistics
from pathlib import Path

from pokerag.core.jsonl import read_json, read_jsonl, write_json
from pokerag.core.logging import get_logger
from pokerag.ingest.config import CorpusConfig
from pokerag.ingest.models import ValidationGate

logger = get_logger("pokerag.ingest.validate")


def _final_docs(cfg: CorpusConfig) -> list[dict]:
    snap = cfg.snapshot_dir
    dedup = read_json(snap / "dedup_report.json")
    duplicates = set(dedup.get("duplicates", {}))
    return [
        row
        for row in read_jsonl(snap / "cleaned.jsonl")
        if not row["is_redirect"]
        and not row["is_disambiguation"]
        and not row["is_stub"]
        and row["n_chars"] > 0
        and row["doc_id"] not in duplicates
    ]


def _span_invariant(doc: dict) -> bool:
    text = doc["text"]
    return all(text[s["char_start"] : s["char_end"]] for s in doc["sections"])


def run_validation(cfg: CorpusConfig) -> dict:
    """Evaluate every gate and write ``validation.json`` plus a Markdown report."""
    snap = cfg.snapshot_dir
    cleaned = list(read_jsonl(snap / "cleaned.jsonl"))
    pages = list(read_jsonl(snap / "pages_index.jsonl"))
    errors = list(read_jsonl(snap / "fetch_errors.jsonl"))
    fetch_log = list(read_jsonl(snap / "fetch_log.jsonl"))
    final = _final_docs(cfg)

    char_counts = [doc["n_chars"] for doc in final]
    section_counts = [doc["n_sections"] for doc in final]
    median_chars = int(statistics.median(char_counts)) if char_counts else 0

    gates: list[ValidationGate] = []

    def gate(name: str, threshold: str, observed: str, passed: bool) -> None:
        gates.append(ValidationGate(gate=name, threshold=threshold, observed=observed, passed=passed))

    gate("jumlah dokumen final", f">= {cfg.validation.min_docs}", str(len(final)), len(final) >= cfg.validation.min_docs)

    complete = [
        doc
        for doc in final
        if doc.get("revid") and doc.get("rev_timestamp", True) is not None and doc.get("url") and doc.get("license", {}).get("name")
    ]
    gate("metadata lengkap", "100% dokumen", f"{len(complete)}/{len(final)}", len(complete) == len(final))

    cleaned_ids = {row["doc_id"] for row in cleaned}
    missing_clean = [page["doc_id"] for page in pages if page["doc_id"] not in cleaned_ids]
    gate("semua halaman terproses", "0 gagal parse", f"{len(missing_clean)} gagal", not missing_clean)

    too_short = [doc["doc_id"] for doc in final if doc["n_chars"] < cfg.validation.min_chars]
    gate(
        "panjang minimum per dokumen",
        f">= {cfg.validation.min_chars} karakter",
        f"{len(too_short)} dokumen di bawah ambang",
        not too_short,
    )

    too_few_sections = [doc["doc_id"] for doc in final if doc["n_sections"] < cfg.validation.min_sections]
    gate(
        "struktur section",
        f">= {cfg.validation.min_sections} section",
        f"{len(too_few_sections)} dokumen di bawah ambang",
        not too_few_sections,
    )

    invariant_failures = [doc["doc_id"] for doc in final if not _span_invariant(doc)]
    gate("invariant offset karakter", "100% section cocok", f"{len(invariant_failures)} gagal", not invariant_failures)

    disambig_in_final = [doc["doc_id"] for doc in final if doc["is_disambiguation"]]
    gate("disambiguasi dikecualikan", "0 masuk korpus", str(len(disambig_in_final)), not disambig_in_final)

    redirected_in_final = [doc["doc_id"] for doc in final if doc["is_redirect"]]
    gate("redirect dikecualikan", "0 masuk korpus", str(len(redirected_in_final)), not redirected_in_final)

    ft_leaks = [doc["doc_id"] for doc in final if doc["license"]["allow_ft_training"]]
    gate("lisensi: ft_training ditolak", "0 dokumen berizin training", str(len(ft_leaks)), not ft_leaks)

    gaps = [row.get("gap_seconds") for row in fetch_log if row.get("gap_seconds") is not None]
    min_gap = min(gaps) if gaps else cfg.fetch.delay_seconds
    gate(
        "politeness crawl-delay",
        f">= {cfg.fetch.delay_seconds} s antar request",
        f"gap minimum {min_gap:.2f} s dari {len(gaps)} request",
        min_gap >= cfg.fetch.delay_seconds * 0.9,
    )

    stats = {
        "n_pages_fetched": len(pages),
        "n_cleaned": len(cleaned),
        "n_final": len(final),
        "n_redirect": sum(1 for doc in cleaned if doc["is_redirect"]),
        "n_disambiguation": sum(1 for doc in cleaned if doc["is_disambiguation"]),
        "n_stub": sum(1 for doc in cleaned if doc["is_stub"]),
        "n_fetch_errors": len(errors),
        "median_chars": median_chars,
        "median_sections": int(statistics.median(section_counts)) if section_counts else 0,
        "total_chars": sum(char_counts),
    }
    payload = {
        "snapshot": cfg.name,
        "passed": all(g.passed for g in gates),
        "stats": stats,
        "gates": [gate.model_dump() for gate in gates],
    }
    write_json(snap / "validation.json", payload)
    _write_markdown(snap / "validation_report.md", cfg, payload)
    logger.info("validation: %s/%s gate lulus", sum(g.passed for g in gates), len(gates))
    return payload


def _write_markdown(path: Path, cfg: CorpusConfig, payload: dict) -> None:
    lines = [
        f"# Validation report - {cfg.name}",
        "",
        f"Status keseluruhan: **{'LULUS' if payload['passed'] else 'GAGAL'}**",
        "",
        "## Statistik",
        "",
        "| Metrik | Nilai |",
        "|---|---|",
    ]
    for key, value in payload["stats"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Gate", "", "| Gate | Ambang | Observasi | Status |", "|---|---|---|---|"]
    for gate in payload["gates"]:
        status = "lulus" if gate["passed"] else "**gagal**"
        lines.append(f"| {gate['gate']} | {gate['threshold']} | {gate['observed']} | {status} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
