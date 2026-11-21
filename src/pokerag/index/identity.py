"""Embedding-model identity, measured rather than assumed.

The thesis requires the embedding model to be identified the way a dataset is:
name and tag, weight digest, architecture, parameter count, quantisation,
dimension, context length, file format and licence. Those facts are collected
from three places:

* the configuration (what we asked for, e.g. ``max_seq_length`` and revision);
* the loaded model object (what actually ran, e.g. parameters, dtype, pooling);
* the Hugging Face API (what the bytes are: commit sha and per-file SHA-256).

Results go to ``artifacts/models/embedder_identity.json`` (hashed into the index
manifest) and to a committed Markdown table under ``docs/models/`` so a reader
of the repository can check it without the artefacts.
"""

from __future__ import annotations

from pathlib import Path

from pokerag.core.hashing import sha256_json
from pokerag.core.jsonl import atomic_write_text, read_json, write_json
from pokerag.core.logging import get_logger
from pokerag.index.config import IndexConfig

logger = get_logger("pokerag.index.identity")

WEIGHT_SUFFIXES = (".safetensors", ".bin")


def hub_metadata(model: str, revision: str | None = None) -> dict:
    """Commit sha, licence and per-file digests, straight from the Hub."""
    try:
        from huggingface_hub import model_info

        info = model_info(model, revision=revision, files_metadata=True)
    except Exception as exc:  # network down should not block an offline build
        logger.warning("metadata Hugging Face tidak tersedia: %s", exc)
        return {"repo_id": model, "error": f"{type(exc).__name__}: {exc}"[:300]}

    files = []
    for sibling in info.siblings:
        entry = {"name": sibling.rfilename, "size": sibling.size}
        digest = (sibling.lfs or {}).get("sha256")
        entry["sha256"] = digest
        entry["blob_id"] = sibling.blob_id if not digest else None
        files.append(entry)
    card = info.cardData or {}
    return {
        "repo_id": model,
        "repo_sha": info.sha,
        "last_modified": info.lastModified.isoformat() if info.lastModified else None,
        "license": card.get("license") or card.get("license_name"),
        "files": files,
    }


def _weights(hub: dict, parameters: int | None) -> list[dict]:
    weights = [f for f in hub.get("files", []) if f["name"].endswith(WEIGHT_SUFFIXES)]
    for weight in weights:
        ratio = None
        if parameters and weight.get("size"):
            ratio = round(weight["size"] / parameters, 2)
        weight["bytes_per_parameter"] = ratio
    return sorted(weights, key=lambda item: item["name"])


def preferred_weight(weights: list[dict]) -> dict | None:
    """sentence-transformers loads safetensors first, then the pytorch bin."""
    for weight in weights:
        if weight["name"].endswith(".safetensors"):
            return weight
    return weights[0] if weights else None


def build_identity(
    cfg: IndexConfig,
    runtime: dict,
    truncation: dict,
    architecture: str | None = None,
) -> dict:
    """Assemble the identity record from config, runtime facts and the Hub."""
    if cfg.embedder.provider in ("st", "local"):
        hub = hub_metadata(cfg.embedder.model, cfg.embedder.revision)
    else:  # hosted or fake: there is no repository to hash
        hub = {"repo_id": cfg.embedder.model, "files": [], "license": None, "repo_sha": None}
    parameters = runtime.get("parameters")
    weights = _weights(hub, parameters)
    weight = preferred_weight(weights)
    if weight and weight.get("bytes_per_parameter"):
        ratio = weight["bytes_per_parameter"]
        stored_dtype = {4.0: "float32", 2.0: "float16", 1.0: "int8"}.get(ratio, f"{ratio} byte/parameter")
    else:
        stored_dtype = "unknown"

    return {
        "model": {
            "id": cfg.embedder.model,
            "tag": "main",
            "revision_requested": cfg.embedder.revision,
            "revision_resolved": hub.get("repo_sha"),
            "hub_error": hub.get("error"),
        },
        "digest": {
            "file": weight["name"] if weight else None,
            "sha256": weight.get("sha256") if weight else None,
            "size_bytes": weight.get("size") if weight else None,
        },
        "weights": weights,
        "architecture": architecture or runtime.get("architecture"),
        "parameters": parameters,
        "dimension": runtime.get("dimension", cfg.embedder.dimension),
        "quantization": {
            "stored": stored_dtype,
            "inference": runtime.get("dtype_inference", cfg.embedder.dtype),
        },
        "context": {
            "model_max_tokens": runtime.get("model_max_seq_length"),
            "used_max_tokens": cfg.embedder.max_seq_length,
            "tokenizer": runtime.get("tokenizer", cfg.embedder.model),
        },
        "pooling": runtime.get("pooling"),
        "normalize": runtime.get("normalize", True),
        "license": hub.get("license"),
        "runtime": runtime,
        "truncation": truncation,
    }


def identity_sha256(identity: dict) -> str:
    return sha256_json(identity)


def save_identity(cfg: IndexConfig, identity: dict, docs_dir: Path = Path("docs/models")) -> dict:
    """Write the JSON artefact and the committed Markdown table."""
    write_json(cfg.identity_path, identity)
    slug = cfg.embedder.model.replace("/", "-")
    markdown_path = docs_dir / f"embedding-{slug}.md"
    atomic_write_text(markdown_path, render_markdown(identity, cfg))
    logger.info("identitas embedder: %s (%s)", cfg.identity_path, markdown_path)
    return identity


def load_identity(cfg: IndexConfig) -> dict | None:
    return read_json(cfg.identity_path) if cfg.identity_path.exists() else None


def _row(label: str, value) -> str:
    text = "tidak tersedia" if value in (None, "") else str(value)
    return f"| {label} | {text} |"


def render_markdown(identity: dict, cfg: IndexConfig) -> str:
    """Human-readable table; the JSON next to it is the machine-readable source."""
    model = identity["model"]
    digest = identity["digest"]
    quantization = identity["quantization"]
    context = identity["context"]
    runtime = identity["runtime"]
    truncation = identity["truncation"]
    gpu = runtime.get("gpu") or {}
    library = runtime.get("library") or {}
    revision = model["revision_resolved"] or model["revision_requested"] or "tidak dipin"

    rows = [
        _row("Nama dan tag", f"`{model['id']}` (`{model['tag']}` @ `{revision}`)"),
        _row(
            "Digest",
            f"`sha256:{digest['sha256']}` ({digest['file']})" if digest["sha256"] else None,
        ),
        _row("Arsitektur", identity["architecture"]),
        _row("Jumlah parameter", f"{identity['parameters']:,}" if identity["parameters"] else None),
        _row(
            "Kuantisasi",
            f"bobot {quantization['stored']}, inferensi {quantization['inference']}",
        ),
        _row("Dimensi embedding", identity["dimension"]),
        _row(
            "Panjang konteks",
            f"model {context['model_max_tokens']} token (config); dipakai {context['used_max_tokens']} token",
        ),
        _row("Format file", "PyTorch " + (digest["file"].rsplit(".", 1)[-1] if digest["file"] else "?" )),
        _row("Lisensi", identity["license"]),
        _row("Pooling", f"{identity['pooling']}, L2-normalised" if identity["normalize"] else identity["pooling"]),
        _row("Tokenizer", f"`{context['tokenizer']}`"),
        _row(
            "Perangkat",
            f"{gpu.get('name', runtime.get('device', 'cpu'))}"
            + (f" ({gpu['total_memory_gib']} GiB)" if gpu.get("total_memory_gib") else ""),
        ),
        _row(
            "Runtime",
            ", ".join(f"{key} {value}" for key, value in sorted(library.items()) if value),
        ),
    ]
    if truncation.get("chunks") is None:
        rows.append(_row("Chunk terpotong", "belum diukur (jalankan `pokerag-index build`)"))
    else:
        rows.append(
            _row(
                "Chunk terpotong",
                f"{truncation.get('chunks_over_window', 0)} dari {truncation.get('chunks', 0)} "
                f"(maksimum {truncation.get('max_tokens', 0)} token, batas {truncation.get('max_seq_length')})",
            )
        )

    return "\n".join(
        [
            f"# Identitas model embedding — {model['id']}",
            "",
            "> Dibuat otomatis oleh `pokerag-index build`. Jangan diedit tangan:",
            "> sumbernya `artifacts/models/embedder_identity.json`, yang sha256-nya",
            "> tercatat di manifest setiap koleksi.",
            "",
            "| Field | Nilai |",
            "|---|---|",
            *rows,
            "",
            "## Cara membacanya",
            "",
            "* **Digest** adalah SHA-256 berkas bobot pada revisi yang dipin, bukan",
            "  hash direktori unduhan. Revisi dipin di config (`embedder.revision`),",
            "  sehingga unduhan ulang tidak bisa menukar bobot di balik nama model.",
            "* **Kuantisasi**: bobot di repositori disimpan pada presisi tertentu",
            "  (dihitung dari byte/parameter), lalu dimuat ke VRAM sebagai",
            "  `dtype_inference`. Keduanya dicatat karena keduanya memengaruhi vektor.",
            "* **Panjang konteks**: `model_max_tokens` diambil dari",
            "  `config.max_position_embeddings` model (kartu model biasanya menyebut",
            "  8192; selisih 2 posisi disediakan untuk penanda khusus). Build hanya",
            "  memakai `used_max_tokens`, dan chunk yang melewati batas itu akan",
            "  dipotong sebelum di-embed, karena itu jumlah chunk terpotong wajib 0.",
            "",
            "## Batas yang tidak diklaim",
            "",
            "Tabel ini menyatakan identitas model dan konfigurasi embedding. Ia tidak",
            "menyatakan kualitas retrieval; itu diukur terpisah di laporan A1.",
            "",
        ]
    )
