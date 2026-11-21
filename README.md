# PokeRAG

Retrieval-grounded Pokemon question answering for the NLP Frontier LLM project.
The corpus is frozen from Bulbapedia (386 documents, National Dex 1-386) and the
retrieval index is built once, outside this repository, then imported here.

## Status

| Milestone | Isi | Status |
| --- | --- | --- |
| M0 | import artefak, verifikasi hash, cek tiga koleksi | selesai |
| M1 | retrieval BM25 + dense + fusion | belum |
| M2 | harness evaluasi qrels | belum |
| M3 | B0-B2 | belum |
| M4 | B3-B4 | belum |
| M5 | B5-B6, red-team | belum |
| M6 | runs, serving, laporan | belum |

## Artefak

Sumber: `meivawijaya/pokerag-corpus-v3` (Hugging Face).

| Komponen | Nilai |
| --- | --- |
| Snapshot | `5eda7c8efc03f6fe1e892289093e04912854aa8addc28ead50cac700dc583e1c` |
| Dokumen | 386 |
| Embedding | `BAAI/bge-m3`, 1024 dimensi, CLS, L2-normalised |
| Chunk | fixed 11.642, heading 27.902, hierarkis 28.786 |
| Vector store | Qdrant local mode, sqlite-backed |
| BM25 | `bm25s` 0.3.16, metode Lucene |

Artefak tidak pernah masuk git. Semua unduhan diverifikasi terhadap
`INTEGRITY.sha256` sebelum dipakai.

## Setup

Butuh Python 3.12 dan [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/fetch_artifacts.py --extract
uv run python scripts/verify_index.py
```

Perintah pertama mengunduh sekitar 330 MB: `indexes.tar.gz`, manifest, chunk,
dan `documents.parquet`. `--all` mengambil seluruh dataset, termasuk raw HTML
dan cache embedding.

Keluaran yang diharapkan:

```
fixed          chunks_fixed          11642   11642   ok
heading        chunks_heading        27902   27902   ok
hierarchical   chunks_hierarchical   28786   28786   ok
```

## Dokumen

- `docs/reports/M0-import.md` untuk hasil verifikasi artefak.
- `docs/adr/0001-frozen-artifacts-and-local-qdrant.md` untuk alasan memakai
  artefak beku dan Qdrant local mode.

## Batasan lisensi

Seluruh 386 dokumen mengizinkan retrieval, tetapi `allow_redistribute` dan
`allow_ft_training` bernilai false. Teks tidak boleh masuk repo maupun image
publik, dan B5 (LoRA/QLoRA) memerlukan dataset berlisensi terpisah atau
dinyatakan tidak dijalankan.

## Struktur

```
src/pokerag/
  core/        config, integrity
  ingest/      download artefak dari Hugging Face
  index/       vector store dan BM25
  stages/      planner, retrieval, sufficiency, generator, verifier
  pipelines/   B0 sampai B6
  eval/        metrik dan harness
  api/         FastAPI
scripts/       entry point operasional
configs/       konfigurasi model dan eksperimen
docs/          ADR, laporan, legal
```

## Lisensi data

Teks berasal dari Bulbapedia, CC BY-NC-SA 2.5. Setiap dokumen menyimpan URL,
`pageid`, `revid`, dan hash di `register.parquet`. Penggunaan non-komersial,
atribusi wajib, dan derivatif tetap CC BY-NC-SA 2.5. Repositori ini tidak
mendistribusikan teks mentah; artefak diunduh dari dataset sumber.
