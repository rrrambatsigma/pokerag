# PokéRAG

**Evidence-Grounded Pokémon Knowledge Assistant Using Hybrid and Adaptive Retrieval**

Status: **tahap 3 — retrieval hybrid (dense + BM25)** plus UI demo (mock); generator LLM dan evaluasi generatif belum ada.

## Apa yang sudah ada

Pipeline korpus yang deterministik dan bisa diaudit:

```
select -> fetch -> clean -> dedup -> validate -> freeze
```

* `select` — memilih 386 halaman spesies dari National Pokédex 1–386 (Gen I–III lengkap, bukan alfabetis).
* `fetch` — wikitext + HTML yang dipin ke satu `revid`, disimpan mentah, bisa di-resume.
* `clean` — HTML → teks bersih dengan section dan offset karakter (dasar chunking & qrels).
* `dedup` — exact hash + MinHash near-duplicate, dengan laporan keputusan.
* `validate` — 10 gate, hasilnya ditulis sebagai tabel, bukan klaim.
* `freeze` — `documents.parquet`, `register.parquet`, `snapshot_id` berbasis hash.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"      # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # Linux/WSL

set POKERAG_CONTACT_EMAIL=kamu@kampus.ac.id   # dipakai di User-Agent

pokerag-ingest select
pokerag-ingest fetch          # resumable, ~39 menit untuk 386 halaman (crawl-delay 5 s)
pokerag-ingest clean
pokerag-ingest dedup
pokerag-ingest validate
pokerag-ingest freeze
```

Tes: `pytest` (54 tes, termasuk invariant offset, uji politeness tanpa jaringan,
uji idempotensi fetch/clean/freeze, uji index: tiga koleksi dalam satu
database, cache terikat identitas embedder, gate pemotongan chunk, serta uji
retrieval: RRF, guard index basi, isolasi mode, dan metrik evaluasi).

## Bootstrap dari artefak beku

Kalau hanya perlu index yang sudah jadi, misalnya di mesin baru, CI, atau image
API, unduh artefak alih-alih menjalankan seluruh pipeline:

```bash
python scripts/fetch_artifacts.py --extract
```

Skrip mengambil 28 file dari `meivawijaya/pokerag-corpus-v3` (Qdrant, BM25,
chunk, manifest), memverifikasi setiap file terhadap `INTEGRITY.sha256`, lalu
mengekstrak `indexes/` ke `artifacts/`. Ukuran sekitar 330 MB dan tidak butuh
GPU. Build tetap sumber kebenaran; ini hanya jalur cepat untuk mengonsumsi hasil
yang sudah dibekukan.

## Idempotensi

Semua stage aman dijalankan berulang:

| Stage | Run kedua |
|---|---|
| `fetch` | 0 request jaringan, index ditulis ulang byte-identik |
| `clean` | 0 dokumen ditulis ulang (versi + hash konfigurasi dicocokkan) |
| `dedup` / `validate` | laporan identik, urutan `doc_id` |
| `freeze` | `snapshot_id` dan hash parquet identik |

Bukti bisa diperiksa sendiri:

```bash
pokerag-ingest freeze && pokerag-ingest verify
```

`verify` menghitung ulang sha256 setiap file konten dan `snapshot_id` dari
`documents.parquet`; ia gagal bila satu byte berubah. Jam dinding dan commit git
disimpan terpisah di `provenance.json` agar tidak menggeser hash.

## Tahap 2 — chunking dan index

```bash
pokerag-chunk build                 # fixed, heading, hierarchical
pokerag-index identity              # tabel identitas embedder (dipakai build juga)
pokerag-index build                 # 3 koleksi Qdrant dalam satu database
pokerag-index search "which ability does Charizard have" --strategy heading
```

| Artefak | Isi |
|---|---|
| `chunks/<strategi>.parquet` | unit retrieval dengan `char_start`/`char_end` |
| `chunks_report.md` | jumlah chunk, p95, over-budget, cakupan |
| `docs/experiments/A1-index.md` | tabel A1 (di-commit): ukuran index, token, context length |
| `docs/models/embedding-BAAI-bge-m3.md` | identitas embedder (di-commit) |
| `artifacts/indexes/qdrant/` | satu database: `chunks_fixed`, `chunks_heading`, `chunks_hierarchical` |
| `artifacts/indexes/bm25/<strategi>/` | BM25 per koleksi, untuk B3 |
| `artifacts/indexes/<strategi>/manifest.json` | `index_version`, jumlah titik, checksum vektor |
| `artifacts/indexes/provenance.json` | waktu build, jumlah embed baru (tidak ikut hash) |
| `artifacts/embeddings/*.parquet` | cache vektor per identitas embedder |

Embedder dipin: `BAAI/bge-m3` revisi `5617a9f6…`, 1024 dim, fp16 di CUDA,
`max_seq_length: 1024`. Tiga strategi memakai fungsi embedding yang sama, jadi
perbandingan A1 hanya mengubah unit retrieval.

Batas 1024 token bukan tebakan: dengan tokenizer bge-m3, chunk `fixed` 2048
karakter adalah 597 token (p50) dan 1.005 (max). Pada 512 token, 95% chunk fixed
dipotong sebelum di-embed. Karena itu build **gagal sebelum embedding** bila ada
chunk melewati jendela (`--allow-truncation` untuk kasus yang sengaja).

Cache diikat ke identitas embedding (model, revisi, dimensi, dtype,
`max_seq_length`), sehingga build ulang dengan input sama = 0 embedding, dan
mengubah salah satu parameter tidak pernah memakai vektor lama. Terukur di
RTX 3050 4 GB: 15–32 chunk/detik, puncak VRAM 1,43 GiB. Setelah build selesai
proses keluar dan VRAM kosong.

### Korpus v2 — Gen I–III lengkap

Snapshot v1 (320 dokumen) memotong Gen III di #320. v2 menambah 66 spesies
#321–386, tetapi **bukan dengan mengambil ulang 386 halaman**: raw HTML/wikitext
v1 disalin ke `artifacts/corpus/pokerag-corpus-v2` dengan `revid` yang sama, lalu
`fetch` hanya mengambil 66 halaman baru. Jadi v2 adalah superset ketat v1 — satu-
satunya variabel yang berubah adalah ukuran korpus, bukan revisi halaman lama.

| | v1 | v2 |
|---|---|---|
| Dokumen | 320 | 386 |
| Unit retrieval (3 strategi) | 58.098 | 68.330 |
| Embedding baru saat build | 58.098 | 10.232 (sisanya cache) |
| Ukuran index (Qdrant + BM25) | ±602 MB | ±708 MB |
| `snapshot_id` | `5f898e4f…` | `5eda7c8e…` |

Chunking dan embedder **tidak diubah** (parameter sama, `bge-m3` sama), supaya
A1 tetap membandingkan metode chunking, bukan campuran korpus dan setelan.
Build v2 butuh 905 detik (223,5 + 329,9 + 351,3) karena hanya 10.232 chunk baru
yang di-embed. Index v1 diarsipkan ke `artifacts/archive/indexes-v1`; angka A1/B1
v1 tetap ada di riwayat git dan backup HF.

`index_version` mencakup snapshot, chunk, strategi, identitas embedder, metrik
jarak dan parameter BM25; waktu build dan jumlah embed baru disimpan terpisah
di `provenance.json` supaya manifest tetap deterministik.

### Varian index

| Config | Embedder | Dim | Koleksi | Catatan |
|---|---|---|---|---|
| `index.gpu.v2.yaml` | `bge-m3` (CUDA, fp16, 1024 token) | 1024 | 3 | jalur utama, default CLI, snapshot v2 |
| `index.gpu.yaml` | `bge-m3` | 1024 | 3 | arsip v1 (320 dokumen) |
| `index.local.yaml` | `bge-small` (ONNX, CPU) | 384 | heading | fallback tanpa GPU |
| `index.v1.yaml` | Gemini API | 768 | heading | arsip; kuota free tier habis |

Ketiganya berbagi database yang sama tetapi nama koleksinya tetap per strategi;
mengganti embedder pada koleksi yang sudah ada **ditolak** tanpa `--force`.

### Backup

`python scripts/upload_hf.py --version v3` mengunggah keadaan penuh (satu
`indexes.tar.gz` berisi tiga koleksi, manifest, identitas embedder, laporan A1)
ke dataset privat `meivawijaya/pokerag-corpus-v3`. Dataset v2 menyimpan keadaan
korpus v1 (320 dokumen) dan tetap utuh sebagai arsip.

## Tahap 3 — retrieval hybrid

Dua index dari tahap 2 dipakai apa adanya: dense di Qdrant, BM25 di `bm25s`.
Tidak ada yang dibangun ulang; yang baru adalah lapisan query yang menggabungkan
keduanya (ADR-0009).

```bash
pokerag-retrieve check                       # manifest, jumlah titik, urutan BM25
pokerag-retrieve search "which ability does Charizard have" --mode hybrid
pokerag-retrieve search "..." --mode bm25 --json
pokerag-retrieve eval --qrels data/benchmark/qrels/dev.jsonl --strategy heading
```

Fusion memakai **reciprocal-rank fusion**, bukan penjumlahan skor:

```
skor(chunk) = Σ bobot_sumber / (rrf_k + rank_sumber)
```

Cosine (`0..1`) dan BM25 (tak dibatasi) tidak pernah dijumlahkan; hanya
peringkatnya yang dipakai. Default `rrf_k: 10`, bobot dense 2 : BM25 1.

| Artefak | Isi |
|---|---|
| `configs/retrieval.v1.yaml` | index yang dibaca, kandidat, `rrf_k`, bobot |
| `src/pokerag/retrieval/` | dense, BM25, fusion, guard, pipeline, evaluasi, cli |
| `data/benchmark/qrels/dev.jsonl` | 30 query dev, satu dokumen relevan per query |
| `docs/experiments/B1-retrieval.md` | tabel B1 (di-commit): recall/MRR/nDCG per mode |
| `artifacts/reports/retrieval.json` | angka + daftar hasil per query (tidak di-commit) |

Baseline B1 (strategi `heading`, 30 query, korpus v2, index `fa5a8dafad1e`):

| Mode | recall@5 | MRR@5 | nDCG@5 | span_hit@5 |
|---|---|---|---|---|
| dense | 1,000 | 0,907 | 0,930 | 0,933 |
| BM25 | 0,967 | 0,758 | 0,810 | 0,633 |
| hybrid | 1,000 | 0,893 | 0,919 | 0,933 |

Hybrid menyamakan recall dense dan mengalahkan BM25, tetapi **belum mengalahkan
dense pada MRR/nDCG**. Itu dilaporkan sebagai temuan, bukan disembunyikan;
dengan satu dokumen relevan per query, fusion menambah noise peringkat.
`rrf_k=60` dan bobot 1:1 bahkan menurunkan recall@5 ke 0,967.

Guard sebelum query: identitas embedder dan hash chunk harus cocok dengan
manifest, jumlah titik Qdrant = jumlah baris parquet, dan urutan
`bm25/chunk_ids.json` = urutan `chunk_id` di parquet (BM25 memetakan hasil lewat
nomor baris, jadi urutan yang bergeser akan menghasilkan sitasi salah tanpa
error). Melanggar salah satunya = error, bukan ranking yang terdengar masuk akal.

## Benchmark test (qrels anotator)

100 query test disusun terpisah di repo anotator, beserta ekspor tiga
anotator. Adapter menggabungkan ekspor itu menjadi gold qrels ber-offset
karakter:

```bash
pokerag-eval build \
  --export A1=exports/a1.jsonl \
  --export A2=exports/a2.jsonl \
  --export A3=exports/a3.jsonl \
  --queries queries.csv \
  --policy union \
  --out data/benchmark/qrels/test.jsonl

pokerag-eval score --strategy heading --modes dense,bm25,hybrid --k 5,10
```

Kebijakan penggabungan: `single` (satu anotator, default selama belum ada
adjudication), `union` (semua span yang ditandai siapa pun), dan `intersection`
(hanya irisan). Query multi-hop menyumbang lebih dari satu dokumen gold;
`doc_recall` dihitung sebagai fraksi seluruh dokumen gold. `span_hit` menuntut
chunk beririsan dengan rentang karakter gold, bukan sekadar dokumen yang sama.

Qrels test tidak di-commit selama benchmark belum dirilis: `data/benchmark/qrels/test*.jsonl`
masuk `.gitignore`.

## API

UI memakai proxy Vite (`/api` → `127.0.0.1:8000`), jadi cukup jalankan dua
proses:

```bash
pokerag-api --port 8000
cd frontend && npm run dev
```

| Endpoint | Isi |
|---|---|
| `GET /api/health` | snapshot, `index_version`, strategi, generator, mode |
| `GET /api/search?q=&mode=&strategy=&k=` | sitasi hasil retrieval |
| `POST /api/chat` | `{message, settings}` → satu pesan asisten dengan sitasi |

Settings UI dipetakan ke mode: dense+bm25 = `hybrid`, salah satu = `dense` atau
`bm25`, keduanya mati = `off` (tanpa retrieval, posisi B0). Selama generator
Qwen belum ada, `StubGenerator` mengembalikan jawaban deterministik dengan
sitasi dari hasil retrieval. Kontraknya sama dengan generator asli.

## UI demo — chat, sitasi, setelan

`frontend/` berisi antarmuka React (Vite + TypeScript) bergaya panel GBA/Pokédex.
UI ini masih memakai **data mock**; belum tersambung ke API retrieval.

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # tsc --noEmit && vite build
```

| Bagian | Yang sudah ada |
|---|---|
| Chat | bubble pertanyaan/jawaban, indikator mengetik, saran pertanyaan saat kosong |
| Sitasi | badge angka, kartu sumber dengan nomor dex + tipe, skor, panel entri lengkap + tautan Bulbapedia |
| Setelan | toggle dense/BM25 (mode HYBRID/MAKNA/KATA KUNCI/MATI), strategi chunk, jumlah sumber |
| Riwayat | daftar percakapan, pilih, hapus, chat baru; disimpan di `localStorage` |
| Responsif | sidebar riwayat menjadi drawer di layar kecil |

Jawaban mock ada di `frontend/src/mock/data.ts` (Charizard, Eevee, Zapdos,
Venusaur). Kontrak API yang dituju ada di `frontend/README.md`; tipe `Citation`
di `frontend/src/types.ts` sudah mengikuti payload retrieval (judul, dex, tipe,
heading, kutipan, URL, skor).

## Etika dan lisensi data

Teks Bulbapedia berlisensi **CC BY-NC-SA 2.5**. Konsekuensinya dikodekan, bukan
hanya ditulis:

* `allow_redistribute = false` → teks korpus masuk `.gitignore`, tidak pernah di-commit.
* `allow_ft_training = false` → gate B5 nanti harus memakai dataset berlisensi lain.
* Setiap dokumen menyimpan URL, `revid`, timestamp, `sha1` dari API, `sha256` teks bersih, dan atribusi.
* Crawler mematuhi `Crawl-delay: 5` dan berhenti (bukan mengakali) kalau mendapat HTTP 403.

Rincian: `docs/legal/data-license.md`.

## Struktur

```
configs/corpus.v1.yaml     seleksi v1 (320 dokumen) - arsip
configs/corpus.v2.yaml     seleksi v2 (386 dokumen, Gen I-III)
configs/chunking.v*.yaml   tiga strategi, tokenizer model, budget token
configs/index.*.yaml       embedder, database, BM25
configs/retrieval.v1.yaml  index yang dibaca, kandidat, RRF, bobot
src/pokerag/core/          hashing, JSONL, logging
src/pokerag/ingest/        select, fetch, clean, dedup, validate, freeze, cli
src/pokerag/chunking/      fixed, heading, hierarchical, tokenizer, cli
src/pokerag/index/         build, embedding cache, Qdrant, BM25, identitas, cli
src/pokerag/retrieval/     dense, BM25, RRF, guard, pipeline, evaluasi, cli
frontend/                  UI React (Vite): chat, sitasi, setelan, riwayat (mock)
data/benchmark/qrels/      qrels dev (di-commit, dinilai di level dokumen)
tests/unit/                offset, dedup, politeness, seleksi, lisensi, index, retrieval
artifacts/corpus/<name>/   artefak (diabaikan git), hanya manifest yang di-commit
docs/adr/                  keputusan teknis
docs/experiments/          tabel A1 dan B1 (di-commit, hasil run)
docs/models/               identitas embedder (di-commit, hasil build)
```

## Keputusan yang masih terbuka

* Sumber data untuk LoRA B5 (korpus ini NC, jadi tidak boleh dipakai untuk training).
* Base generator B0–B5 (3B vs 1.5B) — memengaruhi seluruh rangkaian eksperimen.
