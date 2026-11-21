# ADR-0010 — Korpus v2: Gen I–III lengkap (386 dokumen)

**Status:** diterima
**Tanggal:** 2026-11-21

## Konteks

Seleksi v1 memakai National Pokédex **1–320** (`configs/corpus.v1.yaml`). Gen III
bernomor **#252–386**, jadi v1 hanya memuat #252–320 (69 spesies) dan memotong
66 spesies terakhir. Untuk klaim "Gen I–III" pada laporan, korpusnya harus
lengkap: 151 + 100 + 135 = **386** halaman spesies.

Ada dua hal yang tidak boleh berubah bersamaan dengan perubahan korpus:

1. **Parameter chunking dan embedder.** Kalau ikut diubah, A1 tidak lagi
   membandingkan metode chunking; ia membandingkan campuran korpus dan setelan.
2. **Revisi halaman lama.** Mengambil ulang 386 halaman dari Bulbapedia akan
   mem-pin `revid` terbaru, sehingga 320 dokumen v1 bisa berubah isi. Delta A1
   lalu mencampur efek "korpus bertambah" dengan efek "halaman disunting".

## Keputusan

1. **Snapshot baru `pokerag-corpus-v2`, 386 dokumen.** Range `to_dex: 386`.
   `artifacts/corpus/pokerag-corpus-v1` tidak disentuh.
2. **Superset ketat, bukan refetch.** Raw HTML/wikitext 320 dokumen v1 disalin
   ke direktori v2 dan path di `pages_index.jsonl` + `html_index.jsonl` ditulis
   ulang (`pokerag-corpus-v1` → `pokerag-corpus-v2`). `fetch` lalu melihat 320
   halaman cached dan hanya mengambil **66 halaman baru** (80 request, ±7 menit
   dengan Crawl-delay 5 s). `revid` dokumen lama tetap persis.
3. **Chunking dan embedder tidak berubah.** `configs/chunking.v2.yaml` memakai
   parameter identik v1; `configs/index.gpu.v2.yaml` memakai `BAAI/bge-m3`
   revisi dan `max_seq_length` yang sama. Batch dan dtype sama (fp16, CUDA).
4. **Index v1 diarsipkan, bukan dihapus.** Sebelum build v2,
   `artifacts/indexes` dipindah ke `artifacts/archive/indexes-v1`. Cache
   embedding (`artifacts/embeddings`) tetap di tempatnya, sehingga chunk lama
   yang teksnya tidak berubah tidak di-embed ulang.
5. **Nama snapshot ikut masuk `index_version`,** jadi ketiga koleksi dibangun
   ulang (Qdrant delete + recreate, BM25 rebuild). Yang dihemat hanya embedding,
   bukan upsert.

## Hasil

| | v1 | v2 |
|---|---|---|
| Dokumen | 320 | 386 |
| `snapshot_id` | `5f898e4f…` | `5eda7c8e…` |
| `fixed` | 9.887 | 11.642 |
| `heading` | 23.767 | 27.902 |
| `hierarchical` (unit) | 24.444 | 28.786 (+15.803 parent) |
| Embedding baru | 58.098 | 10.232 (47.866 dari cache) |
| Build index | 667 + 902 + 894 s | 223,5 + 329,9 + 351,3 s |
| Index (Qdrant + BM25) | ±602 MB | ±708 MB |

Validasi: 386/386 lolos 10 gate, `dedup` 0 duplikat, gap crawl minimum 5,00 s
dari 461 request. Cakupan teks chunk: `fixed` 100%, `heading` 99,8%,
`hierarchical` 99,9% (v1 100%; selisihnya chunk <80 karakter yang dibuang,
mis. baris tabel pada halaman baru).

B1 v2 (30 query, `heading`, index `fa5a8dafad1e`) hampir sama dengan v1: dense
tidak berubah (vektor chunk lama identik), BM25 naik tipis (MRR@5 0,741 → 0,758)
karena statistik IDF korpus berubah. Hybrid tetap 1,000 recall@5 dan 0,893
MRR@5.

## Konsekuensi

* Halaman baru hanya 66, tetapi semuanya masuk tiga strategi; total unit naik
  ±18%. Waktu build turun dari ±41 menit (v1 dari nol) menjadi ±15 menit karena
  cache, bukan karena korpusnya lebih kecil.
* Qrels dev tetap valid (semua dokumen lama masih ada), tetapi baru 2 dari 30
  query berasal dari Gen III. Menambah 10–15 query #321–386 adalah pekerjaan
  berikutnya sebelum angka B1 dipakai untuk klaim akhir.
* A1/B1 v1 tetap bisa dirujuk lewat riwayat git; index v1 ada di
  `artifacts/archive/indexes-v1` dan backup HF (`pokerag-corpus-v2` dataset).
* `configs/index.gpu.yaml` (v1) masih ada sebagai arsip; menjalankannya akan
  membangun ulang index v1, bukan memakai yang sudah diarsipkan.
* Backup HF sudah diunggah: `meivawijaya/pokerag-corpus-v3` (privat, 805 file,
  805,8 MB) berisi snapshot v2, chunk, cache embedding, raw 386+386, tiga
  manifest index, dan `indexes.tar.gz` (286,2 MB). Manifest unggahan lokal:
  `artifacts/upload_manifest_v3.json`.
