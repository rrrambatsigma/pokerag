# ADR-0006 — Embedding hosted (Gemini) untuk index dense

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Rencana awal memakai `BAAI/bge-m3` lokal. Dua kendala muncul saat diukur:

* 4,84 juta token untuk 23.767 chunk; di CPU (mesin ini `torch` masih CPU-only)
  itu 6–10 jam, dan RTX 3050 4 GB harus dipakai bergantian dengan model lain;
* kualitas `bge-m3` bagus, tetapi butuh unduhan 2,2 GB dan penyesuaian fp16.

Gemini tersedia dan sudah diuji langsung: `gemini-embedding-001`,
`batchEmbedContents` menerima 100 teks per panggilan, `outputDimensionality`
768 menghasilkan vektor 768-dim.

## Keputusan

Pakai **`gemini-embedding-001`, 768 dimensi**, dengan pemisahan `taskType`:
`RETRIEVAL_DOCUMENT` saat mengindeks dan `RETRIEVAL_QUERY` saat mencari.

Ini substitusi **SETARA** terhadap `bge-m3`: sama-sama model embedding dense,
hanya lokasi eksekusinya yang berpindah. Yang berubah adalah dimensi
(768 vs 1024), sehingga index harus dibangun ulang — tidak bisa dicampur.

## Pengaman yang menyertainya

1. **Cache wajib** (`artifacts/embeddings/<provider>-<model>-<dim>.parquet`)
   dengan kunci `chunk_id` + hash teks. Build ulang = 0 panggilan API; teks yang
   berubah otomatis di-embed ulang, bukan memakai vektor basi.
2. **Model dipin di manifest**, termasuk dimensi dan `taskType`. Kalau penyedia
   mengganti model di balik nama yang sama, `vectors_sha256` dan
   `index_version` berubah dan itu terlihat.
3. **API key** dibaca dari environment variable, tidak pernah ditulis ke repo
   atau log.
4. **Data governance**: hanya teks artikel Bulbapedia (publik, CC BY-NC-SA,
   tanpa PII) yang dikirim. Tidak ada dokumen sensitif.
5. **Vektor dinormalisasi** dan disimpan dengan cosine distance, sehingga
   perbandingan tidak terpengaruh skala panjang vektor.

## Konsekuensi

* Bergantung pada kuota dan jaringan. **Free tier diukur per item, bukan per
  request**: respons 429 menyebut `embed_content_free_tier_requests, limit: 100`
  dan batch 100 item langsung memakai jatah satu menit. Karena itu pacing
  dihitung dari jumlah item (90/menit), dan 23.747 chunk memerlukan sekitar
  4,4 jam pada free tier.
* Kegagalan API ditangani retry terbatas (429/5xx) yang menghormati isyarat
  `retry in Xs`, lalu berhenti dengan error jelas — bukan menghasilkan index
  separuh.
* Cache di-flush setiap batch, sehingga run 4 jam bisa dihentikan kapan saja dan
  dilanjutkan tanpa mengulang pekerjaan.
* Biaya per build hanya sekali; reindex dari cache gratis.
* Jika kuota habis saat demo, jalur mundur yang disarankan adalah index
  `bge-small` lokal sebagai "degraded mode" yang dilaporkan, bukan diam-diam.
* Embedder tidak boleh diganti di tengah A2 atau B2–B6; satu model untuk semua.

## Adendum (2026-11-14): kembali ke GPU lokal

Free tier Gemini dibatasi **100 item per menit** dan dalam praktik tetap gagal
setelah ~500 item meski pacing 90/menit. Karena mesin ini punya RTX 3050 4 GB,
embedding dipindahkan ke GPU lokal dengan `BAAI/bge-m3` fp16 (1024 dim) melalui
`sentence-transformers`.

Yang berubah dari keputusan awal:

* `torch` di venv sebelumnya adalah wheel **CPU** (`+cpu`), sehingga
  `torch.cuda.is_available()` bernilai `False`. Setelah wheel `cu124` dipasang,
  GPU terdeteksi.
* Throughput terukur: **1.000 chunk ≈ 65 detik** (≈15 chunk/detik), sehingga
  23.767 chunk ≈25 menit. Ini bukan estimasi.
* Embedder dan generator tidak perlu aktif bersamaan: build selesai → proses
  keluar → VRAM kosong. Karena itu 4 GB cukup, dengan syarat urutan tersebut
  tidak dilanggar.
* Vektor Gemini yang sudah ter-cache (520 item) tetap disimpan; kalau kuota
  berbayar diaktifkan, build dapat dilanjutkan tanpa mengulang.

