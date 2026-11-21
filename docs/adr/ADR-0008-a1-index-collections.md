# ADR-0008 — A1 di satu database, tiga koleksi, satu identitas embedder

**Status:** diterima
**Tanggal:** 2026-11-17

## Konteks

A1 membandingkan tiga metode chunking pada **dokumen sumber yang sama**. Supaya
perbandingannya bersih, ketiganya harus memakai satu model embedding, satu
snapshot korpus, dan satu metrik jarak; yang boleh berbeda hanya unit
retrieval-nya. Tiga hal menuntut keputusan eksplisit:

1. Qdrant lokal sebelumnya menyimpan satu database per `index_version`, masing-
   masing dengan koleksi bernama `chunks`. Itu membuat "tiga metode" tidak
   terlihat sebagai tiga koleksi, dan menyulitkan pemakaian bersama saat
   retrieval serta backup.
2. Menjalankan tiga metode berarti tiga kali pass embedding. Pertanyaannya:
   apakah pass itu bisa dihindari, dan berapa biayanya.
3. Identitas model embedding harus dicatat sebagai fakta yang bisa diperiksa
   (nama, digest, arsitektur, parameter, kuantisasi, dimensi, konteks, format,
   lisensi), bukan sebagai asumsi dari nama model.

Saat mengukur, muncul masalah keempat: dengan tokenizer model yang sebenarnya
(`BAAI/bge-m3`), chunk `fixed` 2048 karakter adalah **597 token di p50 dan 1.005
di maksimum**, bukan 512 seperti perkiraan `chars/4`.

| Strategi | token p50 / p95 / max | unit > 512 token |
|---|---|---|
| `fixed` | 597 / 782 / 1.005 | 9.431 dari 9.887 (95%) |
| `heading` | 302 / 445 / 627 | 426 dari 23.767 (1,8%) |
| `hierarchical` (child) | 241 / 363 / 512 | 0 dari 24.444 |

Pada `max_seq_length: 512`, sentence-transformers memotong 95% chunk `fixed`
sebelum di-embed, sementara `hierarchical` tidak terpotong sama sekali. Ablasi
A1 akan melaporkan "fixed kalah" padahal penyebabnya konfigurasi embedder, bukan
metode chunking.

## Keputusan

1. **Satu database, tiga koleksi.** Semua strategi di satu Qdrant lokal
   (`artifacts/indexes/qdrant`) dengan koleksi `chunks_fixed`,
   `chunks_heading`, `chunks_hierarchical`. BM25 tetap terpisah per strategi
   (`artifacts/indexes/bm25/<strategi>`) karena B3 memfusikan dua ranking nyata.
2. **Satu embedder untuk ketiganya**: `BAAI/bge-m3`, revisi dipin
   (`5617a9f6…`), fp16 di CUDA, `max_seq_length: 1024`. Batas ini dipilih dari
   pengukuran: 1.005 token adalah chunk terpanjang, jadi 1.024 menjamin nol
   pemotongan. Build **gagal lebih awal** (sebelum embedding) bila ada chunk
   melewati jendela; `--allow-truncation` hanya untuk kasus yang sengaja dan
   wajib dilaporkan.
3. **Tiga pass embedding memang diperlukan**, karena teks chunk berbeda. Yang
   dihemat adalah model load (satu proses, tiga strategi) dan pass yang sudah
   ada di cache. Dedupe lintas strategi diukur dan tidak menolong: 58.098 unit
   menghasilkan 56.990 teks unik (1,9%).
4. **Ditolak: mean-pooling / late chunking** untuk "sekali embed". Vektor chunk
   tidak lagi berasal dari teks chunk itu sendiri, sehingga delta antar strategi
   mencampur efek chunking dengan efek komposisi vektor. Itu merusak A1 dan
   tidak boleh dipakai untuk klaim kausal.
5. **Cache diikat ke identitas fungsi embedding.** Nama file memuat hash
   `model + revisi + dimensi + dtype + max_seq_length + prefix`. Sebelumnya
   `max_seq_length` tidak ikut di kunci, sehingga vektor 512 token bisa dipakai
   diam-diam untuk build 1024 token. Flush dilakukan tiap 2.000 chunk atau 60
   detik, bukan tiap batch: menulis ulang parquet yang tumbuh untuk setiap batch
   adalah O(n²) byte dan membuat GPU menganggur.
6. **Identitas embedder dicatat, bukan diasumsikan.** `pokerag-index build`
   menulis `artifacts/models/embedder_identity.json` (dari config, model yang
   benar-benar dimuat, dan metadata Hugging Face) serta tabel yang di-commit di
   `docs/models/embedding-BAAI-bge-m3.md`. SHA-256 dokumen identitas masuk ke
   manifest setiap koleksi.
7. **Artefak yang di-hash tetap deterministik.** Waktu build, jumlah embed baru,
   ukuran index, dan timestamp hidup di `artifacts/indexes/provenance.json`,
   bukan di manifest; dengan input sama, manifest tetap byte-identik.
8. **Laporan A1 disalin ke repo.** `artifacts/**` di-ignore git, sehingga tabel
   A1 dan identitas model tidak akan terlihat penguji. Karena itu build menulis
   salinan yang di-commit: `docs/experiments/A1-index.md` dan
   `docs/models/embedding-<model>.md`.
9. **Backup naik ke repo v2** (`meivawijaya/pokerag-corpus-v2`, private) berisi
   satu `indexes.tar.gz` dengan tiga koleksi, manifest per koleksi, identitas
   embedder, dan `INTEGRITY.sha256`. Repo v1 tetap utuh sebagai arsip keadaan
   sebelumnya.

## Konsekuensi

* Build penuh ≈58.098 embedding. Terukur pada RTX 3050 4 GB: 15–32 chunk/detik
  (tergantung panjang chunk), jadi **±41 menit**, dengan puncak VRAM 1,43 GiB.
  Generator tidak boleh jalan bersamaan.
* Cache tumbuh ke ±250 MB, Qdrant ±550 MB, total artefak ±2,2 GB. Backup v2
  ±1,2 GB.
* Build ulang dengan input yang sama = **0 embedding**; mengubah salah satu
  parameter identitas (revisi, dimensi, dtype, `max_seq_length`) otomatis
  memakai file cache lain dan tidak mencampur vektor lama.
* Mengganti embedder pada koleksi yang sudah ada **ditolak** tanpa `--force`,
  karena Qdrant lokal menyisakan direktori koleksi lama saat dihapus (bug yang
  juga ditangani dengan menghapus direktorinya setelah client ditutup).
* `context length` A1 dapat dilaporkan dengan angka nyata: pada k=5, `fixed`
  membawa ±2.985 token ke generator, `heading` ±1.510, `hierarchical` ±1.205.
* Passage recall dan groundedness **belum** diukur; keduanya ditulis sebagai
  tertunda beserta definisinya di `docs/experiments/A1-index.md`, bukan
  dikosongkan tanpa penjelasan.
* Audit chunking (`pokerag-chunk report`) menemukan bahwa versi pertama
  `hierarchical` memotong batas h2 (parent dibangun dari jendela seluruh
  dokumen). Temuan, perbaikan, dan angka barunya dicatat di Adendum ADR-0005;
  koleksi `hierarchical` dibangun ulang, `fixed` dan `heading` tidak tersentuh.
